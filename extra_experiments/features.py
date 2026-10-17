import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

# Features of every clip in all 13 versions, for the extra experiments.
# timbral_features and the rhythm part of rhythm_features are the same code as in the main notebooks,
# so the reference feature sets give the same numbers. New here: the shifted cyclic tempogram and the scale transform.

SR, HOP, WIN = 22050, 512, 384               # sampling rate, novelty hop (23 ms), tempogram window (8.9 s)
N_FFT, N_MFCC, N_MELS = 2048, 13, 128
M, TAU_0, CYCLIC_MIN, CYCLIC_MAX = 15, 60.0, 30.0, 480.0
TACTUS_MIN, TACTUS_MAX = 40.0, 240.0
N_SCALE, SCALE_MIN_LAG, SCALE_MAX_LAG, SCALE_POINTS = 20, 0.1, 4.0, 256   # lags in seconds

TIMBRAL_COLUMNS = [f'mfcc_{i:02d}_{statistic}' for i in range(N_MFCC) for statistic in ['mean', 'std']] \
    + ['zcr_mean', 'zcr_std', 'spectral_centroid_mean', 'spectral_centroid_std']
CYCLIC_COLUMNS = [f'cyclic_bin_{i}' for i in range(M)]
SHIFTED_COLUMNS = [f'shifted_bin_{i}' for i in range(M)]
RATIO_COLUMNS = ['R_2_4', 'R_3_4', 'R_4_4']
SCALE_COLUMNS = [f'scale_{i:02d}' for i in range(N_SCALE)]

RATES = [0.90, 0.95, 0.98, 1.02, 1.05, 1.10]


def make_condition(kind, rate):
    name = 'original' if kind == 'original' else f'{kind}_{rate:.2f}'
    tempo_change = 100*(rate - 1) if kind == 'stretch' else 0.0
    return {'condition': name, 'kind': kind, 'rate': rate, 'tempo_change_percent': round(tempo_change, 6)}


CONDITIONS = ([make_condition('original', 1.0)]
              + [make_condition('stretch', rate) for rate in RATES]
              + [make_condition('roundtrip', rate) for rate in RATES])
CONDITION_ORDER = [c['condition'] for c in CONDITIONS]


def change_tempo(y, rate):
    # plays y `rate` times as fast with the same pitch (phase vocoder, as in the main notebooks)
    if rate == 1.0:
        return y
    stretched = librosa.effects.time_stretch(y.astype(np.float64), rate=rate, n_fft=2048, hop_length=512)
    return stretched.astype(np.float32)


def apply_condition(y, condition):
    if condition['kind'] == 'original':
        return y
    if condition['kind'] == 'stretch':
        return change_tempo(y, condition['rate'])
    if condition['kind'] == 'roundtrip':
        return change_tempo(change_tempo(y, condition['rate']), 1.0/condition['rate'])
    raise ValueError(f'unknown condition: {condition}')


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < SR or np.max(np.abs(y)) == 0:
        raise ValueError('audio is too short or silent')
    return y


#--------------------------------------------------------------------------------
# Timbral features (same as Timbral_extraction.py)

def timbral_features(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    sc = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)[0]
    feats = {}
    for i in range(N_MFCC):
        feats[f'mfcc_{i:02d}_mean'] = float(np.mean(mfcc[i]))
        feats[f'mfcc_{i:02d}_std'] = float(np.std(mfcc[i]))
    feats['zcr_mean'], feats['zcr_std'] = float(np.mean(zcr)), float(np.std(zcr))
    feats['spectral_centroid_mean'], feats['spectral_centroid_std'] = float(np.mean(sc)), float(np.std(sc))
    return feats


#--------------------------------------------------------------------------------
# Rhythm features (same as the rhythm notebooks) plus the two tempo-free versions

def cyclic_tempogram(fourier_tempogram, tempo_axis):
    # 30-480 BPM on a log axis with M bins per octave, summed over the octaves (Muller 6.34),
    # averaged over time and normalised to length 1
    octave_min = int(np.floor(np.log2(CYCLIC_MIN/TAU_0)))
    octave_max = int(np.ceil(np.log2(CYCLIC_MAX/TAU_0)))
    n_octaves = octave_max - octave_min
    log_axis = TAU_0*2.0**(octave_min + np.arange(n_octaves*M)/M)
    order = np.argsort(tempo_axis)
    on_log_axis = interp1d(tempo_axis[order], fourier_tempogram[order], axis=0,
                           bounds_error=False, fill_value=0.0)(log_axis)
    cyclic = on_log_axis.reshape(n_octaves, M, -1).sum(axis=0).mean(axis=1)
    norm = np.linalg.norm(cyclic)
    return cyclic/norm if norm > 0 else cyclic


def shift_to_strongest_bin(cyclic):
    # rotate so that the strongest tempo class is always bin 0: the profile keeps its shape
    # but no longer says which tempo class the clip is in (Rythm_extraction.py with shift=True)
    return np.roll(cyclic, -int(np.argmax(cyclic)))


def estimate_tau_hat(fourier_tempogram, tempo_axis):
    # strongest tempo of the time-averaged Fourier tempogram between TACTUS_MIN and TACTUS_MAX (Muller 6.32-6.33),
    # refined by a parabola through the peak bin and its two neighbours (on a log scale)
    average = fourier_tempogram.mean(axis=1)
    candidates = np.where((tempo_axis >= TACTUS_MIN) & (tempo_axis <= TACTUS_MAX))[0]
    k = candidates[np.argmax(average[candidates])]
    a, b, c = np.log(average[k - 1:k + 2] + 1e-12)
    curvature = a - 2*b + c
    offset = float(np.clip(0.5*(a - c)/curvature, -0.5, 0.5)) if curvature < 0 else 0.0
    return float(tempo_axis[k] + offset*(tempo_axis[1] - tempo_axis[0]))


def bar_ratios(autocorr_tempogram, tempo_axis, tau_hat):
    # time-averaged autocorrelation at 2, 3 and 4 beats, divided by its value at 1 beat
    average = autocorr_tempogram.mean(axis=1)
    usable = np.isfinite(tempo_axis) & (tempo_axis > 0) & np.isfinite(average)
    order = np.argsort(tempo_axis[usable])
    curve = interp1d(tempo_axis[usable][order], average[usable][order], bounds_error=False, fill_value=0.0)
    at_beat = float(curve(tau_hat))
    if at_beat <= 0:
        raise ValueError('autocorrelation at the beat is zero, bar ratios are undefined')
    return {name: float(curve(tau_hat/beats))/at_beat for name, beats in zip(RATIO_COLUMNS, [2, 3, 4])}


def scale_transform(novelty):
    # Scale transform of the autocorrelation of the novelty curve (Holzapfel & Stylianou, 2011).
    # Playing the music faster squeezes the autocorrelation along the lag axis. On a log lag axis
    # that squeeze becomes a shift, and the magnitude of a Fourier transform does not change when
    # its input is shifted. So these numbers describe the rhythm pattern but not its speed.
    x = novelty - novelty.mean()
    n = len(x)
    lags = np.arange(n)
    autocorr = np.correlate(x, x, mode='full')[n - 1:]/(n - lags)
    autocorr = autocorr/autocorr[0]

    frames_per_second = SR/HOP
    log_lags = np.exp(np.linspace(np.log(SCALE_MIN_LAG*frames_per_second),
                                  np.log(SCALE_MAX_LAG*frames_per_second), SCALE_POINTS))
    warped = np.interp(log_lags, lags, autocorr)*np.sqrt(log_lags)    # sqrt(t) comes from the definition
    magnitude = np.abs(np.fft.rfft(warped))[:N_SCALE]
    return magnitude/np.linalg.norm(magnitude)


def rhythm_features(y):
    novelty = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    fourier = np.abs(librosa.feature.fourier_tempogram(onset_envelope=novelty, sr=SR, hop_length=HOP, win_length=WIN))
    fourier_axis = librosa.fourier_tempo_frequencies(sr=SR, hop_length=HOP, win_length=WIN)
    autocorr = librosa.feature.tempogram(onset_envelope=novelty, sr=SR, hop_length=HOP, win_length=WIN)
    autocorr_axis = librosa.tempo_frequencies(n_bins=autocorr.shape[0], sr=SR, hop_length=HOP)
    pulse = librosa.beat.plp(onset_envelope=novelty, sr=SR, hop_length=HOP, win_length=WIN,
                             tempo_min=TACTUS_MIN, tempo_max=TACTUS_MAX)

    tau_hat = estimate_tau_hat(fourier, fourier_axis)
    cyclic = cyclic_tempogram(fourier, fourier_axis)
    feats = dict(zip(CYCLIC_COLUMNS, map(float, cyclic)))
    feats.update(bar_ratios(autocorr, autocorr_axis, tau_hat))
    feats['plp_confidence'] = float(np.mean(pulse))
    feats['tau_hat'] = tau_hat
    feats.update(zip(SHIFTED_COLUMNS, map(float, shift_to_strongest_bin(cyclic))))
    feats.update(zip(SCALE_COLUMNS, map(float, scale_transform(novelty))))
    return feats


#--------------------------------------------------------------------------------
# All features of all 13 versions

def condition_rows(path, clip):
    # the features of one clip in all 13 versions; errors are returned, not hidden
    try:
        y = load_audio(path)
        rows = []
        for condition in CONDITIONS:
            y_condition = apply_condition(y, condition)
            rows.append({**clip, **condition, 'duration': len(y_condition)/SR, 'original_duration': len(y)/SR,
                         **timbral_features(y_condition), **rhythm_features(y_condition)})
        return rows, None
    except Exception as e:
        return [], {'filename': clip['filename'], 'error': repr(e)}


def is_complete(path, clips):
    if not path.is_file():
        return False
    saved = pd.read_csv(path, usecols=['filename', 'condition'])
    return (len(saved) == len(clips)*len(CONDITIONS) and set(saved['filename']) == set(clips['filename'])
            and set(saved['condition']) == set(CONDITION_ORDER))


def extract_all_versions(clips, dataset_dir, output_path, n_jobs=8, batch_size=25):
    # computed 25 clips at a time; progress is saved after every batch, so an interrupted run
    # continues where it stopped when it is started again
    partial = output_path.with_name(output_path.stem + '_partial.csv')
    done = set(pd.read_csv(partial, usecols=['filename'])['filename']) if partial.is_file() else set()
    todo = clips[~clips['filename'].isin(done)]
    print(f'{len(done)} clips already done, {len(todo)} to go (using {n_jobs} cores)')
    failed = []
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size]
        jobs = (delayed(condition_rows)(dataset_dir / clip['folder'] / clip['filename'],
                                        {k: clip[k] for k in ['filename', 'folder', 'label']})
                for _, clip in batch.iterrows())
        results = Parallel(n_jobs=n_jobs)(jobs)
        rows = [row for clip_rows, _ in results for row in clip_rows]
        failed += [error for _, error in results if error is not None]
        if rows:
            pd.DataFrame(rows).to_csv(partial, mode='a', header=not partial.is_file(), index=False)
        print(f'  {len(done) + start + len(batch)} of {len(clips)} clips', flush=True)
    if failed:
        raise RuntimeError(f'{len(failed)} clips failed (the others are saved, rerun to retry them): {failed[:5]}')
    table = pd.read_csv(partial)
    table.to_csv(output_path, index=False)
    partial.unlink()
    return table
