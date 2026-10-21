import warnings

import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.signal import get_window

# features of every clip in all 13 versions

SR, HOP, WIN, N_FFT_TEMPOGRAM = 22050, 512, 512, 4096
N_FFT, N_MFCC, N_MELS = 2048, 13, 128
M, TAU_0, CYCLIC_MIN, CYCLIC_MAX = 15, 60.0, 30.0, 480.0
TACTUS_MIN, TACTUS_MAX = 60.0, 280.0          # Rythm_extraction.py uses 220
N_SCALE, SCALE_MIN_LAG, SCALE_MAX_LAG, SCALE_POINTS = 20, 0.1, 4.0, 256   # lags in seconds

TIMBRAL_COLUMNS = [f'mfcc_{i:02d}_{statistic}' for i in range(N_MFCC) for statistic in ['mean', 'std']] \
    + ['zcr_mean', 'zcr_std', 'spectral_centroid_mean', 'spectral_centroid_std']
CYCLIC_COLUMNS = [f'cyclic_bin_{i}' for i in range(M)]
SHIFTED_COLUMNS = [f'shifted_bin_{i}' for i in range(M)]
RATIO_COLUMNS = ['R_2_4', 'R_3_4', 'R_4_4']
PLP_COLUMNS = ['plp_confidence', 'plp_alignment']
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
    # faster or slower, same pitch
    if rate == 1.0:
        return y
    with warnings.catch_warnings():
        # harmless librosa 1.0 warning
        warnings.simplefilter('ignore', FutureWarning)
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


def count_zero_crossings(samples, N, hop):
    pos = samples >= 0
    counts = []
    for n in range(0, len(samples)-N+1, hop):
        chunk = pos[n:n+N]
        counts.append(np.count_nonzero(chunk[1:] != chunk[:-1]))
    return np.array(counts)


def centroid(samples, N, hop, fs):
    out = []
    for n in range(0, len(samples)-N+1, hop):
        mag = abs(np.fft.rfft(samples[n:n+N]))
        freqs = fs*np.arange(len(mag))/N
        out.append(np.sum(freqs*mag)/np.sum(mag))
    return np.array(out)


def hz_to_mel(f):
    if f < 1000:
        return 3*f/200
    return 15 + 27*np.log(f/1000)/np.log(6.4)


def mel_to_hz(m):
    low = 200*m/3
    high = 1000*np.exp((m - 15)*np.log(6.4)/27)
    return np.where(m < 15, low, high)


def mel_filterbank(N, fs, n_mels=128):
    freqs = fs*np.arange(N//2 + 1)/N
    edges = mel_to_hz(np.linspace(0, hz_to_mel(fs/2), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        left, mid, right = edges[m], edges[m+1], edges[m+2]
        up = (freqs - left)/(mid - left)
        down = (right - freqs)/(right - mid)
        fb[m] = np.maximum(0, np.minimum(up, down))
        fb[m] = fb[m]*2/(right - left)
    return fb


def mfcc_from_scratch(samples, N, hop, fs, n_mfcc=13, n_mels=128):
    w = np.hanning(N)
    fb = mel_filterbank(N, fs, n_mels)

    energies = []
    for n in range(0, len(samples)-N+1, hop):
        power = abs(np.fft.rfft(samples[n:n+N]*w))**2
        energies.append(fb @ power)
    energies = np.array(energies)

    # keep the top 80 dB, like librosa
    log_e = 10*np.log10(energies + 1e-10)
    log_e = np.maximum(log_e, log_e.max() - 80)

    # same DCT scaling as librosa
    i = np.arange(n_mfcc).reshape(-1, 1)
    m = np.arange(n_mels).reshape(1, -1)
    dct = np.cos(np.pi*i/n_mels*(m + 0.5))*np.sqrt(2/n_mels)
    dct[0] = dct[0]/np.sqrt(2)

    return dct @ log_e.T


def timbral_features(y):
    mfcc = mfcc_from_scratch(y, N_FFT, HOP, SR, N_MFCC, N_MELS)
    zcr = count_zero_crossings(y, N_FFT, HOP)/N_FFT
    with np.errstate(invalid='ignore'):
        sc = np.nan_to_num(centroid(y, N_FFT, HOP, SR))   # silent frames give 0/0
    feats = {}
    for i in range(N_MFCC):
        feats[f'mfcc_{i:02d}_mean'] = float(np.mean(mfcc[i]))
        feats[f'mfcc_{i:02d}_std'] = float(np.std(mfcc[i]))
    feats['zcr_mean'], feats['zcr_std'] = float(np.mean(zcr)), float(np.std(zcr))
    feats['spectral_centroid_mean'], feats['spectral_centroid_std'] = float(np.mean(sc)), float(np.std(sc))
    return feats


window = get_window('hann', WIN, fftbins=True)
WINDOW_CORRECTION = np.correlate(window, window, mode='full')[WIN - 1:]
WINDOW_CORRECTION = np.maximum(WINDOW_CORRECTION/WINDOW_CORRECTION[0], 0.05)


def fourier_tempogram(novelty):
    # zero padding gives a value every 0.63 BPM
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tempogram = np.abs(librosa.stft(novelty, n_fft=N_FFT_TEMPOGRAM, hop_length=1, win_length=WIN,
                                        window='hann', center=True))
    tempo_axis = librosa.fft_frequencies(sr=SR/HOP, n_fft=N_FFT_TEMPOGRAM)*60
    return tempogram, tempo_axis


def cyclic_tempogram(fourier_tempogram, tempo_axis):
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
    # strongest bin first (cyclic_shift=True)
    return np.roll(cyclic, -int(np.argmax(cyclic)))


def estimate_tau_hat(fourier_tempogram, fourier_axis, autocorr_tempogram, autocorr_axis):
    # Fourier x autocorrelation tempogram (Muller p. 330)
    fourier = fourier_tempogram.mean(axis=1)
    autocorr = autocorr_tempogram.mean(axis=1)
    usable = np.isfinite(autocorr_axis) & (autocorr_axis > 0) & np.isfinite(autocorr)
    order = np.argsort(autocorr_axis[usable])
    autocorr = interp1d(autocorr_axis[usable][order], autocorr[usable][order],
                        bounds_error=False, fill_value=0.0)(fourier_axis)
    autocorr = np.maximum(0.0, autocorr)

    in_range = (fourier_axis >= TACTUS_MIN) & (fourier_axis <= TACTUS_MAX)
    f = fourier[in_range] - fourier[in_range].min()
    a = autocorr[in_range] - autocorr[in_range].min()
    combined = np.zeros_like(fourier)
    combined[in_range] = f/(f.max() + 1e-8)*a/(a.max() + 1e-8)

    k = np.where(in_range)[0][np.argmax(combined[in_range])]
    left, peak, right = combined[k - 1:k + 2]
    curvature = left - 2*peak + right
    offset = float(np.clip(0.5*(left - right)/curvature, -0.5, 0.5)) if abs(curvature) > 1e-12 else 0.0
    return float(fourier_axis[k] + offset*(fourier_axis[k + 1] - fourier_axis[k - 1])/2)


def bar_ratios(autocorr_tempogram, tempo_axis, tau_hat):
    # corrected for the Hann window, strongest value within 5%
    average = autocorr_tempogram.mean(axis=1)/WINDOW_CORRECTION
    usable = np.isfinite(tempo_axis) & (tempo_axis > 0) & np.isfinite(average)
    order = np.argsort(tempo_axis[usable])
    curve = interp1d(tempo_axis[usable][order], average[usable][order], bounds_error=False, fill_value=0.0)

    def strongest_near(bpm):
        return float(np.max(curve(np.linspace(0.95*bpm, 1.05*bpm, 11))))

    at_beat = strongest_near(tau_hat)
    if at_beat <= 1e-8:
        raise ValueError('autocorrelation at the beat is zero, bar ratios are undefined')
    return {name: strongest_near(tau_hat/beats)/at_beat for name, beats in zip(RATIO_COLUMNS, [2, 3, 4])}


def scale_transform(novelty):
    # scale transform (Holzapfel & Stylianou, 2011)
    x = novelty - novelty.mean()
    n = len(x)
    lags = np.arange(n)
    autocorr = np.correlate(x, x, mode='full')[n - 1:]/(n - lags)
    autocorr = autocorr/autocorr[0]

    frames_per_second = SR/HOP
    log_lags = np.exp(np.linspace(np.log(SCALE_MIN_LAG*frames_per_second),
                                  np.log(SCALE_MAX_LAG*frames_per_second), SCALE_POINTS))
    warped = np.interp(log_lags, lags, autocorr)*np.sqrt(log_lags)
    magnitude = np.abs(np.fft.rfft(warped))[:N_SCALE]
    return magnitude/np.linalg.norm(magnitude)


def rhythm_features(y):
    novelty = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    fourier, fourier_axis = fourier_tempogram(novelty)
    centred = np.maximum(0.0, novelty - np.mean(novelty))
    autocorr = librosa.feature.tempogram(onset_envelope=centred, sr=SR, hop_length=HOP, win_length=WIN)
    autocorr_axis = librosa.tempo_frequencies(n_bins=autocorr.shape[0], sr=SR, hop_length=HOP)
    pulse = librosa.beat.plp(onset_envelope=novelty, sr=SR, hop_length=HOP, win_length=WIN,
                             tempo_min=TACTUS_MIN, tempo_max=TACTUS_MAX)

    tau_hat = estimate_tau_hat(fourier, fourier_axis, autocorr, autocorr_axis)
    cyclic = cyclic_tempogram(fourier, fourier_axis)
    feats = dict(zip(CYCLIC_COLUMNS, map(float, cyclic)))
    feats.update(bar_ratios(autocorr, autocorr_axis, tau_hat))
    feats['plp_confidence'] = float(np.mean(pulse))
    # cosine similarity between novelty and pulse
    feats['plp_alignment'] = float(np.dot(novelty, pulse)/(np.linalg.norm(novelty)*np.linalg.norm(pulse) + 1e-8))
    feats['tau_hat'] = tau_hat
    feats.update(zip(SHIFTED_COLUMNS, map(float, shift_to_strongest_bin(cyclic))))
    feats.update(zip(SCALE_COLUMNS, map(float, scale_transform(novelty))))
    return feats


def condition_rows(path, clip):
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
