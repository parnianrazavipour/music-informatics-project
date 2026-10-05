import numpy as np
import os
import librosa
import pandas as pd

# Same functions as in Timbral_extraction.ipynb, so they can be imported from other files.
# The explanations and the formulas are in the notebook.


def load_audio(audio_path, sr=22050, n_fft=2048):
    """
    Loads the whole clip as mono at the given sampling rate (no trimming, no normalizing).
    Raises an error if the audio is too short or silent.
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    if len(y) < n_fft:
        raise ValueError('audio is too short')
    if np.max(np.abs(y)) == 0:
        raise ValueError('audio is silent')
    return y


def summarize(mfcc, zcr, sc):
    """
    Mean and std over the frames of every feature.

    Parameters:
    mfcc: 2D array of shape (n_mfcc, N_frames)
    zcr: 1D array of shape (N_frames,)
    sc: 1D array of shape (N_frames,) (spectral centroid in Hz)

    returns:
    A dict with 2 * n_mfcc + 4 values (30 values for 13 MFCCs)
    """
    feats = {}
    for i in range(len(mfcc)):
        feats[f'mfcc_{i:02d}_mean'] = float(np.mean(mfcc[i]))
        feats[f'mfcc_{i:02d}_std'] = float(np.std(mfcc[i]))
    feats['zcr_mean'] = float(np.mean(zcr))
    feats['zcr_std'] = float(np.std(zcr))
    feats['spectral_centroid_mean'] = float(np.mean(sc))
    feats['spectral_centroid_std'] = float(np.std(sc))
    return feats


#--------------------------------------------------------------------------------
# Version 1: with librosa (this is the main version)

def frame_features(y, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13, n_mels=128):
    """
    MFCCs, zero-crossing rate and spectral centroid for every frame, all on the same frames.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    return mfcc, zcr, sc


def timbral_features(y, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13, n_mels=128):
    """
    Timbral feature vector of one clip (as a dict), computed with librosa.

    Arguments:
    y: 1D array of audio samples (can also be a time-stretched clip)
    sr: Sampling rate of the audio signal
    n_fft: Frame size in samples
    hop_length: Number of samples between successive frames
    n_mfcc: Number of MFCCs to keep
    n_mels: Number of Mel bands used before the MFCCs
    """
    mfcc, zcr, sc = frame_features(y, sr, n_fft, hop_length, n_mfcc, n_mels)
    return summarize(mfcc, zcr, sc)


def extract_timbral_features(audio_path, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13, n_mels=128):
    """Loads one audio file and returns its timbral features (librosa version)."""
    y = load_audio(audio_path, sr, n_fft)
    return timbral_features(y, sr, n_fft, hop_length, n_mfcc, n_mels)


#--------------------------------------------------------------------------------
# Version 2: from scratch (numpy only), kept as a backup

# these two functions are from our lab 1

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
    # Mel scale of librosa (Slaney): linear below 1 kHz, logarithmic above
    if f < 1000:
        return 3*f/200
    return 15 + 27*np.log(f/1000)/np.log(6.4)

def mel_to_hz(m):
    # m is an array here
    low = 200*m/3
    high = 1000*np.exp((m - 15)*np.log(6.4)/27)
    return np.where(m < 15, low, high)

def mel_filterbank(N, fs, n_mels=128):
    # triangular filters, one row per Mel band
    freqs = fs*np.arange(N//2 + 1)/N
    edges = mel_to_hz(np.linspace(0, hz_to_mel(fs/2), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        left, mid, right = edges[m], edges[m+1], edges[m+2]
        up = (freqs - left)/(mid - left)
        down = (right - freqs)/(right - mid)
        fb[m] = np.maximum(0, np.minimum(up, down))
        fb[m] = fb[m]*2/(right - left)     # divide by the width
    return fb

def mfcc_from_scratch(samples, N, hop, fs, n_mfcc=13, n_mels=128):
    w = np.hanning(N)
    fb = mel_filterbank(N, fs, n_mels)

    # power spectrum and Mel band energies of every frame
    energies = []
    for n in range(0, len(samples)-N+1, hop):
        power = abs(np.fft.rfft(samples[n:n+N]*w))**2
        energies.append(fb @ power)
    energies = np.array(energies)

    # dB, and keep only the top 80 dB like librosa does
    log_e = 10*np.log10(energies + 1e-10)
    log_e = np.maximum(log_e, log_e.max() - 80)

    # DCT, with the same scaling as librosa
    i = np.arange(n_mfcc).reshape(-1, 1)
    m = np.arange(n_mels).reshape(1, -1)
    dct = np.cos(np.pi*i/n_mels*(m + 0.5))*np.sqrt(2/n_mels)
    dct[0] = dct[0]/np.sqrt(2)

    return dct @ log_e.T      # shape (n_mfcc, number of frames)


def frame_features_scratch(y, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13, n_mels=128):
    mfcc = mfcc_from_scratch(y, n_fft, hop_length, sr, n_mfcc, n_mels)
    zcr = count_zero_crossings(y, n_fft, hop_length)/n_fft
    with np.errstate(invalid='ignore'):
        sc = np.nan_to_num(centroid(y, n_fft, hop_length, sr))   # silent frames give 0/0
    return mfcc, zcr, sc


def timbral_features_scratch(y, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13, n_mels=128):
    """Same as timbral_features, but computed with our own functions."""
    mfcc, zcr, sc = frame_features_scratch(y, sr, n_fft, hop_length, n_mfcc, n_mels)
    return summarize(mfcc, zcr, sc)


#--------------------------------------------------------------------------------
# Whole dataset

def find_tracks(dataset_dir="BallroomData"):
    """
    Finds all wav files. The folder name is the label, and the three Rumba folders are merged.

    returns:
    A DataFrame with the columns track_id, filename, folder, label
    """
    rows = []
    for folder, subfolders, files in os.walk(dataset_dir):
        for name in files:
            if name.lower().endswith('.wav'):
                style = os.path.basename(folder)
                label = style
                if style.startswith('Rumba'):
                    label = 'Rumba'
                rows.append({'track_id': style + '/' + name[:-4],
                             'filename': name,
                             'folder': style,
                             'label': label})
    return pd.DataFrame(rows).sort_values('track_id').reset_index(drop=True)


def extract_dataset(dataset_dir="BallroomData", from_scratch=False, sr=22050):
    """
    Timbral features of all clips in the dataset.

    Arguments:
    dataset_dir: Folder of the Ballroom dataset
    from_scratch: If True, uses our own functions instead of librosa (much slower)

    returns:
    df: DataFrame with one row per clip (info columns, duration, 30 features)
    failed: list of the files that gave an error (they are not in df)
    """
    tracks = find_tracks(dataset_dir)
    rows = []
    failed = []

    for i, track in tracks.iterrows():
        path = os.path.join(dataset_dir, track['folder'], track['filename'])
        try:
            y = load_audio(path, sr)
            if from_scratch:
                feats = timbral_features_scratch(y, sr)
            else:
                feats = timbral_features(y, sr)
        except Exception as e:
            failed.append({'track_id': track['track_id'], 'error': repr(e)})
            continue

        row = dict(track)
        row['duration'] = len(y)/sr     # only for checking the data, not a feature
        row.update(feats)
        rows.append(row)

        if (i + 1) % 100 == 0:
            print(i + 1, 'of', len(tracks), 'done')

    return pd.DataFrame(rows), failed


#--------------------------------------------------------------------------------
# Usage example:
if __name__ == "__main__":
    path = "BallroomData/VienneseWaltz/Albums-Ballroom_Classics4-11.wav"

    feats = extract_timbral_features(path)
    print(path)
    for name, value in feats.items():
        print(f"  {name:<24} {value:10.3f}")

    # on a time-stretched clip (same settings, so the two can be compared)
    y = load_audio(path)
    y_fast = librosa.effects.time_stretch(y, rate=1.1)
    feats_fast = timbral_features(y_fast)
    print("spectral centroid mean, original: %.1f Hz, 10%% faster: %.1f Hz"
          % (feats['spectral_centroid_mean'], feats_fast['spectral_centroid_mean']))

    # to extract the whole dataset and save it:
    # df, failed = extract_dataset("BallroomData")
    # df.to_csv("ballroom_timbral_features.csv", index=False)
