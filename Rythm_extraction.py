import numpy as np
import os
import librosa
from scipy.interpolate import interp1d

def _cyclic_tempogram_features(base_tempogram, tempo_axis, M, tau_0, max_temp, min_temp):
    """
    Parameters: 
    base_tempogram: 2D array of shape (N_tempo_bins, N_frames) (Fourier tempogram).
    tempo_axis: 1D array of shape (N_tempo_bins,) containing the tempo
    M: Number of bins per octave (or: the number of cyclic features / equivalence classes)
    tau_0: Reference tempo (e.g., 60 BPM)
    max_temp: Maximum tempo (e.g., 480 (60 * 2^3) BPM)
    min_temp: Minimum tempo (e.g., 30 BPM)
    """
    # Converts the tempo axis to a logarithmic scale (base 2) and computes the number of octaves
    octave_min = int(np.floor(np.log2(min_temp / tau_0)))
    octave_max = int(np.ceil(np.log2(max_temp / tau_0)))
    num_octaves = octave_max - octave_min

    # Creates the logarithmic tempo axis with exactly M points per octave
    tot_num_bins = num_octaves * M
    m_indices = np.arange(tot_num_bins)
    tempo_log_axis = tau_0 * (2.0 ** (octave_min + m_indices / M)) # BPM(m) = tau_0 * 2^(octave_min + m/M)

    # Interpolates the base tempogram onto the logarithmic tempo axis (so we get the values at each BPM(m), for m = 0, 1, ..., M-1)
    sort_idx = np.argsort(tempo_axis) # Sorts the tempo axis to ensure monotonicity (increasing) for interpolation
    tempo_sorted = tempo_axis[sort_idx]
    tempogram_sorted = base_tempogram[sort_idx, :]

    interp_func = interp1d(tempo_sorted, tempogram_sorted, kind='linear', axis=0, bounds_error=False, fill_value=0.0)
    T_log = interp_func(tempo_log_axis)  # Form: (num_octaves * M, N_frames)

    # Sums the tempogram values across octaves to create a cyclic tempogram (Muller 6.34)
    T_log_reshaped = T_log.reshape((num_octaves, M, -1))
    cyclic_tempogram = np.sum(T_log_reshaped, axis=0)  # Form: (M, N_frames)

    # Computes the mean across frames and normalizes the cyclic tempogram (Muller 6.32)
    cyclic_mean = np.mean(cyclic_tempogram, axis=1)
    norm = np.linalg.norm(cyclic_mean)
    if norm > 0:
        cyclic_mean = cyclic_mean / norm

    return cyclic_mean  # M dimensions

def _autocorr_subharmonic_features(tempogram_autocorr, tempo_axis_autocorr, tempo_min, tempo_max):
    """
    Extracts subharmonic energy ratios from an autocorrelation tempogram
    to capture meter and measure-level structure (Muller 6.2.3).
    """
    # Average across time to obtain an overall summary curve (Muller 6.32)
    t_a_avg = np.mean(tempogram_autocorr, axis=1)

    # Estimate primary tempo (tactus) within plausible range [tempo_min, tempo_max] BPM
    valid_mask = (tempo_axis_autocorr >= tempo_min) & (tempo_axis_autocorr <= tempo_max)
    if not np.any(valid_mask):
        return np.zeros(3)

    valid_tempi = tempo_axis_autocorr[valid_mask]
    valid_energy = t_a_avg[valid_mask]
    tau_hat = valid_tempi[np.argmax(valid_energy)] # Muller 6.33 

    # Sort tempo axis to ensure monotonicity required by interp1d (this is needed because we may not have a value at say, tau_hat / 3 = 33.33 BPM if tau_hat = 100 BPM)
    sort_idx = np.argsort(tempo_axis_autocorr)
    interp_autocorr = interp1d(
        tempo_axis_autocorr[sort_idx],
        t_a_avg[sort_idx],
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )

    base_energy = float(interp_autocorr(tau_hat))

    # Subharmonic ratios: 3/4 (waltz), 2/4 (half-measure), and 4/4 (full measure)
    r_3_4 = float(interp_autocorr(tau_hat / 3.0) / base_energy)
    r_2_4 = float(interp_autocorr(tau_hat / 2.0) / base_energy)
    r_4_4 = float(interp_autocorr(tau_hat / 4.0) / base_energy)

    return np.array([r_2_4, r_3_4, r_4_4]), float(tau_hat)

def _plp_features(novelty_curve, sr, hop_length, tempo_min, tempo_max):
    """
    Extracts statistical descriptors from the Predominant Local Pulse (PLP) function Gamma
    """
    # Compute PLP function Gamma via overlap-add of optimal sinusoids (Muller 6.39)
    # The possible tempo Theta is restricted to [tempo_min, tempo_max] for tactus tracking.
    gamma = librosa.beat.plp(
        onset_envelope=novelty_curve,
        sr=sr,
        hop_length=hop_length,
        tempo_min=tempo_min,
        tempo_max=tempo_max
    )

    # PLP confidence: mean amplitude of Gamma via constructive interference
    plp_confidence = float(np.mean(gamma))

    # Peak-to-average ratio: measures pulse prominence and accent sharpness
    plp_peak_ratio = float(np.max(gamma) / (plp_confidence))

    return np.array([plp_confidence, plp_peak_ratio])

def extract_full_rhythm_vector(y, sr=22050, hop_length=512, win_length=384, M=15, cyclic_tau_0=60.0, cyclic_min_temp=30.0, cyclic_max_temp=480.0, pa_min_tempo=40.0, pa_max_tempo=240.0):
    """
    Arguments:
    ---General Arguments---
    y: 1D array of audio samples
    sr: Sampling rate of the audio signal
    hop_length: Number of samples between successive frames
    win_length: Window length for tempograms

    ---Cyclic Tempogram Arguments---
    M: Number of bins per octave (or: the number of cyclic features / equivalence classes)
    cyclic_tau_0: Reference tempo (e.g., 60 BPM) (tau_0)
    cyclic_min_temp: Minimum tempo (e.g., 30 BPM. Should be a power of 2 fraction of tau_0, e.g., tau_0 / 2^k for some k)
    cyclic_max_temp: Maximum tempo (e.g., 480 BPM. Should be a power of 2 fraction of tau_0, e.g., tau_0 * 2^k for some k)

    ---PLP Arguments / Autocorrelation Arguments---
    pa_min_tempo: Minimum tempo for PLP and Autocorrelation (e.g., 40 BPM)
    pa_max_tempo: Maximum tempo for PLP and Autocorrelation (e.g., 240 BPM)

    returns:
    A 1D array of shape (M + 3 + 2,) containing the concatenated rhythm features:
    - Cyclic Tempogram features (M dimensions)
    - Subharmonic Ratios (3 dimensions) (R_3/4, R_2/4, R_4/4)
    - Predominant Local Pulse (PLP) features (2 dimensions) (PLP confidence, PLP peak-to-average ratio)
    - tau_hat: Estimated tactus tempo. That is, estimated BPM of the tactus (tempo baseline) based on the autocorrelation tempogram, of the piece. 
    Full output: [cyclic_tempogram_features (M dims), R_2/4, R_3/4, R_4/4, PLP_confidence, PLP_peak_to_average_ratio], tau_hat
    """
    # Novelty (Spectral Flux / Spectral novelty with log-magnitude)
    novelty = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Fourier-tempogram using novelty
    T_F = np.abs(librosa.feature.fourier_tempogram(onset_envelope=novelty, sr=sr, hop_length=hop_length, win_length=win_length))
    tempo_axis_fourier = librosa.fourier_tempo_frequencies(sr=sr, hop_length=hop_length, win_length=win_length)

    # Autocorrelation-tempogram using novelty
    T_A = librosa.feature.tempogram(onset_envelope=novelty, sr=sr, hop_length=hop_length, win_length=win_length)
    tempo_axis_autocorr = librosa.tempo_frequencies(n_bins=T_A.shape[0], sr=sr, hop_length=hop_length)

    # Extract cyclic tempogram features (15 dimensions) using the Fourier tempogram
    feats_cyclic = _cyclic_tempogram_features(
        base_tempogram=T_F,
        tempo_axis=tempo_axis_fourier,
        M=M,
        tau_0=cyclic_tau_0,
        min_temp=cyclic_min_temp,
        max_temp=cyclic_max_temp,
    )

    # Extract autocorrelation subharmonic features (3 dimensions) using the autocorrelation tempogram
    feats_autocorr, tau_hat = _autocorr_subharmonic_features(
        tempogram_autocorr=T_A,
        tempo_axis_autocorr=tempo_axis_autocorr,
        tempo_max=pa_max_tempo,
        tempo_min=pa_min_tempo,
    ) 

    # Extract PLP features (2 dimensions) using the novelty 
    feats_plp = _plp_features(
        novelty_curve=novelty,
        sr=sr,
        hop_length=hop_length,
        tempo_min=pa_min_tempo,
        tempo_max=pa_max_tempo,
    )  

    return np.concatenate([feats_cyclic, feats_autocorr, feats_plp]), tau_hat



#--------------------------------------------------------------------------------
# Debugging / Checking Output 
def print_rhythm_features(features, tau_hat, file_path=None):
    """NB: This is for checking output"""
    # De sista 5 värdena är alltid Subharmonic Ratios (3) + PLP (2)
    cyclic = features[:-5]
    r_2_4, r_3_4, r_4_4 = features[-5:-2]
    plp_conf, plp_ratio = features[-2:]

    M = len(cyclic)
    top_bin = int(np.argmax(cyclic))
    ascii_bar = "".join(
        ["█" if v > 0.35 else "▄" if v > 0.20 else "·" for v in cyclic]
    )
    is_valid = not np.isnan(features).any() and not np.isinf(features).any()

    print("=" * 72)
    if file_path:
        print(f" FILE: {os.path.basename(file_path)}")
    print(
        f" Shape: {features.shape} (M={M}) | Numerically Stable: {is_valid}"
    )
    print("-" * 72)

    # 1. Cyclic Tempogram
    print(
        f" 1. CYCLIC TEMPOGRAM ({M} dims, octave-wrapped tempo classes)"
    )
    print(
        f"    • Dominant bin:       Bin {top_bin:<2} (energy: {cyclic[top_bin]:.3f})"
    )
    print(f"    • Profile (0-{M-1}):     [{ascii_bar}]")
    print(f"    • Vector values:       {np.array2string(cyclic, precision=3)}")
    print()

    # 2. Subharmonic Ratios
    print(" 2. SUBHARMONIC RATIOS (3 dims, meter & metric structure)")
    print(f"    • Estimated tactus tempo (tau_hat): {tau_hat:.2f} BPM")
    marker = "  ◄── Triple-meter dominant" if r_3_4 > 0.8 else ""
    print(f"    • R_3/4 (Waltz / 3-beat):     {r_3_4:.3f}{marker}")
    print(f"    • R_2/4 (Half-bar / 2-beat):   {r_2_4:.3f}")
    print(f"    • R_4/4 (Full-bar / 4-beat):   {r_4_4:.3f}")
    print()

    # 3. Predominant Local Pulse
    print(" 3. PREDOMINANT LOCAL PULSE (2 dims, wave dynamics from Gamma)")
    print(
        f"    • PLP Confidence (pulse prominence): {plp_conf:.3f}  (mean amplitude)"
    )
    print(
        f"    • PLP Peak-to-Average Ratio:        {plp_ratio:.3f}  (accent sharpness)"
    )
    print("=" * 72)

# Usage example:
path = "BallroomData/Rumba-American/Albums-AnaBelen_Veneo-13.wav"
y, sr = librosa.load(path, sr=22050)
features, tau_hat = extract_full_rhythm_vector(y, M=30)

print_rhythm_features(features, tau_hat, file_path=path)