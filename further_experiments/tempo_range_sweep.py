import os
import re
import warnings

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import features as F

# Choosing the upper tempo limit for tau_hat (Table 4 in the report).
# For each limit we check how often tau_hat matches the annotated tempo (or its double) on the
# original audio, and how often it jumps an octave when the excerpt is stretched.
# Run from the repository root: python further_experiments/tempo_range_sweep.py

warnings.filterwarnings('ignore')

LIMITS = [200, 220, 240, 260, 280, 300]
DATASET = 'BallroomData'


def tau_hat_per_limit(y):
    novelty = librosa.onset.onset_strength(y=y, sr=F.SR, hop_length=F.HOP)
    fourier, fourier_axis = F.fourier_tempogram(novelty)
    centred = np.maximum(0.0, novelty - np.mean(novelty))
    autocorr = librosa.feature.tempogram(onset_envelope=centred, sr=F.SR, hop_length=F.HOP, win_length=F.WIN)
    autocorr_axis = librosa.tempo_frequencies(n_bins=autocorr.shape[0], sr=F.SR, hop_length=F.HOP)
    taus = {}
    for limit in LIMITS:
        F.TACTUS_MIN, F.TACTUS_MAX = 60.0, float(limit)
        taus[limit] = F.estimate_tau_hat(fourier, fourier_axis, autocorr, autocorr_axis)
    return taus


def clip_rows(path):
    y = F.load_audio(path)
    rows = []
    for rate in [1.0] + F.RATES:
        audio = y if rate == 1.0 else F.change_tempo(y, rate)
        for limit, tau in tau_hat_per_limit(audio).items():
            rows.append({'filename': os.path.basename(path), 'rate': rate, 'limit': limit, 'tau_hat': tau})
    return rows


def annotated_bpm():
    # the BPM labels are in the .log files of the dataset
    bpm = {}
    for log in sorted(os.listdir(f'{DATASET}/nada')):
        if not log.endswith('.log'):
            continue
        lines = open(f'{DATASET}/nada/{log}', encoding='latin-1').read().splitlines()
        for i, line in enumerate(lines):
            m = re.search(r'/Music/(Media|Albums)/(.+)\.ram', line)
            if not m:
                continue
            found = [re.match(r'(\d+) BPM', l) for l in lines[i + 1:i + 5] if re.match(r'\d+ BPM', l)]
            if not found:
                continue
            if m.group(1) == 'Media':
                name = f'Media-{m.group(2)}.wav'
            else:
                name = 'Albums-' + m.group(2).replace('/', '-') + '.wav'
            bpm[name] = float(found[0].group(1))
    return pd.Series(bpm)


def is_octave_jump(ratio):
    # the estimate should follow the stretch; near double or half means it jumped an octave
    return (np.abs(ratio - 2) < 0.1) | (np.abs(ratio - 0.5) < 0.05)


paths = sorted(f'{DATASET}/{d}/{n}' for d in os.listdir(DATASET) if os.path.isdir(f'{DATASET}/{d}')
               for n in os.listdir(f'{DATASET}/{d}') if n.endswith('.wav'))
results = Parallel(n_jobs=8)(delayed(clip_rows)(p) for p in paths)
taus = pd.DataFrame([row for rows in results for row in rows])
bpm = annotated_bpm()

table = []
for limit in LIMITS:
    t = taus[taus['limit'] == limit]
    original = t[t['rate'] == 1.0].set_index('filename')['tau_hat']
    stretched = t[t['rate'] != 1.0].copy()
    stretched['ratio'] = stretched['tau_hat']/(original.loc[stretched['filename']].to_numpy()*stretched['rate'])
    stretched['jump'] = is_octave_jump(stretched['ratio'])
    jumped = stretched.groupby('filename')['jump'].any()
    vs_label = original/bpm.loc[original.index]
    table.append({'tau_max': limit,
                  '1x (%)': 100*np.mean(np.abs(vs_label - 1) <= 0.04),
                  '2x (%)': 100*np.mean(np.abs(vs_label/2 - 1) <= 0.04),
                  'jump-free (%)': 100*(1 - jumped.mean()),
                  'clips with a jump': f'{jumped.sum()} / {len(jumped)}',
                  'jumps at +10% (%)': 100*stretched[stretched['rate'] == 1.10]['jump'].mean()})

print(pd.DataFrame(table).round(1).to_string(index=False))
