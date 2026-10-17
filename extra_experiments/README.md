# Extra experiments

## In short

**Why we did this.** In the main experiment the rhythm models got about 80% but collapsed when the tempo changed, because our cyclic tempogram still told the model the tempo. So we didn't know yet if rhythm itself can recognise the dances. We also wanted to know if adding timbre makes a model more robust.

**What we changed.** We removed the tempo from the rhythm features in two ways (a shifted cyclic tempogram and a scale transform, explained below), and we trained models on timbre and rhythm together. Data, folds, classifiers, tuning and time-stretching are the same as in the main experiment.

**How to read the numbers.** All scores are macro-F1 in %: the F1 score of each dance, averaged over the 8 dances, so every dance counts the same (chance is 12.5%). Changes are in percentage points (pp), the plain difference between two percentages: going from 80% to 50% is a drop of 30 pp.

**What we found:**

- **Rhythm without the tempo still gets 65-72% macro-F1 (chance is 12.5%, tempo alone 60%), and the drop caused by a tempo change goes from about 31 pp to about 10 pp.** So the rhythm pattern carries real information, and the tempo is a shortcut on top of it.
- **Timbre + tempo-free rhythm is the most robust model we have: 72-74% on the original audio and only 4-5 pp lost to tempo changes (69-70% at +10% tempo, against 24-26% for timbre + the normal rhythm features).**
- **Adding timbre to the normal rhythm features gives the best score (86%) but does not remove the shortcut.** The model still relies mostly on the tempo and collapses at +10%.
- The scale transform ignores the tempo but is very sensitive to the stretching artifacts, so a feature that is tempo-invariant on paper is not automatically robust.

**Main message for the paper:** on Ballroom the tempo is worth about 5-15 pp of macro-F1. The rest can be reached without it, but when the tempo is available, the models prefer it, even when other useful features are there.

<p align="center"><img src="results/summary_tradeoff.png" width="760" alt="Scatter plot of every feature set: macro-F1 on the original audio against the mean macro-F1 on stretched audio"></p>

Every dot is one feature set (mean of Random Forest and SVM). Right means more accurate on the original audio, and up means it still works when the tempo changes. A dot on the dashed line loses nothing when the music is stretched. The old rhythm features (grey, bottom right) are accurate but fragile. The two blue dots at the top, timbre with tempo-free rhythm, give the best balance.

## The two tempo-free rhythm features

**Shifted cyclic tempogram.** Our cyclic tempogram has 15 bins, and the bin with the most energy tells the model the tempo class of the clip. We rotate the profile so that this bin is always the first one. The shape of the profile stays, but the tempo class is gone. This was already an option in `Rythm_extraction.py` (`shift=True`).

**Scale transform** (Holzapfel & Stylianou, 2011). The autocorrelation of the novelty curve shows after which time lags the music repeats itself (one beat, two beats, one bar). When the music is played faster, this curve gets squeezed but keeps its shape. The scale transform describes that shape and ignores the squeeze, so it doesn't change when the music is stretched. It is the same idea as Jensen et al. (2009) from our proposal.

We checked that both really ignore the tempo: under a ±10% stretch the old cyclic tempogram changes a lot, while the two new features stay almost the same (cosine similarity above 0.98).

## Results

Tuned models, macro-F1 in %, Random Forest / SVM, mean over the 5 test folds. "Drop caused by tempo" is the drop under stretching minus the drop in the round trip (stretched and stretched back), the same measure as in the main README.

| Feature set | Original audio | At +10% tempo | Drop caused by tempo (pp) |
|---|---:|---:|---:|
| Timbral | 46 / 47 | 45 / 47 | none |
| Tempo only | 59 / 60 | 9 / 9 | −35 / −38 |
| Rhythm (as before) | 81 / 80 | 16 / 18 | −31 / −31 |
| Rhythm, tempo removed (shifted tempogram) | 65 / 67 | 57 / 53 | −9 / −11 |
| Scale transform | 69 / 72 | 46 / 47 | none |
| Rhythm, tempo removed + tempo added back | 78 / 71 | 30 / 54 | −21 / −13 |
| Timbral + rhythm | 86 / 86 | 26 / 24 | −25 / −24 |
| Timbral + rhythm, tempo removed | 72 / 74 | 69 / 70 | −5 / −4 |
| Timbral + scale transform | 76 / 81 | 60 / 65 | none |

What the numbers mean:

- **The shifted tempogram still drops a little, but not because of the tempo.** One bin covers about 4.7% of tempo. A 5% or 10% stretch moves the profile by almost whole bins, which the rotation removes, but a 2% stretch moves it by half a bin and changes its shape. That's why this model is worse at ±2% than at ±5%.
- **The scale transform's drop comes from the stretching itself.** It loses even more in the round trip, where the tempo is unchanged, than under stretching.
- **The combined model ignores the timbre when it can read the tempo.** Shuffling the cyclic tempogram costs timbre + rhythm about 60 pp of macro-F1, and shuffling the MFCCs only 8 pp (Random Forest) to 26 pp (SVM). Without the tempo, the MFCCs become about twice as important.
- **When the tempo is added back to the tempo-free rhythm features, the Random Forest relies on it again and becomes fragile again.**

## A note on the bar ratios

The main notebooks say the bar ratios don't depend on the tempo. That is not fully true: on a click track with equally strong beats they should all be 1, but they range from 0.6 to 1.1 depending on the tempo. We left them unchanged, because they matter little in every model (at most about 7 pp of macro-F1 when shuffled).

## Files and how to run

```
extra_experiments/
├── features.py                  features of every clip in all 13 versions
├── evaluation.py                folds, classifiers, nested cross-validation, time-stretch evaluation
├── Rhythm_Without_Tempo.ipynb   the tempo-free rhythm features
├── Timbre_Plus_Rhythm.ipynb     timbre and rhythm together
└── results/                     tables and figures of both notebooks
```

`features.py` and `evaluation.py` contain the code of the main notebooks plus the two new features; the reference rows give the same numbers as the main results. Run the notebooks from inside this folder, with `DATASET_DIR` pointing to the `BallroomData` folder. The first notebook computes the features of all versions and saves them in `all_versions_features.csv` (not stored in git); the second one reuses that file.

## Next steps

- More bins per octave in the cyclic tempogram, to remove the dip at ±2%.
- A scale transform that is less sensitive to the stretching artifacts.
- Folds grouped by album (the 698 clips come from 53 albums).
- Training on stretched copies (tempo augmentation).

## References

- A. Holzapfel, Y. Stylianou. *Scale transform in rhythmic similarity of music.* IEEE Transactions on Audio, Speech, and Language Processing, 2011.
- J. H. Jensen, M. G. Christensen, S. H. Jensen. *A tempo-insensitive representation of rhythmic patterns.* EUSIPCO, 2009.
