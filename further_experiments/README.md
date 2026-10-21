# Rhythm without the tempo, and timbre + rhythm

The notebooks in this folder test two more groups of feature sets: rhythm features with the tempo removed (a shifted cyclic tempogram and a scale transform), and the timbral features together with the rhythm features. They use the same data, folds, classifiers, tuning and time-stretching as the notebooks in the main folder.

The method and the results are described in the [main README](../README.md): the feature sets in section 4, the results in sections 7.6 and 7.7.

```
further_experiments/
├── Rhythm_Without_Tempo.ipynb   rhythm features without the tempo (section 7.6)
├── Timbre_Plus_Rhythm.ipynb     timbre and rhythm together (section 7.7)
├── features.py                  features of every clip in all 13 versions
├── evaluation.py                folds, classifiers, nested cross-validation, time-stretch evaluation
└── results/                     tables and figures of both notebooks
```

Run the notebooks from inside this folder, with `DATASET_DIR` pointing to the `BallroomData` folder. The first notebook computes the features of all versions and saves them in `all_versions_features.csv` (about 20 minutes, not stored in git); the second one reuses that file.
