import os
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# folds, models, nested cross-validation and the time-stretch evaluation

RANDOM_SEED = 2026
OUTER_FOLDS = 5
INNER_FOLDS = 3
VARIANTS = ['Default', 'Tuned']
METRICS = ['accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1']

SET_COLORS = {
    'Timbral': '#2a78d6',
    'Rhythm (as before)': '#eb6834',
    'Tempo only': '#1baf7a',
    'Rhythm, tempo removed': '#eda100',
    'Scale transform': '#e87ba4',
    'Rhythm, tempo removed + tempo': '#4a3aa7',
    'Timbral + rhythm': '#008300',
    'Timbral + rhythm, tempo removed': '#e34948',
    'Timbral + scale transform': '#4a3aa7',
}

KNOWN_REPLICAS = [   # known duplicate pairs (Fire-08 and Fire-09 have different labels)
    ('Albums-AnaBelen_Veneo-11.wav', 'Albums-Chrisanne2-12.wav'),
    ('Albums-Fire-08.wav', 'Albums-Fire-09.wav'),
    ('Albums-Latin_Jam2-05.wav', 'Albums-Latin_Jam2-13.wav'),
    ('Albums-Secret_Garden-01.wav', 'Media-104705.wav'),
    ('Albums-AnaBelen_Veneo-03.wav', 'Albums-AnaBelen_Veneo-15.wav'),
    ('Albums-Ballroom_Magic-03.wav', 'Albums-Ballroom_Magic-18.wav'),
    ('Albums-Latin_Jam-04.wav', 'Albums-Latin_Jam-13.wav'),
    ('Albums-Latin_Jam-08.wav', 'Albums-Latin_Jam-14.wav'),
    ('Albums-Latin_Jam-06.wav', 'Albums-Latin_Jam-15.wav'),
    ('Albums-Latin_Jam2-02.wav', 'Albums-Latin_Jam2-14.wav'),
    ('Albums-Latin_Jam2-07.wav', 'Albums-Latin_Jam2-15.wav'),
    ('Albums-Latin_Jam3-02.wav', 'Media-103414.wav'),
    ('Media-103402.wav', 'Media-103415.wav'),
]


def find_clips(dataset_dir):
    rows = []
    for folder, _, files in os.walk(dataset_dir):
        for name in files:
            if name.lower().endswith('.wav') and not name.startswith('.'):
                style = os.path.basename(folder)
                rows.append({'filename': name, 'folder': style,
                             'label': 'Rumba' if style.startswith('Rumba') else style})
    return pd.DataFrame(rows).sort_values('filename').reset_index(drop=True)


def make_folds(clips):
    folds = clips[['filename', 'label']].sort_values('filename').reset_index(drop=True)
    group_of = {name: name.removesuffix('.wav') for name in folds['filename']}
    for first, second in KNOWN_REPLICAS:
        if first in group_of and second in group_of:
            group_of[second] = group_of[first]
    folds['group_id'] = folds['filename'].map(group_of)
    splitter = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds['fold'] = -1
    for fold, (_, test_rows) in enumerate(splitter.split(folds, folds['label'], groups=folds['group_id']), start=1):
        folds.loc[test_rows, 'fold'] = fold
    return folds


MODELS = {
    'Random Forest': {
        'estimator': RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=1),
        'grid': {'max_features': ['sqrt', 0.5, 1.0],
                 'min_samples_leaf': [1, 2, 4, 8, 16],
                 'class_weight': [None, 'balanced']},
    },
    'SVM-RBF': {
        'estimator': Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))]),
        'grid': {'svm__C': [0.1, 1.0, 10.0, 100.0, 1000.0],
                 'svm__gamma': ['scale', 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                 'svm__class_weight': [None, 'balanced']},
    },
}


def classification_scores(y_true, y_pred, class_labels):
    def average(metric, how):
        return metric(y_true, y_pred, labels=class_labels, average=how, zero_division=0)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_precision': average(precision_score, 'macro'),
        'macro_recall': average(recall_score, 'macro'),
        'macro_f1': average(f1_score, 'macro'),
        'weighted_f1': average(f1_score, 'weighted'),
    }


def summarize_folds(fold_metrics, metrics=METRICS, by=('variant',)):
    summary = fold_metrics.groupby(list(by), sort=False)[metrics].agg(['mean', 'std'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    return summary.reset_index()


@dataclass
class CVResults:
    feature_columns: list
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    best_parameters: pd.DataFrame
    models: dict = field(repr=False)


def run_nested_cv(data, feature_columns, model_name, n_jobs=8, verbose=True):
    estimator, grid = MODELS[model_name]['estimator'], MODELS[model_name]['grid']
    class_labels = np.sort(data['label'].unique())
    fold_rows, parameter_rows, models = [], [], {}
    predictions = data[['filename', 'label', 'group_id', 'fold']].copy()
    for variant in VARIANTS:
        predictions[f'prediction_{variant.lower()}'] = ''

    for fold in sorted(data['fold'].unique()):
        train, test = data['fold'] != fold, data['fold'] == fold
        X_train, y_train = data.loc[train, feature_columns], data.loc[train, 'label']
        X_test, y_test = data.loc[test, feature_columns], data.loc[test, 'label']
        assert not set(data.loc[train, 'group_id']) & set(data.loc[test, 'group_id'])

        default = clone(estimator).fit(X_train, y_train)
        inner_cv = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED + fold)
        search = GridSearchCV(estimator=estimator, param_grid=grid, scoring='f1_macro', cv=inner_cv,
                              n_jobs=n_jobs, error_score='raise')
        search.fit(X_train, y_train, groups=data.loc[train, 'group_id'])
        tuned = search.best_estimator_

        models[fold] = {'Default': default, 'Tuned': tuned}
        for variant, model in models[fold].items():
            predicted = model.predict(X_test)
            predictions.loc[test, f'prediction_{variant.lower()}'] = predicted
            fold_rows.append({'fold': fold, 'variant': variant, **classification_scores(y_test, predicted, class_labels)})
        chosen = {name: ('None' if value is None else value) for name, value in search.best_params_.items()}
        parameter_rows.append({'fold': fold, 'inner_best_macro_f1': search.best_score_, **chosen})
    if verbose:
        tuned_f1 = [row['macro_f1'] for row in fold_rows if row['variant'] == 'Tuned']
        print(f'    tuned macro-F1 per fold: {", ".join(f"{f:.3f}" for f in tuned_f1)}', flush=True)
    return CVResults(feature_columns, pd.DataFrame(fold_rows), predictions, pd.DataFrame(parameter_rows), models)


def evaluate_conditions(result, data, versions, conditions):
    class_labels = np.sort(data['label'].unique())
    by_condition = {name: rows.set_index('filename') for name, rows in versions.groupby('condition')}
    columns = result.feature_columns
    metric_rows, prediction_rows = [], []
    for fold, fold_models in result.models.items():
        test_files = data.loc[data['fold'] == fold, 'filename']
        y_test = data.loc[data['fold'] == fold, 'label'].to_numpy()
        for variant, model in fold_models.items():
            baseline = model.predict(by_condition['original'].loc[test_files, columns])
            baseline_scores = classification_scores(y_test, baseline, class_labels)
            for condition in conditions:
                predicted = model.predict(by_condition[condition['condition']].loc[test_files, columns])
                scores = classification_scores(y_test, predicted, class_labels)
                metric_rows.append({'fold': fold, 'variant': variant, **condition, **scores,
                                    'flip_rate': np.mean(predicted != baseline),
                                    'macro_f1_change_pp': 100*(scores['macro_f1'] - baseline_scores['macro_f1'])})
                if variant == 'Tuned':
                    prediction_rows.append(pd.DataFrame({'filename': test_files.to_numpy(), 'label': y_test,
                                                         'fold': fold, 'condition': condition['condition'],
                                                         'prediction': predicted}))
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def tempo_effect_row(condition_metrics):
    tuned = summarize_folds(condition_metrics[condition_metrics['variant'] == 'Tuned'],
                            ['macro_f1', 'flip_rate', 'macro_f1_change_pp'], by=('condition', 'kind'))
    tuned = tuned.set_index('condition')
    stretch = tuned.loc[tuned['kind'] == 'stretch', 'macro_f1_change_pp_mean'].mean()
    roundtrip = tuned.loc[tuned['kind'] == 'roundtrip', 'macro_f1_change_pp_mean'].mean()
    return {'original macro-F1 (%)': 100*tuned.loc['original', 'macro_f1_mean'],
            'at -10% (%)': 100*tuned.loc['stretch_0.90', 'macro_f1_mean'],
            'at +10% (%)': 100*tuned.loc['stretch_1.10', 'macro_f1_mean'],
            'mean change, stretch (pp)': stretch,
            'mean change, round trip (pp)': roundtrip,
            'caused by tempo (pp)': stretch - roundtrip,
            'flip rate, stretch (%)': 100*tuned.loc[tuned['kind'] == 'stretch', 'flip_rate_mean'].mean()}


def group_importance(result, data, groups, repeats=5):
    rows = []
    for fold, fold_models in result.models.items():
        test = data['fold'] == fold
        X_test, y_test = data.loc[test, result.feature_columns], data.loc[test, 'label']
        model = fold_models['Tuned']
        baseline = f1_score(y_test, model.predict(X_test), average='macro', zero_division=0)
        for family, columns in groups.items():
            for repeat in range(repeats):
                order = np.random.default_rng(RANDOM_SEED + 100*fold + repeat).permutation(len(X_test))
                shuffled = X_test.copy()
                shuffled[columns] = X_test[columns].to_numpy()[order]
                score = f1_score(y_test, model.predict(shuffled), average='macro', zero_division=0)
                rows.append({'fold': fold, 'feature_family': family, 'macro_f1_drop': baseline - score})
    per_fold = pd.DataFrame(rows).groupby(['fold', 'feature_family'])['macro_f1_drop'].mean()
    return per_fold.groupby('feature_family').agg(['mean', 'std']).sort_values('mean', ascending=False)


def plot_robustness(condition_metrics, set_colors, model_name, axes):
    for axis, kind, title in zip(axes, ['stretch', 'roundtrip'], ['stretched (tempo changed)', 'round trip (tempo restored)']):
        for set_name, color in set_colors.items():
            metrics = condition_metrics[set_name]
            summary = summarize_folds(metrics[metrics['variant'] == 'Tuned'], ['macro_f1'], by=('kind', 'rate'))
            rows = summary[summary['kind'].isin([kind, 'original'])].sort_values('rate')
            axis.errorbar(100*(rows['rate'] - 1), 100*rows['macro_f1_mean'], yerr=100*rows['macro_f1_std'],
                          marker='o', capsize=3, linewidth=2, color=color, label=set_name)
        axis.set(title=f'{model_name}, tuned: {title}', xlabel='Stretch applied (%)',
                 ylabel='Macro-F1 (%), mean ± SD', ylim=(0, 100))
        axis.set_xticks([-10, -5, -2, 0, 2, 5, 10])
        axis.axhline(12.5, color='#8a8984', linestyle=':', linewidth=1)
        axis.grid(alpha=0.3)
    axes[0].legend(loc='lower left', fontsize=8)
