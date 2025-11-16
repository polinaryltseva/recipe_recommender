from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats


def compute_per_user_metrics(df: pd.DataFrame,preds_col: str,gt_col: str,top_k: int,metrics: Any) -> pd.DataFrame:
    per_user = []
    for _, row in df.iterrows():
        if len(row[gt_col]) == 0:
            continue
        user_metrics = metrics.evaluate_sequence(
            row[preds_col],
            row[gt_col],
            top_k=top_k,
        )
        user_metrics['user_id'] = row['user_id']
        per_user.append(user_metrics)

    return pd.DataFrame(per_user)


def check_normality(per_user_a: pd.DataFrame, per_user_b: pd.DataFrame, metric_name: str) -> dict[str, Any]:
    merged = pd.merge(
        per_user_a[['user_id', metric_name]],
        per_user_b[['user_id', metric_name]],
        on='user_id',
        suffixes=('_a', '_b'),
    )

    if len(merged) == 0:
        return {
            'shapiro_statistic': np.nan,
            'shapiro_p_value': np.nan,
            'is_normal': False,
            'n_users': 0,
        }

    diff = merged[f'{metric_name}_a'] - merged[f'{metric_name}_b']
    
    # Shapiro-Wilk test (n < 5000, otherwise use Kolmogorov-Smirnov)
    if len(diff) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(diff)
    else:
        shapiro_stat, shapiro_p = stats.kstest(
            (diff - diff.mean()) / diff.std(),'norm')
    return {
        'shapiro_statistic': float(shapiro_stat),
        'shapiro_p_value': float(shapiro_p),
        'is_normal': shapiro_p > 0.05,  
        'mean_diff': float(diff.mean()),
        'std_diff': float(diff.std()),
        'skewness': float(stats.skew(diff)),
        'kurtosis': float(stats.kurtosis(diff)),
        'n_users': len(merged)
    }

def compare_models_paired(per_user_a: pd.DataFrame,per_user_b: pd.DataFrame,metric_name: str,test_type: str = 'wilcoxon',alpha: float = 0.05) -> dict[str, Any]:
    """
    Compare two models using paired statistical tests
    Args:
        per_user_a: per-user metrics for model A
        per_user_b: per-user metrics for model B
        metric_name: name of metric column to compare 
        test_type: 'wilcoxon' (default, non-parametric) or 'ttest' (parametric)
        alpha: significance level
    """
    merged = pd.merge(
        per_user_a[['user_id', metric_name]],
        per_user_b[['user_id', metric_name]],
        on='user_id',
        suffixes=('_a', '_b'),
    )

    if len(merged) == 0:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'mean_diff': np.nan,
            'n_users': 0,
        }

    diff = merged[f'{metric_name}_a'] - merged[f'{metric_name}_b']
    mean_a = merged[f'{metric_name}_a'].mean()
    mean_b = merged[f'{metric_name}_b'].mean()
    mean_diff = mean_a - mean_b
    if test_type == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(
            merged[f'{metric_name}_a'],
            merged[f'{metric_name}_b'],
            alternative='two-sided',
        )
    elif test_type == 'ttest':
        statistic, p_value = stats.ttest_rel(
            merged[f'{metric_name}_a'],
            merged[f'{metric_name}_b'],
        )
    else:
        return None
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'mean_diff': float(mean_diff),
        'mean_a': float(mean_a),
        'mean_b': float(mean_b),
        'n_users': len(merged),
        'test_type': test_type,
    }

def bootstrap_confidence_interval(per_user_a: pd.DataFrame,per_user_b: pd.DataFrame,metric_name: str,n_bootstrap: int = 1000,confidence: float = 0.95) -> dict[str, float]:
    merged = pd.merge(
        per_user_a[['user_id', metric_name]],
        per_user_b[['user_id', metric_name]],
        on='user_id',
        suffixes=('_a', '_b'),
    )

    if len(merged) == 0:
        return {
            'mean_diff': np.nan,
            'lower_bound': np.nan,
            'upper_bound': np.nan,
        }
    diff = merged[f'{metric_name}_a'] - merged[f'{metric_name}_b']
    bootstrap_diffs = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        sample = diff.sample(n=len(diff), replace=True)
        bootstrap_diffs.append(sample.mean())

    bootstrap_diffs = np.array(bootstrap_diffs)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return {
        'mean_diff': float(diff.mean()),
        'lower_bound': float(lower),
        'upper_bound': float(upper),
        'confidence': confidence,
    }


def compare_both_tests(per_user_a: pd.DataFrame,per_user_b: pd.DataFrame,metric_name: str,alpha: float = 0.05) -> pd.DataFrame:
    normality = check_normality(per_user_a, per_user_b, metric_name)
    wilcoxon_result = compare_models_paired(
        per_user_a, per_user_b, metric_name, test_type='wilcoxon', alpha=alpha
    )
    ttest_result = compare_models_paired(
        per_user_a, per_user_b, metric_name, test_type='ttest', alpha=alpha
    )
    return pd.DataFrame([{
        'metric': metric_name,
        'n_users': normality['n_users'],
        'is_normal': normality['is_normal'],
        'normality_p_value': normality['shapiro_p_value'],
        'skewness': normality['skewness'],
        'kurtosis': normality['kurtosis'],
        'wilcoxon_p_value': wilcoxon_result['p_value'],
        'wilcoxon_significant': wilcoxon_result['significant'],
        'ttest_p_value': ttest_result['p_value'],
        'ttest_significant': ttest_result['significant'],
        'mean_diff': wilcoxon_result['mean_diff'],
        'mean_a': wilcoxon_result['mean_a'],
        'mean_b': wilcoxon_result['mean_b'],
    }])


def compare_multiple_models(per_user_metrics: dict[str, pd.DataFrame],metric_name: str,alpha: float = 0.05) -> pd.DataFrame:
    model_names = list(per_user_metrics.keys())
    results = []

    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            test_result = compare_models_paired(
                per_user_metrics[name_a],
                per_user_metrics[name_b],
                metric_name,
                test_type='wilcoxon',
                alpha=alpha,
            )
            results.append({
                'model_a': name_a,
                'model_b': name_b,
                'p_value': test_result['p_value'],
                'significant': test_result['significant'],
                'mean_diff': test_result['mean_diff'],
                'mean_a': test_result['mean_a'],
                'mean_b': test_result['mean_b'],
            })
    return pd.DataFrame(results).sort_values('p_value')
