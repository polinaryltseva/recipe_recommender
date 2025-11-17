from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as multi


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
    Сравнение двух моделей с использованием парных статистических тестов
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
        raise ValueError(f"Unknown test type: {test_type}")
    
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

def bootstrap_confidence_interval(per_user_a: pd.DataFrame,per_user_b: pd.DataFrame,metric_name: str,n_bootstrap: int = 1000,confidence: float = 0.95) -> dict[str, Any]:
    """Бутстрап для доверительного интервала разницы метрик"""
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
            'bootstrap_diffs': np.array([]),
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
        'bootstrap_diffs': bootstrap_diffs,
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


def compare_multiple_models(per_user_metrics: dict[str, pd.DataFrame],metric_name: str,alpha: float = 0.05, correction_method: str = 'holm') -> pd.DataFrame:
    """
    Сравнение множественных моделей с коррекцией на множественное тестирование
    Args:
        per_user_metrics: словарь {имя_модели: DataFrame с метриками}
        metric_name: название метрики для сравнения
        alpha: уровень значимости
        correction_method: метод коррекции ('holm', 'bonferroni', 'fdr_bh')
    """
    
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
    
    results_df = pd.DataFrame(results)
    
    # Коррекция на множественное тестирование
    p_values = results_df['p_value'].values
    rejected, corrected_p_values, _, _ = multi.multipletests(
        p_values, alpha=alpha, method=correction_method
    )
    
    results_df['corrected_p_value'] = corrected_p_values
    results_df['statistically_significant'] = corrected_p_values < alpha
    
    return results_df.sort_values('corrected_p_value')


class ModelComparisonPipeline:
    """
    Унифицированный пайплайн для сравнения множественных моделей
    с коррекцией на множественное тестирование и оценкой практической значимости
    """
    
    def __init__(self, per_user_metrics: Dict[str, pd.DataFrame], 
                 alpha: float = 0.05, correction_method: str = 'holm',
                 mde: Optional[float] = None):
        """
        Args:
            per_user_metrics: словарь {имя_модели: DataFrame с метриками по пользователям}
            alpha: уровень значимости
            correction_method: метод коррекции ('holm', 'bonferroni', 'fdr_bh')
            mde: минимальный значимый эффект (Minimum Detectable Effect)
        """
        self.per_user_metrics = per_user_metrics
        self.alpha = alpha
        self.correction_method = correction_method
        self.mde = mde
        self.model_names = list(per_user_metrics.keys())
        self.results = {}
        
    def run_comparisons(self, metric_name: str, test_type: str = 'wilcoxon',
                       n_bootstrap: int = 1000) -> pd.DataFrame:
        """
        Запуск полного пайплайна сравнения моделей
        
        Returns:
            DataFrame с результатами сравнения всех пар моделей
        """
        # Шаг 1: Попарные сравнения и бутстрап
        pairwise_results = []
        bootstrap_results = {}
        
        for i, name_a in enumerate(self.model_names):
            for name_b in self.model_names[i + 1:]:
                # Статистический тест
                test_result = compare_models_paired(
                    self.per_user_metrics[name_a],
                    self.per_user_metrics[name_b],
                    metric_name,
                    test_type=test_type,
                    alpha=self.alpha,
                )
                
                # Бутстрап для доверительного интервала
                ci_result = bootstrap_confidence_interval(
                    self.per_user_metrics[name_a],
                    self.per_user_metrics[name_b],
                    metric_name,
                    n_bootstrap=n_bootstrap,
                    confidence=0.95,
                )
                
                pairwise_results.append({
                    'model_a': name_a,
                    'model_b': name_b,
                    'p_value': test_result['p_value'],
                    'mean_diff': test_result['mean_diff'],
                    'mean_a': test_result['mean_a'],
                    'mean_b': test_result['mean_b'],
                    'n_users': test_result['n_users'],
                    'lower_bound': ci_result['lower_bound'],
                    'upper_bound': ci_result['upper_bound'],
                    'ci_width': ci_result['upper_bound'] - ci_result['lower_bound'],
                })
                
                bootstrap_results[f"{name_a}_vs_{name_b}"] = ci_result['bootstrap_diffs']
        
        results_df = pd.DataFrame(pairwise_results)
        
        # Шаг 2: Коррекция на множественное тестирование
        p_values = results_df['p_value'].values
        rejected, corrected_p_values, _, _ = multi.multipletests(
            p_values, alpha=self.alpha, method=self.correction_method
        )
        
        results_df['corrected_p_value'] = corrected_p_values
        results_df['statistically_significant'] = corrected_p_values < self.alpha
        
        # Шаг 3: Оценка практической значимости
        if self.mde is not None:
            results_df['practically_significant'] = (
                results_df['lower_bound'] > self.mde
            )
            results_df['overall_significant'] = (
                results_df['statistically_significant'] & 
                results_df['practically_significant']
            )
        else:
            results_df['practically_significant'] = True
            results_df['overall_significant'] = results_df['statistically_significant']
        
        # Сохраняем результаты
        self.results[metric_name] = {
            'results_df': results_df,
            'bootstrap_dists': bootstrap_results
        }
        
        return results_df
    
    def get_significant_comparisons(self, metric_name: str) -> pd.DataFrame:
        """Получить только статистически значимые сравнения"""
        if metric_name not in self.results:
            raise ValueError(f"Сравнения для метрики {metric_name} не проводились")
        
        return self.results[metric_name]['results_df'][
            self.results[metric_name]['results_df']['overall_significant']
        ]