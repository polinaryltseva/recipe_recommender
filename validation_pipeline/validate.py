from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import sparse as sps
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from metrics import RankingMetrics

PredictFn = Callable[[Sequence[int]], Sequence[int]]

DEFAULT_STOP_WORDS = ['Соль', 'Сахар-песок', 'Перец черный молотый', 'Мука пшеничная', 'Сода', 'Сода гашеная уксусом']

def _parse_ingredients_cell(cell: Any) -> dict[str, Any]:
    if isinstance(cell, dict):
        return cell
    if isinstance(cell, str):
        try:
            return ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return {}
    if cell is None:
        return {}
    if isinstance(cell, (float, np.floating)) and np.isnan(cell):
        return {}
    if isinstance(cell, (list, tuple)):
        return {str(item): 1 for item in cell}
    if isinstance(cell, pd.Series):
        return cell.to_dict()
    if hasattr(cell, 'items'):
        return dict(cell)
    return {}


def build_user_item_interactions(
    data: pd.DataFrame,
    stop_words: Sequence[str],
    verbose: bool = True,
    recipe2id: dict[Any, int] | None = None,
    item2id: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[Any, int], dict[str, int], dict[int, Any], dict[int, str]]:
    interactions: list[tuple[Any, str]] = []
    iterator = tqdm(
        data.iterrows(), total=len(data), desc='Preparing interactions', disable=not verbose)
    stop_words_set = set(stop_words)

    for idx, row in iterator:
        ingredients_raw = row.get('ingredients_normalized')
        ingredients_parsed = _parse_ingredients_cell(ingredients_raw)
        recipe_id = row.get('url', idx)

        for ingredient in ingredients_parsed.keys():
            if ingredient in stop_words_set:
                continue
            interactions.append((recipe_id, ingredient))

    interactions_df = pd.DataFrame(interactions, columns=['recipe_id', 'ingredient_id'])
    unique_recipes = interactions_df['recipe_id'].unique()
    if recipe2id is None:
        recipe2id = {recipe: i for i, recipe in enumerate(unique_recipes)}
    else:
        missing_recipes = set(unique_recipes) - set(recipe2id.keys())
        if missing_recipes:
            raise ValueError(
                f'Provided recipe2id does not cover all recipes. Missing: {len(missing_recipes)} items.'
            )

    unique_ingredients = interactions_df['ingredient_id'].unique()
    if item2id is None:
        unique_ingredients = sorted(unique_ingredients)
        item2id = {ingredient: i for i, ingredient in enumerate(unique_ingredients)}
    else:
        missing_items = set(unique_ingredients) - set(item2id.keys())
        if missing_items:
            raise ValueError(
                f'Provided item2id does not cover all ingredients. Missing: {len(missing_items)} items.'
            )

    id2recipe = {i: recipe for recipe, i in recipe2id.items()}
    id2item = {i: ingredient for ingredient, i in item2id.items()}

    interactions_df['user_id'] = interactions_df['recipe_id'].map(recipe2id)
    interactions_df['item_id'] = interactions_df['ingredient_id'].map(item2id)
    interactions_df.dropna(subset=['item_id'], inplace=True)
    interactions_df['item_id'] = interactions_df['item_id'].astype(int)

    return interactions_df.reset_index(drop=True), recipe2id, item2id, id2recipe, id2item

def train_val_test_split(
    interactions_df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'item_id',
    k_core: int = 3,
    test_size: float = 0.2,
    val_size: float = 0.1,
    holdout_frac_range: tuple[float, float] = (0.3, 0.4),
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered_df = interactions_df.copy()

    while True:
        user_counts = filtered_df.groupby(user_col)[item_col].count()
        valid_users = user_counts[user_counts >= k_core].index

        item_counts = filtered_df.groupby(item_col)[user_col].count()
        valid_items = item_counts[item_counts >= k_core].index

        before_count = len(filtered_df)
        filtered_df = filtered_df[
            (filtered_df[user_col].isin(valid_users)) &
            (filtered_df[item_col].isin(valid_items))
        ]
        after_count = len(filtered_df)

        if before_count == after_count:
            break

    print(f'interactions left after k-core filtering: {len(filtered_df)}')

    rng = np.random.default_rng(seed)
    shuffled_users = rng.permutation(filtered_df[user_col].unique())

    n_users = len(shuffled_users)
    n_test = int(n_users * test_size)
    n_val = int(n_users * val_size)

    test_users = shuffled_users[:n_test]
    val_users = shuffled_users[n_test:n_test + n_val]
    train_users = shuffled_users[n_test + n_val:]

    train_df = filtered_df[filtered_df[user_col].isin(train_users)].copy()
    val_df = filtered_df[filtered_df[user_col].isin(val_users)].copy()
    test_df = filtered_df[filtered_df[user_col].isin(test_users)].copy()

    def _support_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df.copy(), df.copy()
        support_rows = []
        holdout_rows = []
        for _, user_df in df.groupby(user_col):
            user_df = user_df.sample(frac=1, random_state=rng.integers(0, 1_000_000))
            holdout_share = rng.uniform(*holdout_frac_range)
            n_holdout = min(len(user_df) - 1, max(1, int(round(len(user_df) * holdout_share))))
            holdout_rows.append(user_df.iloc[:n_holdout])
            support_rows.append(user_df.iloc[n_holdout:])
        return (
            pd.concat(support_rows).reset_index(drop=True),
            pd.concat(holdout_rows).reset_index(drop=True)
        )

    val_support_df, val_holdout_df = _support_holdout(val_df)
    test_support_df, test_holdout_df = _support_holdout(test_df)
    return (
        train_df.reset_index(drop=True),
        val_support_df,
        val_holdout_df,
        test_support_df,
        test_holdout_df,
    )


class RecommendationValidator:
    def __init__(
        self,
        data: pd.DataFrame | None = None,
        data_path: str | Path | None = None,
        stop_words: Sequence[str] | None = None,
        k_core: int = 5,
        test_size: float = 0.2,
        val_size: float = 0.15,
        holdout_frac_range: tuple[float, float] = (0.3, 0.4),
        seed: int = 42,
        verbose: bool = True,
        recipe2id: dict[Any, int] | None = None,
        item2id: dict[str, int] | None = None,
        metrics: RankingMetrics | None = None,
    ) -> None:
        if data is None:
            if data_path is None:
                data_path = Path(__file__).with_name('recipes_normalized.csv')
            data = pd.read_csv(data_path)

        self.raw_data = data
        self.stop_words = list(stop_words or DEFAULT_STOP_WORDS)
        self._stop_words_set = set(self.stop_words)
        self.verbose = verbose
        self.seed = seed
        self.metrics = metrics or RankingMetrics()

        (self.interactions_df, self.recipe2id, self.item2id, self.id2recipe, self.id2item) = build_user_item_interactions(
            self.raw_data, stop_words=self._stop_words_set, verbose=self.verbose, recipe2id=recipe2id, item2id=item2id)

        (self.train_df, self.val_support_df, self.val_holdout_df, self.test_support_df, self.test_holdout_df) = train_val_test_split(
            self.interactions_df, k_core=k_core, test_size=test_size, val_size=val_size, holdout_frac_range=holdout_frac_range, seed=seed)

        self.full_grouped_data = self._build_full_grouped_data()
        self._train_matrix: sps.coo_matrix | None = None

    def _build_full_grouped_data(self) -> pd.DataFrame:
        split_frames = {
            'train': (self.train_df, None),
            'val': (self.val_support_df, self.val_holdout_df),
            'test': (self.test_support_df, self.test_holdout_df),
        }
        rows = []
        for split_name, (support_df, holdout_df) in split_frames.items():
            support_grp = (
                support_df.groupby('user_id')['item_id'].apply(list)
                if support_df is not None and not support_df.empty
                else pd.Series(dtype=object)
            )
            holdout_grp = (
                holdout_df.groupby('user_id')['item_id'].apply(list)
                if holdout_df is not None and not holdout_df.empty
                else pd.Series(dtype=object)
            )
            combined_users = support_grp.index.union(holdout_grp.index)
            for user_id in combined_users:
                rows.append(
                    {
                        'user_id': user_id,'split': split_name,
                        'support_items': support_grp.get(user_id, []) or [],
                        'holdout_items': holdout_grp.get(user_id, []) or [],
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _ensure_list(items: Any) -> list[int]:
        if isinstance(items, list):
            return [int(i) for i in items]
        if isinstance(items, tuple):
            return [int(i) for i in items]
        if isinstance(items, np.ndarray):
            return [int(i) for i in items.tolist()]
        if items is None or (isinstance(items, float) and np.isnan(items)):
            return []
        return [int(items)]

    def get_grouped_split(self, split: str, only_with_holdout: bool = False) -> pd.DataFrame:
        df = self.full_grouped_data[self.full_grouped_data['split'] == split].copy().reset_index(drop=True)
        if only_with_holdout:
            df = df[df['holdout_items'].map(len) > 0].reset_index(drop=True)
        return df

    def build_train_matrix(self, fmt: str = 'coo') -> sps.spmatrix:
        if self._train_matrix is None:
            rows = self.train_df['user_id'].values
            cols = self.train_df['item_id'].values
            values = np.ones(len(self.train_df), dtype=np.float32)
            self._train_matrix = sps.coo_matrix((values, (rows, cols)), shape=(len(self.recipe2id), len(self.item2id)))
        if fmt == 'coo':
            return self._train_matrix
        if fmt == 'csr':
            return self._train_matrix.tocsr()
        if fmt == 'csc':
            return self._train_matrix.tocsc()
        raise ValueError("fmt must be one of {'coo', 'csr', 'csc'}")

    def _call_predictor(self, predictor: PredictFn, items: Sequence[int], top_k: int | None, predictor_kwargs: dict[str, Any]) -> list[int]:
        pred = predictor(items, **predictor_kwargs)
        if isinstance(pred, np.ndarray):
            pred_list = pred.tolist()
        else:
            pred_list = list(pred) if isinstance(pred, Sequence) else list(pred or [])
        pred_list = [int(p) for p in pred_list]
        if top_k is not None:
            pred_list = pred_list[:top_k]
        return pred_list

    def _generate_predictions(self,supports: pd.Series,predictor: PredictFn,top_k: int | None,predictor_kwargs: dict[str, Any] | None,show_progress: bool,desc: str) -> list[list[int]]:
        predictor_kwargs = predictor_kwargs or {}
        support_lists = [self._ensure_list(items) for items in supports.tolist()]
        iterator = tqdm(support_lists, total=len(support_lists), desc=desc, disable=not show_progress)  
        preds: list[list[int]] = []
        for items in iterator:
            preds.append(self._call_predictor(predictor, items, top_k, predictor_kwargs))
        return preds

    def make_split_predictions(self,split: str,predictor: PredictFn,preds_col: str = 'preds',top_k: int | None = None,predictor_kwargs: dict[str, Any] | None = None,show_progress: bool = True) -> pd.DataFrame:
        mask = self.full_grouped_data['split'] == split
        if not mask.any():
            raise ValueError(f"Unknown split '{split}'")

        supports = self.full_grouped_data.loc[mask, 'support_items']
        preds = self._generate_predictions(
            supports,
            predictor=predictor,
            top_k=top_k,
            predictor_kwargs=predictor_kwargs,
            show_progress=show_progress,
            desc=f'{split} predictions',
        )
        if preds_col not in self.full_grouped_data.columns:
            self.full_grouped_data[preds_col] = pd.Series([[] for _ in range(len(self.full_grouped_data))],dtype=object)

        preds_series = pd.Series([list(p) for p in preds], index=self.full_grouped_data.index[mask], dtype=object)
        self.full_grouped_data.loc[mask, preds_col] = preds_series.values
        return self.full_grouped_data.loc[mask, ['user_id', 'support_items', 'holdout_items', preds_col]].copy().reset_index(drop=True)

    def _cross_validated_metrics(self, df: pd.DataFrame, preds_col: str,top_k: int, n_folds: int,random_state: int | None) -> tuple[dict[str, float], list[dict[str, float]]]:
        unique_users = df['user_id'].unique()
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_metrics = []
        metric_keys: list[str] = []
        for fold_idx, (_, test_idx) in enumerate(kfold.split(unique_users), start=1):
            fold_users = unique_users[test_idx]
            fold_df = df[df['user_id'].isin(fold_users)]
            metrics = self.metrics.evaluate_frame(fold_df,preds_col,'holdout_items',top_k=top_k)
            metric_keys = list(metrics.keys())
            fold_metrics.append({'fold': fold_idx, **metrics})

        aggregated = {metric: float(np.mean([fm[metric] for fm in fold_metrics])) for metric in metric_keys}
        return aggregated, fold_metrics

    def evaluate_split(
        self,
        split: str,
        predictor: PredictFn,
        preds_col: str = 'preds',
        top_k: int = 20,
        cross_val_folds: int | None = None,
        random_state: int | None = None,
        predictor_kwargs: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        split_df = self.make_split_predictions(
            split=split,
            predictor=predictor,
            preds_col=preds_col,
            top_k=top_k,
            predictor_kwargs=predictor_kwargs,
            show_progress=show_progress,
        )
        metrics = self.metrics.evaluate_frame(split_df, preds_col, 'holdout_items', top_k=top_k)
        result: dict[str, Any] = {'split': split, 'metrics': metrics}
        if cross_val_folds and cross_val_folds > 1:
            aggregated, folds = self._cross_validated_metrics(split_df, preds_col, top_k, cross_val_folds, random_state or self.seed)
            result['cross_val_mean'] = aggregated
            result['fold_metrics'] = folds
        return result

    def cross_validate_predictions(
        self,
        split: str,
        preds_col: str,
        top_k: int = 20,
        n_folds: int = 5,
        random_state: int | None = None,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        """
        Reuse stored predictions on a split and run KFold metrics without recomputing them.
        """
        split_df = self.get_grouped_split(split, only_with_holdout=True)
        if preds_col not in split_df.columns:
            raise ValueError(f"Column '{preds_col}' not found. Generate predictions first.")
        rng = random_state if random_state is not None else self.seed
        return self._cross_validated_metrics(
            split_df,
            preds_col=preds_col,
            top_k=top_k,
            n_folds=n_folds,
            random_state=rng,
        )

    def autoregressive_validation(
        self,
        split: str,
        predictor: PredictFn,
        top_k: int = 20,
        max_steps: int | None = None,
        min_holdout_len: int = 3,
        predictor_kwargs: dict[str, Any] | None = None,
        show_progress: bool = True,
        return_detailed: bool = False,
    ) -> Any:
        df = self.get_grouped_split(split, only_with_holdout=True)
        df = df[df['holdout_items'].map(len) >= min_holdout_len].reset_index(drop=True)
        predictor_kwargs = predictor_kwargs or {}

        if df.empty:
            empty = pd.DataFrame(columns=['revealed_items', *self.metrics.get_metric_names(top_k)])
            return (empty, empty) if return_detailed else empty

        iterator = tqdm(df.itertuples(index=False), total=len(df), desc=f'{split} autoregressive', disable=not show_progress)
        details: list[dict[str, Any]] = []
        for row in iterator:
            base_context = self._ensure_list(row.support_items)
            holdout_seq = self._ensure_list(row.holdout_items)
            steps = len(holdout_seq)
            if max_steps is not None:
                steps = min(steps, max_steps)

            for reveal_count in range(1, steps):
                context = base_context + holdout_seq[:reveal_count]
                target = holdout_seq[reveal_count:]
                if not target:
                    break
                preds = self._call_predictor(predictor, context, top_k, predictor_kwargs)
                metrics = self.metrics.evaluate_sequence(preds, target, top_k=top_k)
                #print(metrics, preds)
                details.append({
                        'user_id': row.user_id,'revealed_items': reveal_count,
                        'context_size': len(context), **metrics
                    }
                )

        if not details:
            empty = pd.DataFrame(columns=['revealed_items', *self.metrics.get_metric_names(top_k)])
            return (empty, empty) if return_detailed else empty

        details_df = pd.DataFrame(details)
        metric_cols = self.metrics.get_metric_names(top_k)
        summary = details_df.groupby('revealed_items')[metric_cols].mean().reset_index().sort_values('revealed_items').reset_index(drop=True)

        if return_detailed:
            return summary, details_df
        return summary

    def get_per_user_metrics(
        self,
        split: str,
        preds_col: str,
        top_k: int | None = None,
    ) -> pd.DataFrame:
    
        from stats import compute_per_user_metrics

        split_df = self.get_grouped_split(split, only_with_holdout=True)
        if preds_col not in split_df.columns:
            raise ValueError(f"Column '{preds_col}' not found. Generate predictions first.")

        top_k = top_k or self.metrics.top_k
        return compute_per_user_metrics(
            split_df,
            preds_col,
            'holdout_items',
            top_k,
            self.metrics,
        )

    def inspect_user(self, user_id: int, split: str = 'test', preds_col: str = 'preds', top_k: int | None = 20) -> None:
        split_df = self.get_grouped_split(split)
        if preds_col not in split_df.columns:
            raise ValueError(f"Column '{preds_col}' not found. Generate predictions first.") 
        user_rows = split_df[split_df['user_id'] == user_id]
        if user_rows.empty:
            print(f"No data found for user {user_id} in split '{split}'.")
            return

        user_data = user_rows.iloc[0]
        recipe_name = self.id2recipe.get(user_id, 'Unknown Recipe')
        support_items = self._ensure_list(user_data.get('support_items'))
        holdout_items = self._ensure_list(user_data.get('holdout_items'))
        pred_items = self._ensure_list(user_data.get(preds_col))
        if top_k is not None:
            pred_items = pred_items[:top_k]
        support_names = [self.id2item.get(i, 'Unknown') for i in support_items]
        holdout_names = [self.id2item.get(i, 'Unknown') for i in holdout_items]
        pred_names = [self.id2item.get(i, 'Unknown') for i in pred_items]

        print('=' * 80)
        print(f"RECIPE: {recipe_name} (User ID: {user_id}) | split: {split}")
        print('=' * 80)
        print(f"Observed Ingredients ({len(support_names)})")
        print(', '.join(support_names))
        print(f"\nGround Truth Ingredients ({len(holdout_names)})")
        print(', '.join(holdout_names))
        print(f"\nTop Recommended Ingredients ({len(pred_names)})")
        print(', '.join(pred_names))
        print('=' * 80)