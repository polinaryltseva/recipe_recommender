from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


class EaseOfflineScorer:
    """Minimal EASE scorer that works with the exported pickle artifact."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}
        self.weights: np.ndarray | None = None
        self._load()

    def _load(self):  # загружает модель
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)

        data = (
            payload if isinstance(payload, dict) else getattr(payload, "__dict__", {})
        )

        value = data.get("model_weights")
        matrix = value
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        self.weights = np.asarray(matrix)  # матрица весов в numpy массив

        value = data.get("item2id")
        mapping = value  # маппинг item -> index

        normalized = {int(k): int(v) for k, v in mapping.items()}
        self.item2idx = normalized
        self.idx2item = {int(k): int(v) for k, v in data["id2item"].items()}

    # рекомендует товары, которые похожи на те, что в корзине
    def recommend(
        self, cart_items: Sequence[int], top_k: int = 50
    ) -> List[tuple[int, float]]:
        if not cart_items or self.weights is None:
            return []

        # Конвертируем pov_id в индексы строк/столбцов. Если каких-то id нет в модели — пропускаем
        seed_idx = [
            self.item2idx.get(int(item)) for item in cart_items
        ]  # получаем индексы строк/столбцов
        seed_idx = [idx for idx in seed_idx if idx is not None]
        if not seed_idx:
            return []

        # Суммируем веса по строкам, чтобы получить рейтинг для каждого товара
        scores = self.weights[seed_idx].sum(axis=0)
        if scores.ndim > 1:
            scores = scores.sum(axis=0)
        scores = np.asarray(scores, dtype=np.float64)

        # не рекомендуем то, что уже в корзине
        scores[seed_idx] = -np.inf

        k = min(top_k, scores.size)
        if k <= 0:
            return []

        # берем top_k наибольших значений
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.idx2item[idx], float(scores[idx])) for idx in top_idx]
