from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from ...base import BaseRecommender


class KNNRecipesRecommender(BaseRecommender):
    """
    На вход: список product_id (VV).
    На выход: список product_id (VV).
    Внутри: пространство индексов ингредиентов (pov_idx).
    """

    def __init__(self) -> None:
        print(1)
        self.nn: Optional[NearestNeighbors] = None
        self.X: Optional[csr_matrix] = None

        # VV product_id -> ing_idx (pov_idx)
        self.ing2idx: Dict[int, int] = {}

        # ing_idx -> VV product_id (если нужен)
        self.idx2ing: Dict[int, int] = {}

        # ing_idx -> VV product_id ИЛИ список VV product_id
        self.ing2products: Dict[int, Any] = {}

        self.n_neighbors: int = 50
        self.n_items: int = 0  # число колонок в X

    # ---------------------------------------------------------
    # ЗАГРУЗКА ВЕСОВ
    # ---------------------------------------------------------

    def load(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        with open(model_dir / "knn_model.pkl", "rb") as f:
            data = pickle.load(f)

        self.nn = data["nn"]
        self.X = data["X"]

        # аккуратно приводим ключи/значения к int
        self.ing2idx = {int(k): int(v) for k, v in data["ing2idx"].items()}
        self.idx2ing = {int(k): int(v) for k, v in data["idx2ing"].items()}
        self.ing2products = {int(k): v for k, v in data["ing2products"].items()}
        self.n_neighbors = int(data.get("n_neighbors", 50))

        # ВАЖНО: число признаков берём из X
        self.n_items = self.X.shape[1]

    # ---------------------------------------------------------
    # ОСНОВНОЙ МЕТОД РЕКОМЕНДАЦИЙ
    # ---------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        cart_items: List[int],
        k: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """
        cart_items: список product_id ВкусВилла (int)
        return: список product_id ВкусВилла (int)
        """
        if self.X is None or self.nn is None:
            return []

        if not cart_items:
            return []

        # 1) VV product_id -> ing_idx
        ing_idxs = self._products_to_ingidx(cart_items)
        if not ing_idxs:
            return []

        # 2) one-hot вектор пользователя в пространстве ингредиентов
        user_vec = self._build_vec(ing_idxs)

        # 3) ближайшие рецепты по KNN
        recipe_idxs = self._find_recipes(user_vec)

        # 4) недостающие ингредиенты
        missing_ing = self._missing_ingredients(ing_idxs, recipe_idxs)

        # 5) недостающие ингредиенты -> товары ВкусВилла
        rec_products = self._ingredients_to_products(missing_ing)

        # 6) выкидываем товары, которые уже в корзине
        cart_set = set(cart_items)
        final = [p for p in rec_products if p not in cart_set]
        print(final)
        return final[:k]

    # ---------------------------------------------------------
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ---------------------------------------------------------

    def _products_to_ingidx(self, product_ids: List[int]) -> List[int]:
        """
        VV product_id -> ing_idx (pov_idx).
        """
        result: List[int] = []
        for pid in product_ids:
            idx = self.ing2idx.get(int(pid))
            if idx is not None:
                result.append(idx)
        return result

    def _build_vec(self, ing_idxs: List[int]) -> np.ndarray:
        """
        ONE-HOT вектор пользователя длины n_items.
        ВАЖНО: длина = числу колонок X (self.n_items), а не len(idx2ing).
        """
        vec = np.zeros((1, self.n_items), dtype=np.float32)
        for idx in ing_idxs:
            if 0 <= idx < self.n_items:
                vec[0, idx] = 1.0
        return vec

    def _find_recipes(self, vec: np.ndarray) -> List[int]:
        """
        Ищем ближайшие рецепты в пространстве X.
        """
        dists, idxs = self.nn.kneighbors(vec, n_neighbors=self.n_neighbors)
        return idxs[0].tolist()

    def _missing_ingredients(
        self,
        ing_idxs: List[int],
        recipe_idxs: List[int],
    ) -> List[int]:
        """
        Считаем, какие ингредиенты чаще всего встречаются в соседних рецептах,
        но отсутствуют в корзине.
        """
        user_set = set(ing_idxs)
        scores: Dict[int, float] = {}

        for r in recipe_idxs:
            row = self.X[r]
            if hasattr(row, "nonzero"):
                items = row.nonzero()[1]
            else:
                items = np.where(row > 0)[0]

            for ing in items:
                if ing in user_set:
                    continue
                scores[ing] = scores.get(ing, 0.0) + 1.0

        # сортируем по убыванию счёта
        sorted_ing = sorted(scores.keys(), key=lambda i: -scores[i])
        return sorted_ing

    def _ingredients_to_products(self, ing_idxs: List[int]) -> List[int]:
        """
        ing_idx -> product_id ВкусВилла.
        Поддерживаем и int, и список int в значениях self.ing2products.
        """
        results: List[int] = []
        for idx in ing_idxs:
            if idx not in self.ing2products:
                continue
            val = self.ing2products[idx]
            if isinstance(val, list):
                results.extend(int(v) for v in val)
            else:
                results.append(int(val))

        # remove duplicates, preserve order
        seen = set()
        out: List[int] = []
        for p in results:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out
