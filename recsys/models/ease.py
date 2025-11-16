from typing import List, Optional, Dict, Any
import numpy as np
from ..base import BaseRecommender

class EASERecommender(BaseRecommender):
    """Заглушка под EASE. Реальную матрицу W можно обучить и сохранить в .npz."""

    def __init__(self):
        self.W: Optional[np.ndarray] = None

    def load(self, model_path: str):
        try:
            data = np.load(model_path)
            self.W = data["W"]
        except Exception:
            self.W = None  # пока нет модели

    def recommend(
        self,
        user_id: int,
        cart_items: List[int],
        k: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        # TODO: реализовать нормальную логику EASE
        # пока — просто возвращаем пустой список, чтобы не ломать пайплайн
        return []
