from typing import List, Optional, Dict, Any
from ..base import BaseRecommender

class PopularityRecommender(BaseRecommender):
    """Простейший рекомендер по популярности (заглушка)."""

    def __init__(self):
        self.top_items: List[int] = []

    def load(self, model_path: str):
        # TODO: загрузить топ популярных товаров из файла/БД
        # пока — заглушка из нескольких item_id
        self.top_items = [1, 2, 3, 4, 5]

    def recommend(
        self,
        user_id: int,
        cart_items: List[int],
        k: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        return self.top_items[:k]
