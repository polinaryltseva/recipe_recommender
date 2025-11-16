from typing import List, Optional, Dict, Any

class BaseRecommender:
    """Базовый интерфейс для всех рекомендательных моделей."""

    def load(self, model_path: str):
        raise NotImplementedError

    def recommend(
        self,
        user_id: int,
        cart_items: List[int],
        k: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Вернуть топ-k item_id в порядке убывания релевантности."""
        raise NotImplementedError
