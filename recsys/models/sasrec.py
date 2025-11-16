from ..base import BaseRecommender

class SASRecRecommender(BaseRecommender):
    # TODO: реализовать последовательную модель рекомендаций
    def load(self, model_path: str):
        pass

    def recommend(self, user_id, cart_items, k=10, context=None):
        return []
