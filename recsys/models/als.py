from ..base import BaseRecommender

class ALSRecommender(BaseRecommender):
    # TODO: реализовать ALS / MF при надобности
    def load(self, model_path: str):
        pass

    def recommend(self, user_id, cart_items, k=10, context=None):
        return []
