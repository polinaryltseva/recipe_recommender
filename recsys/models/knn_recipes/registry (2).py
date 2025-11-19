from typing import Dict
import logging

from .models.popularity import PopularityRecommender
from .models.ease import EASERecommender
# from .models.sasrec import SASRecRecommender
# from .models.als import ALSRecommender
from recsys.models.ease_popular.inference import EasePopularRecommender
from recsys.models.knn_recipes import KNNRecipesRecommender  # наш KNN

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Dict] = {
    "ease_popular_v1": {
        "class": EasePopularRecommender,
        "path": "models/ease_popular",  # сейчас по сути не используется, но оставляем
    },
    # "baseline_pop": {
    #     "class": PopularityRecommender,
    #     "path": "models/popularity/baseline.pkl",
    # },
    # "ease_v1": {
    #     "class": EASERecommender,
    #     "path": "models/ease/ease_v1.npz",
    # },
    # "sasrec_v1": {
    #     "class": SASRecRecommender,
    #     "path": "models/sasrec/sasrec_v1.pt",
    # },
    "knn_recipes_v1": {
        "class": KNNRecipesRecommender,
        "path": "models/knn_recipes",  # папка с knn_model.pkl + p2v/v2p, как ты сделал
    },
}

# Один-единственный эксперимент над реестром
DEFAULT_EXPERIMENT = {
    "name": "main_recsys_ab",
    "variants": {
        # сейчас весь трафик на KNN, чтобы потестить его предсказания
        "ease_popular_v1": 0.0,
        "knn_recipes_v1": 1.0,
    },
}

from .ab_testing import choose_variant_for_user  # noqa


def get_recommender_for_user(user_id: int):
    """
    Выбирает вариант модели по A/B-эксперименту
    и инициализирует соответствующий рекоммендер.
    """
    variant = choose_variant_for_user(user_id, DEFAULT_EXPERIMENT)

    if variant not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model variant: {variant}")

    cfg = MODEL_REGISTRY[variant]
    cls = cfg["class"]

    logger.info(
        "get_recommender_for_user: user_id=%s, experiment=%s, variant=%s, model_class=%s, model_path=%s",
        user_id,
        DEFAULT_EXPERIMENT["name"],
        variant,
        cls.__name__,
        cfg.get("path"),
    )

    model = cls()
    model.load(cfg["path"])
    return model
