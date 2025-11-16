from typing import Dict
from .models.popularity import PopularityRecommender
from .models.ease import EASERecommender
# from .models.sasrec import SASRecRecommender
# from .models.als import ALSRecommender
from recsys.models.ease_popular.inference import EasePopularRecommender

import logging
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
}

DEFAULT_EXPERIMENT = {
    "name": "main_recsys_ab",
    "variants": {
        "ease_popular_v1": 1.0,
    },
}

from .ab_testing import choose_variant_for_user  # noqa


def get_recommender_for_user(user_id: int):
    variant = choose_variant_for_user(user_id, DEFAULT_EXPERIMENT)

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
