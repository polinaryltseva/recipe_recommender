from typing import Dict
from .models.popularity import PopularityRecommender
from .models.ease import EASERecommender
# from .models.sasrec import SASRecRecommender
# from .models.als import ALSRecommender
from recsys.models.ease_popular.inference import EasePopularRecommender
from recsys.models.llm_recs import LLMRecommender
from recsys.models.knn_recipes.wrap import KNNWrapperRecommender 

import logging
logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Dict] = {
    "ease_popular": {
        "class": EasePopularRecommender,
        "path": "models/ease_popular",
    },
    "llm_recs": {
        "class": LLMRecommender,
        "path": "models/llm_recs",
    },
    "knn_recipes": {
        "class": KNNWrapperRecommender,
        "path": "models/knn_recipes",  # папка с knn_model.pkl + p2v/v2p, как ты сделал
    },
}



from .ab_testing import choose_variant_for_user  # noqa


# def get_recommender_for_user(user_id: int):
#     variant = choose_variant_for_user(user_id, DEFAULT_EXPERIMENT)

#     cfg = MODEL_REGISTRY[variant]
#     cls = cfg["class"]

#     logger.info(
#         "get_recommender_for_user: user_id=%s, experiment=%s, variant=%s, model_class=%s, model_path=%s",
#         user_id,
#         DEFAULT_EXPERIMENT["name"],
#         variant,
#         cls.__name__,
#         cfg.get("path"),
#     )

#     model = cls()
#     model.load(cfg["path"])
#     return model

class RecModel:
    def __init__(self):
        

        # cfg = MODEL_REGISTRY[variant]
        # cls = cfg["class"]

        # logger.info(
        #     "get_recommender_for_user: user_id=%s, experiment=%s, variant=%s, model_class=%s, model_path=%s",
        #     user_id,
        #     DEFAULT_EXPERIMENT["name"],
        #     variant,
        #     cls.__name__,
        #     cfg.get("path"),
        # )

        # model = cls()
        # model.load(cfg["path"])

        # EASE
        
        self.ease_model = EasePopularRecommender()
        # self.ease_popular = ease_model.load('models/ease_popular')

        self.llm_recs = LLMRecommender()
        
        self.knn_recs = KNNWrapperRecommender()
        self.knn_recs.load('models/knn_recipes')
        # self.llm_recs = llm_model.load('models/llm_recs')
    def models_dict_generator(self):
        models_dict = {'ease_popular': self.ease_model,'llm_recs': self.llm_recs, 'knn_recipes': self.knn_recs}
        return models_dict
    

rec_model = RecModel()
models_dict = rec_model.models_dict_generator()
print(models_dict)

