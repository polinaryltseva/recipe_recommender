from typing import List, Optional, Dict, Any

from recsys.base import BaseRecommender
from .inference import KNNRecipesRecommender 
import logging

logger = logging.getLogger(__name__)


class KNNWrapperRecommender(BaseRecommender):
    """
    Обёртка для KNNRecipesRecommender, чтобы он соответствовал
    общему интерфейсу RecSys-case.
    """

    def __init__(self):
        """
        Инициализирует обёртку. Внутренняя система (KNN) ещё не создана.
        """
        self.system: Optional[KNNRecipesRecommender] = None

    def load(self, model_path: str):
        """
        Загружает модель KNN.

        В отличие от EASE, здесь model_path является обязательным и указывает
        на директорию с файлом 'knn_model.pkl' и другими данными.
        """
        if self.system is None:
            logger.info(
                "KNNWrapper.load: initializing KNNRecipesRecommender (model_path=%s)",
                model_path,
            )
            # 1. Создаём экземпляр основной системы
            self.system = KNNRecipesRecommender()
            # 2. Вызываем её собственный метод load для загрузки весов
            self.system.load(model_path)
        else:
            logger.debug("KNNWrapper.load: system already initialized, skipping re-init")

        # Логируем информацию о готовности модели
        n_items = getattr(self.system, "n_items", "unknown")
        n_neighbors = getattr(self.system, "n_neighbors", "unknown")
        logger.info(
            "KNNWrapper: model ready, n_items=%s, n_neighbors=%s",
            n_items,
            n_neighbors,
        )

    def recommend(
        self,
        user_id: int,
        cart_items: List[int],
        k: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """
        Основной метод для получения рекомендаций.
        Просто проксирует вызов к методу recommend из KNNRecipesRecommender.
        """
        logger.debug(
            "KNNWrapper.recommend: user_id=%s, cart_items=%s, k=%s, context_keys=%s",
            user_id,
            cart_items,
            k,
            list(context.keys()) if isinstance(context, dict) else None,
        )

        # Обработка случая, когда корзина пуста
        if not cart_items:
            logger.info(
                "KNNWrapper.recommend: empty cart_items for user_id=%s -> returning empty list",
                user_id,
            )
            return []

        # Критически важно: система должна быть загружена до вызова recommend.
        # В отличие от EASE, мы не можем инициализировать её "на лету",
        # так как нам нужен model_path, которого здесь нет.
        if self.system is None:
            logger.error(
                "KNNWrapper.recommend: system is None! You must call .load(model_path) before "
                "using .recommend(). Returning empty list for user_id=%s.",
                user_id,
            )
            return []

        try:
            # Интерфейс KNNRecipesRecommender.recommend полностью совпадает с
            # интерфейсом этой обёртки, поэтому мы просто передаём все аргументы дальше.
            rec_product_ids = self.system.recommend(
                user_id=user_id,
                cart_items=cart_items,
                k=k,
                context=context,
            )
        except Exception:
            logger.exception(
                "KNNWrapper.recommend: error in KNNRecipesRecommender.recommend "
                "(user_id=%s, cart_items=%s, k=%s)",
                user_id,
                cart_items,
                k,
            )
            return []

        # Гарантируем, что результат является списком (хотя он и так должен быть)
        if not isinstance(rec_product_ids, list):
            try:
                rec_product_ids = list(rec_product_ids)
            except TypeError:
                logger.error(
                    "KNNWrapper.recommend: cannot convert recommendations to list "
                    "(type=%s, value=%r, user_id=%s)",
                    type(rec_product_ids),
                    rec_product_ids,
                    user_id,
                )
                return []
        
        logger.info(
            "KNNWrapper.recommend: user_id=%s, input_cart_items=%s, rec_product_ids=%s",
            user_id,
            cart_items,
            rec_product_ids,
        )

        return rec_product_ids