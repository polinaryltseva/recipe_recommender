from typing import List, Optional, Dict, Any

from recsys.base import BaseRecommender
from .main_ease import EASERecommendationSystem
import logging

logger = logging.getLogger(__name__)


class EasePopularRecommender(BaseRecommender):
    """
    Обёртка под твою исходную систему EASERecommendationSystem,
    чтобы она вписалась в общий интерфейс RecSys-case.
    """

    def __init__(self):
        self.system: Optional[EASERecommendationSystem] = None

    def load(self, model_path: str):
        """
        model_path сейчас можно игнорировать, потому что
        EASERecommendationSystem сам берёт пути из config.py,
        где мы уже прописали относительные пути к models/ease_popular/*.

        Но метод должен существовать по интерфейсу.
        """
        if self.system is None:
            logger.info(
                "EasePopular.load: initializing EASERecommendationSystem (model_path=%s)",
                model_path,
            )
            self.system = EASERecommendationSystem()
        else:
            logger.debug("EasePopular.load: system already initialized, skipping re-init")

        # Пытаемся аккуратно залогировать размерность модели, если есть такие атрибуты
        n_items = getattr(self.system, "n_items", None)
        if n_items is not None:
            logger.info("EasePopular: model ready, n_items=%s", n_items)
        else:
            logger.info("EasePopular: model ready (n_items unknown)")

    def recommend(
        self,
        user_id: int,
        cart_items: List[int],
        k: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """
        Здесь:
        - cart_items — это product_id из твоей БД (они же vv-id из маппингов);
        - мы просто прокидываем их в EASERecommendationSystem и получаем vv-id на выходе.
        """
        logger.debug(
            "EasePopular.recommend: user_id=%s, raw_cart_items=%s, k=%s, context_keys=%s",
            user_id,
            cart_items,
            k,
            list(context.keys()) if isinstance(context, dict) else None,
        )

        if not cart_items:
            logger.info(
                "EasePopular.recommend: empty cart_items, user_id=%s → возвращаем пустой список",
                user_id,
            )
            return []

        if self.system is None:
            logger.warning(
                "EasePopular.recommend: system is None, initializing on the fly (user_id=%s)",
                user_id,
            )
            self.system = EASERecommendationSystem()

        try:
            # Твой старый интерфейс:
            # system.get_recommendations(user_activity: List[int], top_k: int, exclude_seen: bool)
            rec_vv_ids = self.system.get_recommendations(
                user_activity=cart_items,
                top_k=k,
                exclude_seen=True,
            )
        except Exception:
            logger.exception(
                "EasePopular.recommend: error in EASERecommendationSystem.get_recommendations "
                "(user_id=%s, cart_items=%s, k=%s)",
                user_id,
                cart_items,
                k,
            )
            return []

        # На всякий случай приводим к list
        if not isinstance(rec_vv_ids, list):
            try:
                rec_vv_ids = list(rec_vv_ids)
            except TypeError:
                logger.error(
                    "EasePopular.recommend: cannot convert rec_vv_ids to list "
                    "(type=%s, value=%r, user_id=%s)",
                    type(rec_vv_ids),
                    rec_vv_ids,
                    user_id,
                )
                return []

        logger.info(
            "EasePopular.recommend: user_id=%s, input_cart_items=%s, rec_vv_ids=%s",
            user_id,
            cart_items,
            rec_vv_ids,
        )

        return rec_vv_ids
