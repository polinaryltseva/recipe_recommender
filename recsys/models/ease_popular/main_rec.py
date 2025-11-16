# recsys.py
from functools import lru_cache

from actual_popular import DbTopPopular
from .main_ease import EASERecommendationSystem

K_RECS = 20


@lru_cache(maxsize=1)
def get_ease_system() -> EASERecommendationSystem:
    """
    Ленивая инициализация модели EASE.
    Вызывается один раз за запуск приложения.
    """
    system = EASERecommendationSystem()
    return system


@lru_cache(maxsize=1)
def get_top_pop() -> DbTopPopular:
    """
    Ленивая инициализация модели EASE.
    Вызывается один раз за запуск приложения.
    """
    system = DbTopPopular()
    return system


def get_recommendations(
    user_id, cart_product_ids, limit: int = 5, exclude_cart: bool = True
):
    """
    Рекомендации на основе модели EASE.
    - cart_product_ids — список product_id, которые пользователь уже трогал (корзина/просмотры).
    - если корзина пустая — возвращаем просто топ популярных.
    - если нет — используем EASE + фильтруем товары из корзины.
    """
    # Если нет истории — fallback на топ-популярное
    top_pop_model = get_top_pop()
    if not cart_product_ids:
        # return get_popular_products(limit=K_RECS)
        return top_pop_model.get_popular_products()

    system = get_ease_system()
    # print(f"{list(cart_product_ids) = }\n")
    # Берём немного больше кандидатов, чтобы после фильтрации корзины всё ещё осталось `limit`
    raw_recs = system.get_recommendations(
        user_activity=list(cart_product_ids), top_k=limit + len(cart_product_ids)
    )
    print(f"{raw_recs = }\n\n\n")

    # На всякий случай приводим к int, если модель вернула строки
    try:
        raw_recs = [int(x) for x in raw_recs]
    except Exception:
        pass

    if exclude_cart:
        raw_recs = [pid for pid in raw_recs if pid not in cart_product_ids]

    return raw_recs[:limit]
