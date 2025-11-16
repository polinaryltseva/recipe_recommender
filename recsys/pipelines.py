"""Здесь можно собирать пайплайны: кандидаты + реранкер."""


def generate_candidates(user_id: int, cart_items, models):
    # TODO: объединять кандидатов из разных моделей
    return []


def rerank_candidates(candidates, features_builder, ranker):
    # TODO: ML-ранжирование (LightGBM / CatBoost)
    return candidates
