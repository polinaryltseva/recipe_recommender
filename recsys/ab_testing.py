import hashlib
from typing import Dict

def _hash_to_bucket(user_id: int, buckets: int = 100) -> int:
    """Хэширует user_id в один из N бакетов."""
    h = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
    return int(h, 16) % buckets

def choose_variant_for_user(user_id: int, experiment: Dict) -> str:
    """
    Динамический A/B роутер.
    Количество бакетов = 100 * количество вариантов для большей точности.
    """
    variants = experiment["variants"]
    num_variants = len(variants)
    
    # Если вариант всего один, не тратим время на расчеты
    if num_variants <= 1:
        return list(variants.keys())[0]

    # Динамически определяем количество бакетов
    num_buckets = 100 * num_variants
    
    bucket = _hash_to_bucket(user_id, buckets=num_buckets)

    threshold = 0
    for name, weight in variants.items():
        # Распределяем доли в рамках нового, увеличенного пространства бакетов
        share = int(weight * num_buckets)
        threshold += share
        
        if bucket < threshold:
            return name
            
    # Запасной вариант на случай ошибок округления (хотя теперь это маловероятно)
    return list(variants.keys())[0]