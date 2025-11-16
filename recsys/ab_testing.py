import hashlib
from typing import Dict

def _hash_to_bucket(user_id: int, buckets: int = 100) -> int:
    h = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
    return int(h, 16) % buckets

def choose_variant_for_user(user_id: int, experiment: Dict) -> str:
    """Простейший A/B роутер по хэшу user_id и весам вариантов."""
    variants = experiment["variants"]  # dict: name -> weight (сумма ~ 1.0)
    bucket = _hash_to_bucket(user_id)
    threshold = 0
    current = 0
    for name, weight in variants.items():
        threshold += int(weight * 100)
        if bucket < threshold:
            return name
        current = threshold
    # запасной вариант
    return list(variants.keys())[0]
