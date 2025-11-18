import hashlib
from typing import Dict

def _hash_to_bucket(user_id: int, buckets: int = 100) -> int:
    """Детерминированное распределение без хеша — просто %."""
    return abs(int(user_id)) % buckets

def choose_variant_for_user(user_id: int, experiment: Dict) -> str:
    """
    Детерминированный A/B роутер.
    Теперь без весов и лишних бакетов — просто user_id % num_variants.
    """
    variants = list(experiment["variants"].keys())
    num_variants = len(variants)

    if num_variants <= 1:
        return variants[0]

    bucket = abs(int(user_id)) % num_variants
    return variants[bucket]