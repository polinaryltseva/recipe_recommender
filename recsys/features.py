from typing import Dict, Any, List

def build_user_context(user_id: int, cart_items: List[int]) -> Dict[str, Any]:
    """Простейший конструктор фич — можно развивать под ranker/ML."""
    return {
        "user_id": user_id,
        "cart_size": len(cart_items),
        "cart_items": cart_items,
    }
