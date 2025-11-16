# recsys.py
from collections import Counter
from db import get_connection


def get_popular_products(limit: int = 10):
    """
    Возвращает список product_id по популярности,
    считая события 'add_to_cart' и 'purchase' из таблицы event.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT item_id
            FROM event
            WHERE event_type IN ('add_to_cart', 'purchase')
              AND item_id IS NOT NULL
        """)
        rows = cur.fetchall()

    counts = Counter(row[0] for row in rows if row[0] is not None)
    popular_ids = [pid for pid, _ in counts.most_common(limit)]
    return popular_ids


def get_recommendations(user_id, cart_product_ids, limit: int = 5, exclude_cart=True):
    popular_ids = get_popular_products(limit=50)

    if exclude_cart:
        recs = [pid for pid in popular_ids if pid not in cart_product_ids]
    else:
        recs = popular_ids

    return recs[:limit]
