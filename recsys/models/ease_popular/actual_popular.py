# recsys.py
from collections import Counter

from db import get_connection


class DbTopPopular:
    """НУ какой-то код"""
    def __init__(self, k):
        self.k = k

    def get_popular_products(self):
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
        popular_ids = [pid for pid, _ in counts.most_common(self.k)]
        return popular_ids

    def get_recommendations(self, user_id, cart_product_ids, exclude_cart=True):
        popular_ids = self.get_popular_products(self.k)

        if exclude_cart:
            recs = [pid for pid in popular_ids if pid not in cart_product_ids]
        else:
            recs = popular_ids

        return recs[: self.k]
