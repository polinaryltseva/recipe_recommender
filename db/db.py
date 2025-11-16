
# db.py
import sqlite3
from pathlib import Path
import csv
from typing import List, Dict, Optional, Tuple, Any
import json

from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "shop.db"

# DB_PATH = Path("db") / "shop.db"


# ================== БАЗА ==================

def get_connection() -> sqlite3.Connection:
    """Возвращает подключение к SQLite. Не забывай, что with сам закроет соединение."""
    conn = sqlite3.connect(DB_PATH)
    # Включаем поддержку внешних ключей
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Создаёт все таблицы, если их ещё нет."""
    with get_connection() as conn:
        cur = conn.cursor()

        # ---------- USER ----------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # ---------- CATEGORY ----------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS category (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        )
        """)

        # ---------- PRODUCT ----------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS product (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,

            name             TEXT NOT NULL,
            image_url        TEXT,

            price            REAL NOT NULL,
            unit             TEXT NOT NULL,

            weight_value     REAL,
            weight_unit      TEXT,

            rating           REAL,
            shelf_life_days  INTEGER,

            category_id      INTEGER,
            brand            TEXT,

            description      TEXT,
            composition      TEXT,

            metadata         TEXT,

            FOREIGN KEY (category_id) REFERENCES category(id)
        )
        """)

        # ---------- ORDERS ----------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            order_time  DATETIME DEFAULT CURRENT_TIMESTAMP,
            status      TEXT,
            total_price REAL,
            FOREIGN KEY (user_id) REFERENCES user(id)
        )
        """)

        # ---------- ORDER_ITEM ----------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS order_item (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id   INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity   INTEGER NOT NULL,
            price      REAL NOT NULL,
            FOREIGN KEY (order_id)  REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES product(id)
        )
        """)

        # ---------- RECIPE ----------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS recipe (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            title        TEXT NOT NULL,
            description  TEXT,
            instructions TEXT,
            prep_time    INTEGER,
            cook_time    INTEGER,
            image_url    TEXT
        )
        """)

        # ---------- RECIPE_PRODUCT ----------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS recipe_product (
            recipe_id   INTEGER NOT NULL,
            product_id  INTEGER NOT NULL,
            quantity    REAL NOT NULL,
            unit        TEXT NOT NULL,
            PRIMARY KEY (recipe_id, product_id),
            FOREIGN KEY (recipe_id)  REFERENCES recipe(id),
            FOREIGN KEY (product_id) REFERENCES product(id)
        )
        """)

        # ---------- EVENT (логи) ----------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS event (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            session_id     TEXT,
            event_time     DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type     TEXT NOT NULL,
            item_id        INTEGER,
            page_type      TEXT,
            source         TEXT,
            experiment_key TEXT,
            variant        TEXT,
            position       INTEGER,
            value          REAL,
            request_id     TEXT,
            metadata       TEXT,
            FOREIGN KEY (user_id) REFERENCES user(id),
            FOREIGN KEY (item_id) REFERENCES product(id)
        )
        """)

        # ---------- SESSION_INFO ----------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS session_info (
            session_id   TEXT PRIMARY KEY,
            user_id      INTEGER,
            started_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            ended_at     DATETIME,
            device_type  TEXT,
            user_agent   TEXT,
            ip_hash      TEXT,
            screen_w     INTEGER,
            screen_h     INTEGER,
            FOREIGN KEY (user_id) REFERENCES user(id)
        )
        """)

        # Индексы для ускорения запросов (не ломают данные)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_product_category ON product(category_id)
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id, order_time DESC)
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_user ON event(user_id, event_time DESC)
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_type_item ON event(event_type, item_id)
        """)

        conn.commit()


# ================== USER ==================

def create_user(username: str, password_hash: str) -> int:
    """Создаёт пользователя и возвращает его id."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()
        return cur.lastrowid


def get_user_by_username(username: str) -> Optional[tuple]:
    """Возвращает пользователя по username или None."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, password_hash, created_at FROM user WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
    return row


def get_user_by_id(user_id: int) -> Optional[tuple]:
    """Возвращает пользователя по id или None."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, password_hash, created_at FROM user WHERE id = ?",
            (user_id,),
        )
        row = cur.fetchone()
    return row


# ================== CATEGORY ==================

def create_category(name: str) -> int:
    """Создаёт категорию и возвращает её id."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO category (name) VALUES (?)", (name,))
        conn.commit()
        return cur.lastrowid


def get_all_categories() -> List[tuple]:
    """Возвращает список всех категорий."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM category ORDER BY name")
        rows = cur.fetchall()
    return rows


# ================== PRODUCT ==================

def create_product(
    name: str,
    price: float,
    unit: str,
    category_id: Optional[int] = None,
    image_url: Optional[str] = None,
    weight_value: Optional[float] = None,
    weight_unit: Optional[str] = None,
    rating: Optional[float] = None,
    shelf_life_days: Optional[int] = None,
    brand: Optional[str] = None,
    description: Optional[str] = None,
    composition: Optional[str] = None,
    metadata: Optional[str] = None,
) -> int:
    """
    Создаёт товар и возвращает его id.
    Необязательные поля можно не указывать.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO product (
                name, image_url, price, unit,
                weight_value, weight_unit,
                rating, shelf_life_days,
                category_id, brand,
                description, composition,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, image_url, price, unit,
            weight_value, weight_unit,
            rating, shelf_life_days,
            category_id, brand,
            description, composition,
            metadata,
        ))
        conn.commit()
        return cur.lastrowid


def get_all_products() -> List[tuple]:
    """Возвращает все товары (лучше не использовать в каталоге при 10k+ товаров)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, price, category_id, image_url, description
            FROM product
            ORDER BY id
        """)
        rows = cur.fetchall()
    return rows


def get_product_by_id(product_id: int) -> Optional[tuple]:
    """Возвращает один товар по id или None."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, image_url, price, unit,
                   weight_value, weight_unit,
                   rating, shelf_life_days,
                   category_id, brand,
                   description, composition,
                   metadata
            FROM product
            WHERE id = ?
        """, (product_id,))
        row = cur.fetchone()
    return row


def get_products_by_category(category_id: int) -> List[tuple]:
    """Возвращает товары по категории."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, price, category_id, image_url, description
            FROM product
            WHERE category_id = ?
            ORDER BY id
        """, (category_id,))
        rows = cur.fetchall()
    return rows


def get_products_count() -> int:
    """Возвращает количество товаров (для пагинации)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM product")
        row = cur.fetchone()
    return row[0] if row else 0


def get_products_page(offset: int, limit: int) -> List[tuple]:
    """Возвращает страницу товаров по offset/limit (для каталога)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, price, category_id, image_url, description
            FROM product
            ORDER BY id
            LIMIT ? OFFSET ?
        """, (limit, offset))
        rows = cur.fetchall()
    return rows


def get_products_by_ids(product_ids: List[int], preserve_order: bool = False) -> List[tuple]:
    """
    Возвращает товары по списку id. Если список пустой — [].

    Если preserve_order=True — порядок в результате совпадает с порядком
    в переданном product_ids.
    """
    if not product_ids:
        return []

    placeholders = ",".join("?" for _ in product_ids)
    query = f"""
        SELECT id, name, price, category_id, image_url, description
        FROM product
        WHERE id IN ({placeholders})
    """

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(query, product_ids)
        rows = cur.fetchall()

    if not preserve_order:
        # старое поведение — просто сортируем по id
        return sorted(rows, key=lambda r: r[0])

    # сохраняем порядок согласно product_ids
    by_id = {row[0]: row for row in rows}
    ordered = [by_id[pid] for pid in product_ids if pid in by_id]
    return ordered


# ================== ORDERS ==================

def create_order(user_id: int,
                 items: List[Tuple[int, int, float]],
                 status: str = "created") -> int:
    """
    Создаёт заказ и позиции заказа.

    items: список кортежей (product_id, quantity, price)
           price — цена за единицу на момент покупки.
    """
    with get_connection() as conn:
        cur = conn.cursor()

        # считаем total_price
        total_price = sum(q * p for _, q, p in items)

        # создаём сам заказ
        cur.execute("""
            INSERT INTO orders (user_id, status, total_price)
            VALUES (?, ?, ?)
        """, (user_id, status, total_price))
        order_id = cur.lastrowid

        # создаём позиции
        for product_id, quantity, price in items:
            cur.execute("""
                INSERT INTO order_item (order_id, product_id, quantity, price)
                VALUES (?, ?, ?, ?)
            """, (order_id, product_id, quantity, price))

        conn.commit()
        return order_id


def get_orders_by_user(user_id: int) -> List[tuple]:
    """Возвращает список заказов пользователя."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, user_id, order_time, status, total_price
            FROM orders
            WHERE user_id = ?
            ORDER BY order_time DESC
        """, (user_id,))
        rows = cur.fetchall()
    return rows


def get_order_items(order_id: int) -> List[tuple]:
    """Возвращает позиции конкретного заказа."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, order_id, product_id, quantity, price
            FROM order_item
            WHERE order_id = ?
        """, (order_id,))
        rows = cur.fetchall()
    return rows


# ================== RECIPE ==================

def create_recipe(
    title: str,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    prep_time: Optional[int] = None,
    cook_time: Optional[int] = None,
    image_url: Optional[str] = None,
) -> int:
    """Создаёт рецепт и возвращает его id."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO recipe (title, description, instructions, prep_time, cook_time, image_url)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, description, instructions, prep_time, cook_time, image_url))
        conn.commit()
        return cur.lastrowid


def add_product_to_recipe(recipe_id: int,
                          product_id: int,
                          quantity: float,
                          unit: str):
    """Добавляет продукт в рецепт (строка в recipe_product)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO recipe_product (recipe_id, product_id, quantity, unit)
            VALUES (?, ?, ?, ?)
        """, (recipe_id, product_id, quantity, unit))
        conn.commit()


def get_all_recipes() -> List[tuple]:
    """Возвращает список всех рецептов (без ингредиентов)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, title, description, prep_time, cook_time, image_url
            FROM recipe
            ORDER BY id
        """)
        rows = cur.fetchall()
    return rows


def get_recipe_with_products(recipe_id: int) -> Dict[str, object]:
    """
    Возвращает рецепт и его продукты в виде словаря:
    {
      "recipe": (id, title, description, ...),
      "products": [ (product_id, name, quantity, unit), ... ]
    }
    """
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT id, title, description, instructions, prep_time, cook_time, image_url
            FROM recipe
            WHERE id = ?
        """, (recipe_id,))
        recipe_row = cur.fetchone()

        if recipe_row is None:
            return {}

        cur.execute("""
            SELECT rp.product_id, p.name, rp.quantity, rp.unit
            FROM recipe_product rp
            JOIN product p ON p.id = rp.product_id
            WHERE rp.recipe_id = ?
        """, (recipe_id,))
        products = cur.fetchall()

    return {
        "recipe": recipe_row,
        "products": products,
    }


# ================== EVENT (логи) ==================

def log_event(user_id: int,
              event_type: str,
              item_id: Optional[int] = None,
              session_id: Optional[str] = None,
              page_type: Optional[str] = None,
              source: Optional[str] = None,
              experiment_key: Optional[str] = None,
              variant: Optional[str] = None,
              position: Optional[int] = None,
              value: Optional[float] = None,
              request_id: Optional[str] = None,
              metadata: Optional[str] = None):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO event (
                user_id, session_id, event_type, item_id,
                page_type, source,
                experiment_key, variant,
                position, value, request_id,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, session_id, event_type, item_id,
            page_type, source,
            experiment_key, variant,
            position, value, request_id,
            metadata,
        ))
        conn.commit()


def log_ui_event(user_id: int,
                 session_id: str,
                 event_type: str,
                 page_type: str,
                 source: str,
                 item_id: Optional[int] = None,
                 position: Optional[int] = None,
                 value: Optional[float] = None,
                 request_id: Optional[str] = None,
                 experiment_key: Optional[str] = None,
                 variant: Optional[str] = None,
                 **meta: Any):
    metadata_str = json.dumps(meta, ensure_ascii=False) if meta else None
    log_event(
        user_id=user_id,
        event_type=event_type,
        item_id=item_id,
        session_id=session_id,
        page_type=page_type,
        source=source,
        experiment_key=experiment_key,
        variant=variant,
        position=position,
        value=value,
        request_id=request_id,
        metadata=metadata_str,
    )



def get_events(limit: int = 1000) -> List[tuple]:
    """Возвращает последние N событий (по времени)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, user_id, session_id, event_time, event_type,
                   item_id, page_type, source, experiment_key, variant, metadata
            FROM event
            ORDER BY event_time DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
    return rows


# ================== SEED ИЗ ГОТОВЫХ BASE_TABLE_* ==================

def seed_categories_from_base_csv(csv_path: Optional[str] = None):
    """
    Заливает данные в таблицу category из заранее подготовленного CSV.

    Ожидается CSV с колонками:
    id,name

    ВАЖНО: функция просто делает INSERT.
    Если запустить её несколько раз подряд по одной и той же БД,
    будут дубликаты по id (упадёт по UNIQUE / PK).
    То есть это одноразовая инициализация.
    """
    from pathlib import Path

    if csv_path is None:
        csv_path = Path(__file__).parent / "datasets" / "base_table_categories.csv"
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"CSV {csv_path} не найден, пропускаю заливку категорий.")
        return

    with get_connection() as conn, open(csv_path, newline="", encoding="utf-8") as f:
        cur = conn.cursor()
        reader = csv.DictReader(f)

        count = 0
        for row in reader:
            # предполагаем, что id уже готовый Integer
            cat_id = int(row["id"]) if row.get("id") not in (None, "") else None
            name = (row.get("name") or "").strip()
            if not name:
                continue

            if cat_id is not None:
                cur.execute(
                    "INSERT INTO category (id, name) VALUES (?, ?)",
                    (cat_id, name),
                )
            else:
                cur.execute(
                    "INSERT INTO category (name) VALUES (?)",
                    (name,),
                )
            count += 1

        conn.commit()
    print(f"Залито категорий из {csv_path}: {count}")


def seed_products_from_base_csv(csv_path: Optional[str] = None):
    """
    Заливает данные в таблицу product из заранее подготовленного CSV,
    который уже соответствует схеме product.

    Ожидается CSV с колонками минимум:
      id,name,image_url,price,unit,
      weight_value,weight_unit,
      rating,shelf_life_days,
      category_id,brand,
      description,composition,
      metadata

    ВАЖНО: делаем простой INSERT.
    Если запустить несколько раз подряд по одной и той же БД,
    будут конфликты по id (PRIMARY KEY).
    """
    from pathlib import Path

    if csv_path is None:
        csv_path = Path(__file__).parent / "datasets" / "base_table_products.csv"
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"CSV {csv_path} не найден, пропускаю заливку продуктов.")
        return

    with get_connection() as conn, open(csv_path, newline="", encoding="utf-8") as f:
        cur = conn.cursor()
        reader = csv.DictReader(f)

        count = 0
        for row in reader:
            # id
            prod_id = int(row["id"]) if row.get("id") not in (None, "") else None

            name = (row.get("name") or "").strip()
            if not name:
                continue

            image_url = row.get("image_url")
            # price и числовые поля аккуратно приводим
            def _to_float(x):
                if x is None or x == "":
                    return None
                try:
                    return float(str(x).replace(",", "."))
                except Exception:
                    return None

            def _to_int(x):
                if x is None or x == "":
                    return None
                try:
                    return int(x)
                except Exception:
                    return None

            price = _to_float(row.get("price")) or 0.0
            unit = row.get("unit") or "шт"
            weight_value = _to_float(row.get("weight_value"))
            weight_unit = row.get("weight_unit")
            rating = _to_float(row.get("rating"))
            shelf_life_days = _to_int(row.get("shelf_life_days"))
            category_id = _to_int(row.get("category_id"))
            brand = row.get("brand")
            description = row.get("description")
            composition = row.get("composition")
            metadata_str = row.get("metadata")

            # вставка с сохранением id, если он есть
            if prod_id is not None:
                cur.execute("""
                    INSERT INTO product (
                        id,
                        name, image_url, price, unit,
                        weight_value, weight_unit,
                        rating, shelf_life_days,
                        category_id, brand,
                        description, composition,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prod_id,
                    name, image_url, price, unit,
                    weight_value, weight_unit,
                    rating, shelf_life_days,
                    category_id, brand,
                    description, composition,
                    metadata_str,
                ))
            else:
                cur.execute("""
                    INSERT INTO product (
                        name, image_url, price, unit,
                        weight_value, weight_unit,
                        rating, shelf_life_days,
                        category_id, brand,
                        description, composition,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    name, image_url, price, unit,
                    weight_value, weight_unit,
                    rating, shelf_life_days,
                    category_id, brand,
                    description, composition,
                    metadata_str,
                ))

            count += 1

        conn.commit()
    print(f"Залито товаров из {csv_path}: {count}")


if __name__ == "__main__":
    VERSION = "v2"

    BASE_DIR = Path(__file__).resolve().parent  # <--- ПАПКА db/

    init_db()

    seed_categories_from_base_csv(
        BASE_DIR / "datasets" / f"base_table_categories_{VERSION}.csv"
    )
    seed_products_from_base_csv(
        BASE_DIR / "datasets" / f"base_table_products_{VERSION}.csv"
    )

    print("DB инициализирована и заполнена из base_table_*.csv.")
