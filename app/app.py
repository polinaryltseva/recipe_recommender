# app.py
# Основное приложение Streamlit для "Мини-Лавки":
# - авторизация пользователей
# - каталог товаров с пагинацией + поиск (Whoosh, с опечатками)
# - корзина и оформление заказов
# - логирование действий и рекомендации (справа колонкой)

import streamlit as st
import uuid
import json
import hashlib

import logging
from pathlib import Path

# === НАСТРОЙКА ЛОГИРОВАНИЯ ===

ROOT_DIR = Path(__file__).resolve().parents[1]  # корень проекта
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,  # можно поставить DEBUG, если нужно больше детализации
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(),  # дублирует в консоль, удобно при отладке
    ],
)

logger = logging.getLogger(__name__)
logger.info("=== App started ===")



import __main__
# from easy_pipeline.top_popular import TopPopular

# чтобы pickle, который ищет __main__.TopPopular, его нашёл
# setattr(__main__, "TopPopular", TopPopular)



from db import (
    init_db,
    create_user,
    get_user_by_username,
    get_all_products,
    get_product_by_id,
    get_products_count,
    get_products_page,
    get_products_by_ids,
    create_order,
    log_event,      # можно оставить, если ещё где-то нужен
    log_ui_event,   # новый helper
)

from recsys.registry import get_recommender_for_user
from recsys.features import build_user_context


# ================== ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ ==================

# Создаём таблицы в БД (если их ещё нет)
init_db()

# Настройка страницы Streamlit
st.set_page_config(page_title="Мини-Лавка", layout="wide")

# Глобальный CSS для карточек и сетки
st.markdown(
    """
<style>
/* карточка товара внутри колонки */
div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
    background-color: #f7f7f9;
    border-radius: 16px;
    padding: 10px 10px 14px 10px;
    margin-bottom: 16px;
    height: 100%;
}

/* область картинки/описания фиксированной высоты */
.product-media {
    height: 230px;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 8px;
    background-color: #ffffff;
}
.product-media img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* блок описания вместо картинки */
.product-media-desc {
    height: 230px;
    border-radius: 12px;
    overflow-y: auto;
    padding: 8px;
    background-color: #ffffff;
    font-size: 0.9rem;
}

/* заголовки и текст в описании */
.product-desc-title {
    font-weight: 600;
    margin-bottom: 6px;
}
.product-desc-label {
    font-weight: 500;
    margin-top: 6px;
}
.product-desc-text {
    margin-top: 2px;
    white-space: pre-wrap;
}

/* название и цена под медиа-блоком */
.product-name {
    font-weight: 600;
    font-size: 0.95rem;
    height: 40px;                  /* ровно 2 строки */
    margin-bottom: 4px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;         /* максимум 2 строки */
    -webkit-box-orient: vertical;
}

.product-price {
    font-weight: 500;
    margin-bottom: 6px;
}

/* кнопка описания чуть компактнее */
button[kind="secondary"] {
    padding-top: 2px !important;
    padding-bottom: 2px !important;
}

/* чуть уменьшим отступы между строками в каталоге */
.block-container {
    padding-top: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)


st.title("Лавка рекомендаций")


# ================== РАБОТА С СЕССИЕЙ ==================

def ensure_session():
    """
    Инициализирует состояние Streamlit:
    - session_id: уникальный идентификатор сессии
    - user_id / username: текущий пользователь
    - cart: корзина вида {product_id: quantity}
    - page / page_size: пагинация каталога
    - search_page / last_search_query: пагинация и состояние поиска
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    if "username" not in st.session_state:
        st.session_state.username = ""

    if "cart" not in st.session_state:
        st.session_state.cart = {}  # product_id -> quantity

    if "page" not in st.session_state:
        st.session_state.page = 1

    if "page_size" not in st.session_state:
        st.session_state.page_size = 32  # 4x8

    if "search_page" not in st.session_state:
        st.session_state.search_page = 1

    if "last_search_query" not in st.session_state:
        st.session_state.last_search_query = ""

    if "show_add_toast" not in st.session_state:
        st.session_state.show_add_toast = False

    if "reset_search" not in st.session_state:
        st.session_state.reset_search = False


ensure_session()

session_id = st.session_state.session_id
user_id = st.session_state.user_id  # будет None до авторизации


def cart_snapshot():
    """Возвращает JSON-строку со снимком текущей корзины (для логов)."""
    return json.dumps(st.session_state.cart, ensure_ascii=False)


# ================== АВТОРИЗАЦИЯ / РЕГИСТРАЦИЯ ==================

def hash_password(password: str) -> str:
    """Простой SHA-256 хеш пароля (для учебного проекта)."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def auth_block():
    """
    Блок авторизации/регистрации в левом сайдбаре.
    - Если пользователь залогинен - приветствие + кнопка "Выйти".
    - Если нет - форма логина и регистрации.
    """
    st.sidebar.header("Профиль")

    # Уже залогинен
    if st.session_state.user_id is not None:
        st.sidebar.write(f"Привет, **{st.session_state.username}**!")

        if st.sidebar.button("Выйти"):
            st.session_state.user_id = None
            st.session_state.username = ""
            st.session_state.cart = {}
            st.session_state.page = 1
            st.session_state.search_page = 1
            st.rerun()
        return

    # Не залогинен - форма логина/регистрации
    username = st.sidebar.text_input("Логин")
    password = st.sidebar.text_input("Пароль", type="password")

    col_login, col_register = st.sidebar.columns(2)
    with col_login:
        do_login = st.button("Sign in")
    with col_register:
        do_register = st.button("Sign up")

    # Регистрация
    if do_register:
        if not username or not password:
            st.sidebar.error("Логин и пароль не должны быть пустыми.")
            logger.warning(
                "Попытка регистрации с пустыми полями: username=%r", username
            )
        else:
            existing = get_user_by_username(username)
            if existing is not None:
                st.sidebar.error("Пользователь с таким логином уже существует.")
                logger.warning(
                    "Попытка регистрации с уже существующим логином: username=%s",
                    username,
                )
            else:
                pwd_hash = hash_password(password)
                new_id = create_user(username, pwd_hash)
                logger.info(
                    "Регистрация нового пользователя: user_id=%s, username=%s",
                    new_id,
                    username,
                )

                st.session_state.user_id = new_id
                st.session_state.username = username
                st.session_state.cart = {}
                st.session_state.page = 1
                st.session_state.search_page = 1
                st.sidebar.success("Регистрация успешна, вы вошли в систему.")
                st.rerun()

    # Вход
    if do_login:
        if not username or not password:
            st.sidebar.error("Введите логин и пароль.")
            logger.warning(
                "Попытка входа с пустыми полями: username=%r", username
            )
        else:
            user_row = get_user_by_username(username)
            if user_row is None:
                st.sidebar.error("Неверный логин или пароль.")
                logger.warning(
                    "Попытка входа с несуществующим логином: username=%s",
                    username,
                )
            else:
                uid, uname, stored_hash, created_at = user_row
                if stored_hash != hash_password(password):
                    st.sidebar.error("Неверный логин или пароль.")
                    logger.warning(
                        "Неверный пароль при входе: username=%s, user_id=%s",
                        uname,
                        uid,
                    )
                else:
                    st.session_state.user_id = uid
                    st.session_state.username = uname
                    st.session_state.page = 1
                    st.session_state.search_page = 1

                    logger.info(
                        "Успешный вход: user_id=%s, username=%s",
                        uid,
                        uname,
                    )


auth_block()
user_id = st.session_state.user_id  # мог измениться


# если до этого добавляли товар (например, из рекомендаций),
# показываем тост и сразу сбрасываем флаг
if st.session_state.show_add_toast:
    st.toast("Добавлено!", icon="🛒")
    st.session_state.show_add_toast = False

# ================== WHOOSH: ИНДЕКС ДЛЯ ПОИСКА ==================
from whoosh.fields import Schema, TEXT, ID
from whoosh.filedb.filestore import RamStorage
from whoosh import scoring
from whoosh.query import And, Or, FuzzyTerm, Term


@st.cache_resource(show_spinner=False)
def build_search_index():
    """
    Строит in-memory индекс Whoosh по всем товарам (name + description).
    Кэшируется на уровне приложения - строится один раз на запуск.
    """
    schema = Schema(
        pid=ID(stored=True, unique=True),
        name=TEXT(stored=True),
        description=TEXT(stored=True),
    )

    storage = RamStorage()
    idx = storage.create_index(schema)

    writer = idx.writer()
    all_products = get_all_products()  # (id, name, price, category_id, image_url, description)
    for pid, name, price, category_id, image_url, description in all_products:
        writer.add_document(
            pid=str(pid),
            name=(name or ""),
            description=(description or ""),
        )
    writer.commit()
    return idx


def search_products_fuzzy(query: str, limit: int = 256):
    """
    Оптимизированный поиск:
    - если запрос < 3 символов - очень лёгкий (почти без fuzzy);
    - если >= 3 символов - fuzzy только по name, по description - точный Term;
    - индекс кэшируется на весь рантайм.

    Возвращает список продуктов в формате get_all_products().
    """
    query = (query or "").strip()
    if not query:
        return []

    idx = build_search_index()

    terms = [w for w in query.lower().split() if w.strip()]
    if not terms:
        return []

    subqueries = []
    # для каждого слова строим Or по полям
    for t in terms:
        if len(t) < 3:
            # короткие куски - без heavy fuzzy
            subqueries.append(
                Or([
                    Term("name", t),
                    Term("description", t),
                ])
            )
        else:
            # более длинные - fuzzy по name (prefixlength=2 для скорости),
            # а по description - точный Term (чаще всего хватает)
            subqueries.append(
                Or([
                    FuzzyTerm("name", t, maxdist=1, prefixlength=2),
                    Term("description", t),
                ])
            )

    whoosh_query = And(subqueries)

    result_ids = []
    with idx.searcher(weighting=scoring.BM25F()) as searcher:
        results = searcher.search(whoosh_query, limit=limit)
        for hit in results:
            result_ids.append(int(hit["pid"]))

    if not result_ids:
        return []

    # Сохраняем порядок по релевантности
    products = get_products_by_ids(result_ids, preserve_order=True)
    return products

# ================== ПАГИНАЦИЯ ==================

def pagination_controls(page: int, total_pages: int, total_products: int, position: str):
    """
    Пагинация для обычного каталога (использует st.session_state.page).
    """
    col_prev, col_page_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Назад", disabled=(page <= 1), key=f"{position}_page_prev_{page}"):
            st.session_state.page = max(1, page - 1)
            st.rerun()
    with col_page_info:
        st.write(
            f"Страница **{page}** из **{total_pages}** "
            f"(товаров всего: {total_products}, на странице: 32)"
        )
    with col_next:
        if st.button("Вперёд →", disabled=(page >= total_pages), key=f"{position}_page_next_{page}"):
            st.session_state.page = min(total_pages, page + 1)
            st.rerun()


def search_pagination_controls(page: int, total_pages: int, total_products: int, position: str):
    """
    Пагинация для результатов поиска (использует st.session_state.search_page).
    position: 'top' или 'bottom' - чтобы ключи кнопок были уникальными.
    """
    col_prev, col_page_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Назад", disabled=(page <= 1), key=f"search_{position}_page_prev_{page}"):
            st.session_state.search_page = max(1, page - 1)
            st.rerun()
    with col_page_info:
        st.write(
            f"Страница **{page}** из **{total_pages}** "
            f"(найдено товаров: {total_products}, на странице: 32)"
        )
    with col_next:
        if st.button("Вперёд →", disabled=(page >= total_pages), key=f"search_{position}_page_next_{page}"):
            st.session_state.search_page = min(total_pages, page + 1)
            st.rerun()



# ================== КАРТОЧКА ТОВАРА ==================

def render_product_card(
    pid,
    name,
    price,
    category_id,
    image_url,
    description,
    user_id,
    session_id,
    page_type: str = "catalog",
    source: str = "catalog",
    position: int | None = None,      # номер товара в выдаче/блоке
    request_id: str | None = None,    # id запроса (поиск и т.п.)
    query: str | None = None,         # текст запроса (для поиска)
):

    """
    Отрисовывает карточку товара:
    - фото или описание (переключается кнопкой)
    - обрезанное название (2 строки)
    - цена
    - кнопка "Добавить в корзину" с логом

    БЕЗ дополнительных запросов к БД - используем только то,
    что уже передано (name, description, image_url, price).
    """
    full_name = name or ""
    full_description = description or ""
    composition = ""  # состав в этом варианте не тянем (для перформанса)

    # Состояние: показывать фото или описание
    show_desc_key = f"show_desc_{pid}"
    if show_desc_key not in st.session_state:
        st.session_state[show_desc_key] = False
    show_desc = st.session_state[show_desc_key]

    # Медиа-блок
    if not show_desc:
        if image_url:
            st.markdown(
                f"""
                <div class="product-media">
                    <img src="{image_url}" alt="{full_name}">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="product-media">
                    <div style="display:flex;align-items:center;justify-content:center;height:100%;color:#888;">
                        Нет изображения
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        desc_text = full_description or "Описание отсутствует."
        comp_text = composition or "Состав не указан."
        st.markdown(
            f"""
            <div class="product-media product-media-desc">
                <div class="product-desc-title">{full_name}</div>
                <div class="product-desc-label">Описание:</div>
                <div class="product-desc-text">{desc_text}</div>
                <div class="product-desc-label">Состав:</div>
                <div class="product-desc-text">{comp_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Переключатель "Описание/Фото"
    toggle_label = "Описание" if not show_desc else "Фото"
    if st.button(toggle_label, key=f"toggle_desc_{pid}"):
        st.session_state[show_desc_key] = not show_desc
        st.rerun()

    # Название (2 строки)
    short_name = full_name
    max_len = 40
    if len(short_name) > max_len:
        short_name = short_name[: max_len - 1] + "…"

    st.markdown(
        f'<div class="product-name">{short_name}</div>',
        unsafe_allow_html=True,
    )

    # Цена
    st.markdown(
        f'<div class="product-price">{price:.2f} ₽</div>',
        unsafe_allow_html=True,
    )

    # Кнопка "Добавить в корзину"
    if user_id is None:
        st.caption("Войдите, чтобы добавить в корзину.")
    else:
        if st.button("Добавить в корзину", key=f"add_{pid}_{page_type}"):
            st.session_state.cart[pid] = st.session_state.cart.get(pid, 0) + 1

            # логируем клик по товару с его позицией в выдаче
            log_ui_event(
                user_id=user_id,
                session_id=session_id,
                event_type="add_to_cart",
                page_type=page_type,
                source=source,
                item_id=pid,
                position=position,
                request_id=request_id,
                query=query,
                cart=st.session_state.cart,
            )

            st.toast("Добавлено!", icon="🛒")


# ================== ОСНОВНОЙ ЛЕЙАУТ: ЛЕВО (ТАБЫ) + ПРАВО (РЕКОМЕНДАЦИИ) ==================

main_col, recs_col = st.columns([4, 1])

with main_col:
    tab_catalog, tab_cart = st.tabs(["Каталог", "Корзина"])

    # ---------- ТАБ "КАТАЛОГ" ----------
    with tab_catalog:
        st.subheader("Каталог")

        # --- обработка сброса ПО ДО text_input ---
        if st.session_state.reset_search:
            # здесь мы можем спокойно трогать widget-key до его создания
            st.session_state.catalog_search_query = ""
            st.session_state.last_search_query = ""
            st.session_state.search_page = 1
            st.session_state.reset_search = False

        # --- поиск + кнопка сброса ---
        search_col, clear_col = st.columns([4, 1])
        with search_col:
            search_query = st.text_input(
                "Поиск по товарам",
                key="catalog_search_query",
                placeholder="Например: молоко, йогурт, банан",
            )
        with clear_col:
            clear_search = st.button("✕ Сбросить", key="clear_search", help="Очистить поиск и вернуть каталог")

        # нажали «сброс» -> ставим флаг и делаем rerun
        if clear_search:
            st.session_state.reset_search = True
            st.rerun()

        # если запрос изменился вручную - сбрасываем страницу результатов поиска
        if search_query != st.session_state.last_search_query:
            st.session_state.last_search_query = search_query
            st.session_state.search_page = 1

        # ---- если поиска нет - обычный каталог с пагинацией ----
        if not search_query:
            st.subheader("Все товары")

            total_products = get_products_count()
            if total_products == 0:
                st.info("В базе пока нет товаров. Запусти db.py, чтобы создать таблицы и залить CSV.")
            else:
                page_size = st.session_state.page_size
                total_pages = (total_products + page_size - 1) // page_size

                if st.session_state.page < 1:
                    st.session_state.page = 1
                if st.session_state.page > total_pages:
                    st.session_state.page = total_pages

                page = st.session_state.page

                pagination_controls(page, total_pages, total_products, position="top")

                offset = (page - 1) * page_size
                products = get_products_page(offset, page_size)

                num_cols = 4
                cols = st.columns(num_cols)

                for idx, (pid, name, price, category_id, image_url, description) in enumerate(products):
                    col = cols[idx % num_cols]
                    # глобальная позиция товара в текущем каталоге
                    global_pos = (page - 1) * page_size + idx + 1

                    with col:
                        with st.container():
                            render_product_card(
                                pid, name, price, category_id, image_url, description,
                                user_id, session_id,
                                page_type="catalog",
                                source="catalog",
                                position=global_pos,
                            )

                pagination_controls(page, total_pages, total_products, position="bottom")

        # ---- если есть текст в поиске ----
        else:
            st.subheader("Результаты поиска")

            q_strip = (search_query or "").strip()
            if len(q_strip) < 3:
                # ВАЖНО: эта надпись теперь ТОЛЬКО внутри таба "Каталог"
                st.info("Введите хотя бы 3 символа для поиска (для скорости работы).")
            else:
                # один request_id на текущую поисковую выдачу
                search_request_id = uuid.uuid4().hex

                results = search_products_fuzzy(search_query)
                total_found = len(results)

                if total_found == 0:
                    st.warning("По вашему запросу ничего не найдено.")
                else:
                    page_size = st.session_state.page_size
                    total_pages = (total_found + page_size - 1) // page_size

                    if st.session_state.search_page < 1:
                        st.session_state.search_page = 1
                    if st.session_state.search_page > total_pages:
                        st.session_state.search_page = total_pages

                    page = st.session_state.search_page

                    search_pagination_controls(page, total_pages, total_found, position="top")

                    start_idx = (page - 1) * page_size
                    end_idx = start_idx + page_size
                    page_items = results[start_idx:end_idx]

                    num_cols = 4
                    cols = st.columns(num_cols)

                    for idx, (pid, name, price, category_id, image_url, description) in enumerate(page_items):
                        col = cols[idx % num_cols]
                        # позиция товара в общей поисковой выдаче
                        global_pos = (page - 1) * page_size + idx + 1

                        with col:
                            with st.container():
                                render_product_card(
                                    pid, name, price, category_id, image_url, description,
                                    user_id, session_id,
                                    page_type="search",
                                    source="search",
                                    position=global_pos,
                                    request_id=search_request_id,
                                    query=search_query,
                                )

                    search_pagination_controls(page, total_pages, total_found, position="bottom")

    # ---------- ТАБ "КОРЗИНА" ----------
    with tab_cart:
        st.subheader("Ваша корзина")

        if user_id is None:
            st.info("Чтобы пользоваться корзиной и оформлять заказы, войдите в свой профиль.")
        else:
            try:
                cart = st.session_state.get("cart", {})
                if not isinstance(cart, dict):
                    cart = {}
                    st.session_state.cart = {}

                if not cart:
                    st.info("Корзина пуста 😔")
                else:
                    cart_product_ids = list(cart.keys())
                    products_list = get_products_by_ids(cart_product_ids, preserve_order=True)
                    products_by_id = {p[0]: p for p in products_list}

                    total = 0.0

                    for pid, qty in list(cart.items()):
                        prod = products_by_id.get(pid)
                        if not prod:
                            continue

                        _, name, price, category_id, image_url, description = prod

                        col_name, col_qty, col_btn = st.columns([3, 1, 1])
                        with col_name:
                            st.write(f"**{name}**")
                            st.caption(f"{price:.2f} ₽ за единицу")
                        with col_qty:
                            st.write(f"x {qty}")
                        with col_btn:
                            if st.button("−", key=f"remove_{pid}"):
                                if cart[pid] > 1:
                                    cart[pid] -= 1
                                else:
                                    del cart[pid]

                                log_event(
                                    user_id=user_id,
                                    event_type="remove_from_cart",
                                    item_id=pid,
                                    session_id=session_id,
                                    page_type="cart",
                                    source="cart",
                                    metadata=cart_snapshot(),
                                )
                                st.rerun()

                        total += price * qty

                    st.write(f"**Итого:** {total:.2f} ₽")

                    if st.button("Оформить заказ"):
                        order_items = []
                        for pid, qty in cart.items():
                            prod = products_by_id.get(pid)
                            if not prod:
                                continue
                            _, name, price, *_ = prod
                            order_items.append((pid, qty, price))

                        if order_items:
                            order_id = create_order(
                                user_id=user_id,
                                items=order_items,
                                status="created",
                            )
                            meta = {
                                "cart": st.session_state.cart,
                                "order_id": order_id,
                                "total_price": total,
                            }
                            log_event(
                                user_id=user_id,
                                event_type="purchase",
                                item_id=None,
                                session_id=session_id,
                                page_type="cart",
                                source="cart",
                                metadata=json.dumps(meta, ensure_ascii=False),
                            )

                            st.session_state.cart = {}
                            st.success(f"Заказ №{order_id} оформлен! 🎉 Лог записан.")
                        else:
                            st.warning("Не удалось сформировать заказ (корзина пустая).")
            except Exception:
                st.error("Произошла ошибка при загрузке корзины. Корзина была сброшена.")
                st.session_state.cart = {}


# ================== ПРАВАЯ ПАНЕЛЬ: РЕКОМЕНДАЦИИ ОДНИМ СТОЛБЦОМ ==================

with recs_col:
    with st.expander("Рекомендации для вас", expanded=True):
        if user_id is None:
            st.info("Рекомендации доступны после входа в аккаунт.")
        else:
            cart_product_ids = list(st.session_state.cart.keys())

            # LOG: логируем вход в модель (кто и с каким контекстом)
            logger.info(
                "Recsys call: user_id=%s, cart_product_ids=%s",
                user_id,
                cart_product_ids,
            )

            # 1. Строим контекст (если нужно что-то ещё — добавим позже)
            ctx = build_user_context(user_id=user_id, cart_items=cart_product_ids)

            # 2. Берём нужную модель из реестра (с учётом A/B, если включишь)
            try:
                recsys_model = get_recommender_for_user(user_id)
            except Exception:
                logger.exception(
                    "Ошибка при получении модели рекомендаций для user_id=%s",
                    user_id,
                )
                st.write("Не удалось загрузить модель рекомендаций.")
                recsys_model = None

            rec_ids = []
            if recsys_model is not None:
                try:
                    rec_ids = recsys_model.recommend(
                        user_id=user_id,
                        cart_items=cart_product_ids,
                        k=8,
                        context=ctx,
                    )

                    # LOG: что вернула модель
                    logger.info(
                        "Recsys response: user_id=%s, rec_ids=%s",
                        user_id,
                        rec_ids,
                    )
                except Exception:
                    logger.exception(
                        "Ошибка при вызове recsys_model.recommend для user_id=%s",
                        user_id,
                    )
                    st.write("Не удалось получить рекомендации, попробуйте позже.")
                    rec_ids = []

            # st.write("DEBUG user_id:", user_id)
            # st.write("DEBUG cart_product_ids:", cart_product_ids)
            # st.write("DEBUG rec_ids:", rec_ids)

            if not rec_ids:
                st.write("Пока нет рекомендаций - нужно, чтобы накопились события.")
            else:
                rec_products = get_products_by_ids(rec_ids)

                # LOG: проверим, все ли rec_ids есть в базе
                db_product_ids = [p[0] for p in rec_products]
                missing = set(rec_ids) - set(db_product_ids)
                if missing:
                    logger.warning(
                        "Некоторые рекомендованные товары не найдены в БД: user_id=%s, missing_ids=%s",
                        user_id,
                        list(missing),
                    )

                # один request_id на весь показ набора рекомендаций
                rec_request_id = uuid.uuid4().hex

                for pos, (pid, name, price, category_id, image_url, description) in enumerate(rec_products, start=1):
                    with st.container():
                        st.markdown('<div class="product-card">', unsafe_allow_html=True)

                        # Картинка товара
                        if image_url:
                            st.markdown(
                                f"""
                                <div class="product-media">
                                    <img src="{image_url}" alt="{name}">
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                """
                                <div class="product-media">
                                    <div style="display:flex;align-items:center;justify-content:center;height:100%;color:#888;">
                                        Нет изображения
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Название (обрезаем до ~2 строк)
                        short_name = name[:38] + "…" if len(name or "") > 40 else (name or "")
                        st.markdown(
                            f'<div class="product-name">{short_name}</div>',
                            unsafe_allow_html=True,
                        )

                        # Цена
                        st.markdown(
                            f'<div class="product-price">{price:.2f} ₽</div>',
                            unsafe_allow_html=True,
                        )

                        # Логируем показ рекомендации с её позицией
                        log_ui_event(
                            user_id=user_id,
                            session_id=session_id,
                            event_type="rec_impression",
                            page_type="recs_sidebar",
                            source="recs",
                            item_id=pid,
                            position=pos,
                            request_id=rec_request_id,
                            cart=st.session_state.cart,
                        )

                        # LOG: дополнительно можно писать это и в тех.лог (если хочешь там видеть показы)
                        logger.debug(
                            "Rec impression: user_id=%s, item_id=%s, position=%s, request_id=%s",
                            user_id,
                            pid,
                            pos,
                            rec_request_id,
                        )

                        # Кнопка "В корзину"
                        if st.button("В корзину", key=f"minirec_add_{pid}"):
                            st.session_state.cart[pid] = st.session_state.cart.get(pid, 0) + 1

                            # Логируем клик по рекомендации
                            log_ui_event(
                                user_id=user_id,
                                session_id=session_id,
                                event_type="rec_click",
                                page_type="recs_sidebar",
                                source="recs",
                                item_id=pid,
                                position=pos,
                                request_id=rec_request_id,
                                cart=st.session_state.cart,
                            )

                            # LOG: тех.лог про клик
                            logger.info(
                                "Rec click: user_id=%s, item_id=%s, position=%s, request_id=%s",
                                user_id,
                                pid,
                                pos,
                                rec_request_id,
                            )

                            st.session_state.show_add_toast = True
                            st.rerun()

                        st.markdown("</div>", unsafe_allow_html=True)
