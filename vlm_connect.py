from openai import AsyncOpenAI
import sys
import time
import asyncio


def initialize_model():
    return AsyncOpenAI(
        base_url="mask",
        api_key="sk-no-key-required",  # No API key needed for local vLLM server
    )


def get_current_cart(user_id):
    """Функция, возращающая текущую корзину пользователя по его ID (ЗАГЛУШКА)
    Должно обращаться к БД АСИНХРОННО и возвращать текущую корзину пользователя"""
    prompt = input().split("; ")
    print(prompt)
    if prompt == [""]:
        sys.exit()
    # ["Колбаса вареная", "Батон", "Молоко"]
    # ["Пиво \"Бельгийское\" светлое нефильтр., 500 мл",  "Пиво \"Clausthaler\" ж/б, 500 мл",  "Напиток пивной \"Гиннесс Драфт\", 440 мл", "Соломка из кеты вяленая"]
    return prompt


def get_previous_n_cart(user_id, n):
    """Функция, возращающая прошлые n покупок пользователя по его ID (ЗАГЛУШКА)
    Должно обращаться к БД АСИНХРОННО и возвращать прошлые n корзин пользователя"""
    return []


def get_simillar_n_cart(user_id, n, current_cart=None):
    """Функция, возвращающая n наиболее похожих прошлых покупок пользователя по его ID (ЗАГЛУШКА)
    Должно обращаться к БД АСИНХРОННО и возвращать прошлые n наиболее похожих на текущую корзин пользователя"""
    if current_cart is None:
        current_cart = get_current_cart(user_id)
    return []


def make_prompt_prefetch(current_cart, prev_cart, sim_cart, n):
    prompt = f"""Контекст:
Текущая корзина: {"; ".join(current_cart)}
Прошлые {n} покупок пользователя: {"; ".join(prev_cart)}
Прошлые {n} наиболее похожих покупок пользователя: {"; ".join(sim_cart)}
### ЗАДАЧА
На основе КОНТЕКСТА, сгенерируй 20 поисковых запросов, которые помогут пользователю добавить недостающие товары в корзину.
Важные правила:
1. Запросы должны быть краткими, отражать общие категории или идеи, а не конкретными товарами с брендом или весом.
2. Не повторяй товары, которые уже есть в корзине.
3. Поисковые запросы должны быть реалистичными для продуктового магазина с 10-15 тысячами наименований  
4. Вывод должен быть в формате [<запрос 1>, <запрос 2>, ...]

### ПРИМЕР (для демонстрации логики, а не для копирования)
- Пример входной корзины: [Мука пшеничная; Яйца куриные; Сахар-песок]
- Пример правильного вывода: ["разрыхлитель", "ванильный экстракт", "сливочное масло", "шоколад", "кондитерские украшения"]
    """
    return prompt


async def get_recs(client, user_id, context_window):
    curr_cart = get_current_cart(user_id)
    final_prompt = make_prompt_prefetch(
        curr_cart,
        get_previous_n_cart(user_id, context_window),
        get_simillar_n_cart(user_id, context_window, curr_cart),
        context_window
    )
    print("making recs...")
    start_time = time.time()
    response = await client.chat.completions.create(
        model="QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
        messages=[
            {"role": "system", "content":
                """Ты — ИИ-ассистент для системы продуктовых рекомендаций.
                Твоя задача — анализировать корзину и генерировать идеи для поиска недостающих товаров в виде списка поисковых запросов."""},
            {"role": "user", "content": final_prompt},
        ],
        max_tokens=200,
    )
    print(f"LLM time: {time.time()-start_time}")
    return response.choices[0].message.content


async def main():  # Главная асинхронная функция
    client = initialize_model()
    while True:
        result = await get_recs(client, 1, 5)  # Ждем выполнения асинхронной функции
        # тут надо сделать матчинг с реальными товарами и выдавать реки
        print(result)


if __name__ == "__main__":
    asyncio.run(main())  # Запускаем асинхронную main функцию
