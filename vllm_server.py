from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
import sys
import time
import asyncio
import uuid # Для генерации уникальных ID запросов

def initialize_model(model_name: str):
    print(f"Загрузка модели {model_name}...")
    engine_args = AsyncEngineArgs(model=model_name, gpu_memory_utilization=0.9, max_model_len=2048)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    print("Модель успешно загружена.")
    return engine


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
На основе КОНТЕКСТА, сгенерируй 20 поисковых запросов, которые помогут пользователю добавить недостающие для приготовления блюд товары в корзину.
Важные правила:
1. Запросы должны быть краткими, отражать общие категории или идеи, а не конкретными товарами с брендом или весом.
2. Не повторяй товары, которые уже есть в корзине и не допускай повторений в генерируемом списке.
3. Поисковые запросы должны быть реалистичными для продуктового магазина с 10-15 тысячами наименований  
4. Вывод должен быть в формате [<запрос 1>, <запрос 2>, ...]

### ПРИМЕР (для демонстрации логики, а не для копирования)
- Пример входной корзины: [Мука пшеничная; Яйца куриные; Сахар-песок]
- Пример правильного вывода: ["разрыхлитель", "ванильный экстракт", "сливочное масло", "шоколад", "кондитерские украшения"]
    """
    return prompt

async def get_recs(engine: AsyncLLMEngine, user_id, context_window):
    curr_cart = get_current_cart(user_id)
    user_prompt = make_prompt_prefetch(
        curr_cart,
        get_previous_n_cart(user_id, context_window),
        get_simillar_n_cart(user_id, context_window, curr_cart),
        context_window
    )

    system_prompt = "/no_think Ты — ИИ-ассистент для системы продуктовых рекомендаций. Твоя задача — анализировать корзину и генерировать идеи для поиска недостающих для рецептов товаров в виде списка поисковых запросов."
    final_prompt = f"{system_prompt}\n\n{user_prompt}"

    sampling_params = SamplingParams(max_tokens=200, temperature=0.0, top_p=1.0)

    print("\nГенерация рекомендаций...")
    start_time = time.time()

    # Создаем уникальный ID для каждого запроса
    request_id = str(uuid.uuid4())
    
    # Вызываем метод generate. Он возвращает асинхронный генератор.
    results_generator = engine.generate(final_prompt, sampling_params, request_id)
    
    # Итерируемся по генератору, чтобы получить конечный результат
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    print(f"Время работы LLM: {time.time()-start_time:.2f} сек.")
    if final_output is None or not getattr(final_output, "outputs", None):
        # ничего не сгенерировано — безопасный fallback
        return ""
    # проверяем наличие текста
    out0 = final_output.outputs[0]
    text = getattr(out0, "text", "")
    return text


async def main():
    # Инициализируем движок с нужной моделью. Это произойдет один раз при запуске.
    engine = initialize_model("/home/recsys1_user01/recipe_recommender/qwen_finetuned_merged")
    print("Модель запущена")
    while True:
        result = await get_recs(engine, 1, 5)
        # тут надо сделать матчинг с реальными товарами и выдавать реки
        print("\n--- Результат ---")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())