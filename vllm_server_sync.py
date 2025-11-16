import sys
import time
import uuid
import ast

from transformers import AutoTokenizer
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

SYSTEM_PROMPT = """/no_think Ты — ИИ-ассистент для системы продуктовых рекомендаций. Твоя задача — анализировать корзину и генерировать идеи для поиска недостающих для рецептов товаров в виде списка поисковых запросов."""

USER_PROMPT_TEMPLATE = """Контекст:
Текущая корзина: {current_basket}
Прошлые {n} покупок пользователя: {prev_cart}
Прошлые {n} наиболее похожих покупок пользователя: {sim_cart}
### ЗАДАЧА
На основе КОНТЕКСТА, сгенерируй 10 поисковых запросов, которые помогут пользователю добавить недостающие для приготовления блюд товары в корзину.
Важные правила:
1. Запросы должны быть краткими, отражать общие категории или идеи, а не конкретными товарами с брендом или весом.
2. Не повторяй товары, которые уже есть в корзине и не допускай повторений в генерируемом списке.
3. Поисковые запросы должны быть реалистичными для продуктового магазина с 10-15 тысячами наименований
4. Вывод должен быть в формате [<запрос 1>, <запрос 2>, ...]

### ПРИМЕР (для демонстрации логики, а не для копирования)
- Пример входной корзины: [Мука пшеничная; Яйца куриные; Сахар-песок]
- Пример правильного вывода: ["разрыхлитель", "ванильный экстракт", "сливочное масло", "шоколад", "кондитерские украшения"]
"""


def initialize_model(model_name: str):
    """
    Инициализирует движок VLLM и связанный с ним токенизатор в синхронном режиме.
    Возвращает кортеж (engine, tokenizer).
    """
    print(f"Загрузка модели из '{model_name}'...")
    engine_args = EngineArgs(
        model=model_name,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        trust_remote_code=True
    )
    engine = LLMEngine.from_engine_args(engine_args)
    # Важно: получаем токенизатор именно из движка, чтобы гарантировать соответствие
    tokenizer = engine.tokenizer

    print("Модель и токенизатор успешно загружены.")
    return engine, tokenizer


def get_current_cart(user_id: int) -> list[str]:
    """
    Функция-заглушка, возвращающая текущую корзину пользователя.
    """
    prompt_str = input("Введите товары через '; ': ")
    prompt_list = [item.strip() for item in prompt_str.split(";") if item.strip()]

    return prompt_list

def get_previous_n_cart(user_id: int, n: int) -> list[str]:
    """Функция-заглушка, возвращающая прошлые n покупок."""
    return []

def get_simillar_n_cart(user_id: int, n: int, current_cart: list = None) -> list[str]:
    """Функция-заглушка, возвращающая n наиболее похожих прошлых покупок."""
    return []


def get_recs(engine: LLMEngine, tokenizer, user_id: int, context_window: int) -> str:
    """
    Синхронно генерирует рекомендации, используя LLMEngine.
    """
    curr_cart = get_current_cart(user_id)
    prev_cart_str = "; ".join(get_previous_n_cart(user_id, context_window))
    sim_cart_str = "; ".join(get_simillar_n_cart(user_id, context_window, curr_cart))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            current_basket="; ".join(curr_cart),
            n=context_window,
            prev_cart=prev_cart_str,
            sim_cart=sim_cart_str
        )},
    ]

    final_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(max_tokens=200, temperature=0.0)
    request_id = random_uuid()
    
    start_time = time.time()
    
    # ИСПРАВЛЕНИЕ: Передаем только prompt (строку), а не prompt_token_ids.
    # Движок сам выполнит токенизацию.
    engine.add_request(request_id, final_prompt, sampling_params)

    final_output = []
    while engine.has_unfinished_requests():
        # Выполняем шаг генерации
        request_outputs = engine.step()
        
        # Проверяем завершенные на этом шаге запросы
        for output in request_outputs:
            if output.finished:
                final_output.append(output)
    print(f"Время работы LLM: {time.time()-start_time:.2f} сек.")

    if not final_output or not final_output[0].outputs:
        return ""

    text = final_output[0].outputs[0].text
    return text.strip()


def parse_llm_response(text: str) -> list[str]:
    try:
        start_index = text.find('[')
        end_index = text.rfind(']')

        if start_index == -1 or end_index == -1:
            return []

        parsed_list = ast.literal_eval(text[start_index : end_index + 1])

        if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
            return parsed_list
        else:
            return []

    except (ValueError, SyntaxError, TypeError):
        return []


def main():
    """
    Основной синхронный цикл программы.
    """
    model_path = "/home/recsys1_user01/recipe_recommender/qwen_finetuned_merged"
    # model_path = "Qwen/Qwen3-14b"
    engine, tokenizer = initialize_model(model_path)

    try:
        while True:
            raw_result = get_recs(engine, tokenizer, 1, 5)
            if raw_result:
                parsed_result = parse_llm_response(raw_result)
                print("Сгенерированные рекомендации:", parsed_result)
            else:
                print("Не удалось сгенерировать рекомендации.")

    except (KeyboardInterrupt, SystemExit):
        print("\nПрограмма завершена.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()