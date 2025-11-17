import time
import ast
import threading

from transformers import AutoTokenizer
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
# tuned model path Qwen/Qwen3-14b
class vllm_recomender:
    def __init__(self, model_path="/home/recsys1_user01/recipe_recommender/llm_work/recipe_recommender/models/llm_recs"):
        self.model_path = model_path
        self.engine, self.tokenizer = self.initialize_model(model_path)
        self.lock = threading.Lock() 

        self.SYSTEM_PROMPT = """/no_think Ты — ИИ-ассистент для системы продуктовых рекомендаций. Твоя задача — анализировать корзину и генерировать идеи для поиска недостающих для рецептов товаров в виде списка поисковых запросов."""

        self.USER_PROMPT_TEMPLATE = """Контекст:
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


    def initialize_model(self, model_name: str):
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


    def get_current_cart(self, cart, user_id: int) -> list[str]:
        """
        Функция-заглушка, возвращающая текущую корзину пользователя.
        """
        prompt_str = cart
        prompt_list = [item.strip() for item in prompt_str.split(";") if item.strip()]

        return prompt_list

    def get_previous_n_cart(self, user_id: int, n: int) -> list[str]:
        """Функция-заглушка, возвращающая прошлые n покупок."""
        return []

    def get_simillar_n_cart(self, user_id: int, n: int, current_cart: list = None) -> list[str]:
        """Функция-заглушка, возвращающая n наиболее похожих прошлых покупок."""
        return []


    def get_recs(self, cart, user_id: int, context_window: int) -> str:
        """
        Синхронно генерирует рекомендации, используя LLMEngine.
        """
        curr_cart = self.get_current_cart(cart, user_id)
        prev_cart_str = "; ".join(self.get_previous_n_cart(user_id, context_window))
        sim_cart_str = "; ".join(self.get_simillar_n_cart(user_id, context_window, curr_cart))

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(
                current_basket="; ".join(curr_cart),
                n=context_window,
                prev_cart=prev_cart_str,
                sim_cart=sim_cart_str
            )},
        ]

        final_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(max_tokens=200, temperature=0.0)
        request_id = random_uuid()
        with self.lock:
            start_time = time.time()
            
            # ИСПРАВЛЕНИЕ: Передаем только prompt (строку), а не prompt_token_ids.
            # Движок сам выполнит токенизацию.
            self.engine.add_request(request_id, final_prompt, sampling_params)

            final_output = []
            while self.engine.has_unfinished_requests():
                # Выполняем шаг генерации
                request_outputs = self.engine.step()
                
                # Проверяем завершенные на этом шаге запросы
                for output in request_outputs:
                    if output.finished:
                        final_output.append(output)
            print(f"Время работы LLM: {time.time()-start_time:.2f} сек.")

        if not final_output or not final_output[0].outputs:
            return ""
        text = final_output[0].outputs[0].text
        parsed_recs = self.parse_llm_response(text.strip())
        return parsed_recs


    def parse_llm_response(self, text: str) -> list[str]:
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

'''
def main():
    """
    Основной синхронный цикл программы.
    """
    model_path = "/home/recsys1_user01/recipe_recommender/qwen_finetuned_merged"
    # model_path = "Qwen/Qwen3-14b"
    model = vllm_recomender(model_path)

    try:
        while True:
            recs = model.get_recs(input("Введите список продуктов через ; "), 1, 5)
            if recs:
                print("Сгенерированные рекомендации:", recs)
            else:
                print("Не удалось сгенерировать рекомендации.")

    except (KeyboardInterrupt, SystemExit):
        print("\nПрограмма завершена.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()'''