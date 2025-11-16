import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# --- НАСТРОЙКИ ---
# Путь к папке с вашим обученным LoRA-адаптером
adapter_path = "./qwen_finetuned-bf16" 
# Путь к папке со смердженной моделью
merged_model_path = "./qwen_finetuned_merged"
# ------------------

# --- Общие данные для теста ---
SYSTEM_PROMPT = """Ты — ИИ-ассистент для системы продуктовых рекомендаций.
Твоя задача — анализировать корзину и генерировать идеи для поиска недостающих товаров в виде списка поисковых запросов."""

USER_PROMPT_TEMPLATE = """Контекст:
    Текущая корзина: {current_basket}
    Прошлые 5 покупок пользователя:
    Прошлые 5 наиболее похожих покупок пользователя:
### ЗАДАЧА
На основе КОНТЕКСТА, сгенерируй не более 20 поисковых запросов, которые помогут пользователю добавить недостающие товары в корзину.
Важные правила:
1. Запросы должны быть краткими, отражать общие категории или идеи, а не конкретными товарами с брендом или весом.
2. Не повторяй товары, которые уже есть в корзине.
3. Поисковые запросы должны быть реалистичными для продуктового магазина с 10-15 тысячами наименований
4. Вывод должен быть в формате [<запрос 1>, <запрос 2>, ...]"""

test_input = ['булочки для бургеров', 'кетчуп', 'горчица', 'лук репчатый', 'помидоры']
current_basket_str = "; ".join(test_input)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(current_basket=current_basket_str)},
]

# --- 1. ИНФЕРЕНС БЕЗ ЯВНОГО СЛИЯНИЯ (ВАШ КОД) ---

print("--- 1. Тест модели с адаптером 'на лету' (без слияния) ---")

# Загрузка базовой модели + адаптера
model_with_adapter = AutoModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model_with_adapter.device)

# Генерация с детерминированными параметрами для сравнения
# При temperature=0.0 параметр do_sample=True игнорируется, но для ясности лучше do_sample=False
outputs = model_with_adapter.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False, # Установлено в False для детерминированного вывода
    temperature=0.0,
    pad_token_id=tokenizer.eos_token_id
)

response_adapter = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Сгенерированный ответ (адаптер):", response_adapter)

# Очистка памяти
del model_with_adapter
gc.collect()
torch.cuda.empty_cache()

# --- 2. ИНФЕРЕНС ПОСЛЕ СЛИЯНИЯ ---

print("\n--- 2. Тест смердженной модели ---")

# Загрузка объединенной модели
merged_model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
# Токенизатор тот же самый
# tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)

# Промпт и inputs остаются теми же
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

# Генерация с теми же параметрами
outputs = merged_model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.0,
    pad_token_id=tokenizer.eos_token_id
)

response_merged = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Сгенерированный ответ (смердженная):", response_merged)

# Очистка памяти
del merged_model
gc.collect()
torch.cuda.empty_cache()

# --- 3. СРАВНЕНИЕ ---

print("\n--- 3. Результат сравнения ---")
if response_adapter.strip() == response_merged.strip():
    print("ВЫВОД: Успех. Результаты идентичны. Процесс слияния прошел корректно.")
else:
    print("ВЫВОД: Ошибка. Результаты отличаются. Проблема в процессе слияния или сохранения/загрузки.")