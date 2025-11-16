#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import ast
import gc

# Очистка памяти
gc.collect()
torch.cuda.empty_cache()


# In[3]:


model_name = "Qwen/Qwen3-14B"

# Загрузка модели в bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Настраиваем модель для PEFT (Parameter-Efficient Fine-Tuning) с использованием LoRA.
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# In[5]:


import ast
from datasets import load_dataset

dataset = load_dataset("csv", data_files="/home/recsys1_user01/recipe_recommender/data-ready_lmm_train.csv", split="train")
dataset = dataset.filter(lambda example: example["split"] == "train")
print(f"Датасет успешно загружен. Количество записей: {len(dataset)}")

# Определяем неизменяемую часть промптов
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
4. Вывод должен быть в формате [<запрос 1>, <запрос 2>, ...]

### ПРИМЕР (для демонстрации логики, а не для копирования)
- Пример входной корзины: [Мука пшеничная; Яйца куриные; Сахар-песок]
- Пример правильного вывода: ["разрыхлитель", "ванильный экстракт", "сливочное масло", "шоколад", "кондитерские украшения"]"""

def format_data_as_messages(example):
    try:
        input_basket_list = ast.literal_eval(example["support_items"])
        target_basket_list = ast.literal_eval(example["holdout_items"])
    except (ValueError, SyntaxError):
        return {"text": None}

    # 1. Собираем контент для каждого сообщения
    current_basket_str = "; ".join(input_basket_list)
    user_content = USER_PROMPT_TEMPLATE.format(current_basket=current_basket_str)
    assistant_content = str(target_basket_list)

    # 2. Создаем структуру messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    # 3. Используем токенизатор для преобразования messages в одну строку для обучения
    # add_generation_prompt=False, так как мы предоставляем и ответ ассистента
    return { "text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) }

# Применяем новую функцию форматирования
dataset = dataset.map(format_data_as_messages)
dataset = dataset.filter(lambda example: example["text"] is not None)
print("Данные отформатированы с использованием chat-шаблона.")


# In[6]:


dataset[0]


# In[7]:


print("Первый пример из датасета:")
print(dataset[0]["text"])
print(f"\nДлина текста: {len(dataset[0]['text'])}")

# Проверка токенизации
sample = dataset[0]["text"]
tokens = tokenizer(sample, truncation=True, max_length=2048)
print(f"Количество токенов: {len(tokens['input_ids'])}")


# In[15]:


from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    dataloader_pin_memory=False,

    # SFT-specific:
    dataset_text_field="text",
    max_length=2048,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

trainer.train()


# In[ ]:


output_dir = "./qwen_finetuned-bf16"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Модель сохранена в {output_dir}")

# Можно также сохранить полную модель
model.save_pretrained(output_dir)


# In[ ]:


from transformers import pipeline

# Загрузка для инференса
model_for_inference = AutoModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer_for_inference = AutoTokenizer.from_pretrained(output_dir)

# Тестовый пример
test_input = ['булочки для бургеров', 'кетчуп', 'горчица', 'лук репчатый', 'помидоры']
current_basket_str = "; ".join(test_input)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(current_basket=current_basket_str)},
]

# Форматирование промпта
prompt = tokenizer_for_inference.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Генерация
inputs = tokenizer_for_inference(prompt, return_tensors="pt").to(model_for_inference.device)
outputs = model_for_inference.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.0,
    pad_token_id=tokenizer_for_inference.eos_token_id
)

response = tokenizer_for_inference.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Сгенерированный ответ:", response)

