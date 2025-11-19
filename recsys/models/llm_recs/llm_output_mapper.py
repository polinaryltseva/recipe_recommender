# recsys/models/llm_recs/llm_output_mapper.py

from typing import List
import ast

from db import get_all_products  # уже есть в твоём db.py


class LLMOutputSearchDB:
    """
    Упрощённый маппер вывода LLM -> product_id из нашей SQLite-базы.

    Идея:
    - LLM возвращает строку вида "['молоко', 'масло', 'сыр']"
    - мы парсим её в список строк;
    - для каждого запроса ищем товары, где query входит в name или description;
    - собираем id, удаляем дубликаты, режем по limit.
    """

    def __init__(self):
        # можно кэшировать список продуктов
        self._products_cache = None

    def _load_products(self):
        if self._products_cache is None:
            # get_all_products возвращает tuples: (id, name, price, category_id, image_url, description)
            self._products_cache = list(get_all_products())
        return self._products_cache

    def search2list(self, llm_outputs: List[str] | str, limit: int = 20) -> List[int]:
        """
        Принимает:
        - строку с Python-списком (как возвращает LLM),
        - либо уже список строк.

        Возвращает:
        - список product_id (int) длиной до limit.
        """
        # --- 1. Приводим к списку строк ---
        if isinstance(llm_outputs, str):
            try:
                parsed = ast.literal_eval(llm_outputs)
                if isinstance(parsed, list):
                    queries = [str(x) for x in parsed]
                else:
                    queries = [str(parsed)]
            except Exception:
                queries = [llm_outputs]
        else:
            queries = [str(q) for q in llm_outputs]

        products = self._load_products()
        matched_ids: List[int] = []

        # --- 2. Для каждого текстового запроса ищем по name/description ---
        for q in queries:
            q_norm = q.strip().lower()
            if not q_norm:
                continue

            for (pid, name, price, category_id, image_url, description) in products:
                text = ((name or "") + " " + (description or "")).lower()
                if q_norm in text:
                    matched_ids.append(int(pid))

        # --- 3. Убираем дубликаты, сохраняем порядок ---
        seen = set()
        uniq_ids: List[int] = []
        for pid in matched_ids:
            if pid not in seen:
                seen.add(pid)
                uniq_ids.append(pid)

        return uniq_ids[:limit]
