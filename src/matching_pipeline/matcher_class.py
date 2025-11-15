import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import tqdm
from rapidfuzz import fuzz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Токенизация (lower) только внутри матчинга
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)


def tokenize_lower(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def to_index_text(*parts: str) -> str:
    return " ".join(p for p in parts if isinstance(p, str) and p.strip())


def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    uni = len(sa | sb)
    return 0.0 if uni == 0 else inter / uni


def has_whole_word_ci(word: str, text: str) -> bool:
    if not word or not text:
        return False
    pattern = rf"(?<![A-Za-zА-Яа-яЁё0-9]){re.escape(word)}(?![A-Za-zА-Яа-яЁё0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _normalize_query_for_matching(text: str) -> str:
    tokens = tokenize_lower(text or "")
    tokens = [t for t in tokens if not t.startswith("цедр")]
    # кус-кус → кускус (после разбиения на токены "кус", "кус")
    merged: List[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == "кус" and tokens[i + 1] == "кус":
            merged.append("кускус")
            i += 2
            continue
        merged.append(tokens[i])
        i += 1
    return " ".join(merged).strip()


def _rewrite_special_ingredients(text: str) -> Optional[str]:
    t = (text or "").lower()
    # любые формы «желток/желтки/яичные желтки» → «яйца»
    if "желтк" in t:
        return "яйцо куриное"
    return None


_GENERIC_RU: Set[str] = {
    "масло",
    "смесь",
    "икра",
    "сироп",
    "крем",
    "тесто",
    "пюре",
    "соус",
    "напиток",
    "цедра",
}  # Эти слова могут требовать дополнительных уточнений


@dataclass
class FaissTfidfIndex:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    index: "faiss.Index"
    X: np.ndarray  # (n_docs, dims) float32, L2-normalized

    @classmethod
    def build(
        cls, docs: Sequence[str], *, dims: int = 256, max_features: int = 200_000
    ) -> "FaissTfidfIndex":
        """
        Строим Faiss index
        """
        texts = [d if isinstance(d, str) else "" for d in docs]
        vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)[A-Za-zА-Яа-яЁё0-9]+",  # берем русские английские слова и циферки
            lowercase=True,
            min_df=1,
            max_features=max_features,
        )
        M = vec.fit_transform(texts)
        svd = TruncatedSVD(n_components=dims, random_state=42)
        X = svd.fit_transform(M).astype("float32")
        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(dims)  # косинус через dot по L2-нормализации
        index.add(X)
        return cls(vectorizer=vec, svd=svd, index=index, X=X)  # строим индекс

    def _embed_query(self, query: str) -> np.ndarray:  # эмбеддим запрос
        q = self.vectorizer.transform([query or ""])
        q = self.svd.transform(q).astype("float32")
        faiss.normalize_L2(q)
        return q  # shape (1, dims)

    def score_query(
        self, query: str, top_k: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:  # скорим метрику запрса
        q = self._embed_query(query)
        scores, idx = self.index.search(q, min(top_k, self.X.shape[0]))
        return scores[0], idx[0]  # скорим и получаем индекс

    def score_doc(self, query: str, doc_idx: int) -> float:
        q = self._embed_query(query)
        return float(np.dot(self.X[doc_idx], q[0]))


from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class VVCatalogItem:
    doc_id: int
    title: str
    description: str


class VVCatalogIndex:
    def __init__(
        self,
        penalize_ready_to_eat: bool = True,
    ) -> None:
        self.items: List[VVCatalogItem] = []
        self.title_texts: List[str] = []
        self.all_texts: List[str] = []
        self.penalize_ready_to_eat = penalize_ready_to_eat
        self._titles_fuzzy: List[str] = []
        self._faiss_title: Optional[FaissTfidfIndex] = None

    def fit(
        self,
        df: pd.DataFrame,
        text_fields: Sequence[str] = ("name", "description"),
    ) -> "VVCatalogIndex":
        df = df.copy()

        # Check required columns
        if "id" not in df.columns or "name" not in df.columns:
            raise ValueError("DataFrame must contain 'id' and 'name' columns")

        print(f"Building index from {len(df)} items...")

        self.items = [
            VVCatalogItem(
                doc_id=int(r["id"]),
                title=str(r["name"]),
                description=str(r["description"])
                if "description" in df.columns and pd.notna(r.get("description"))
                else "",
            )
            for i, r in tqdm.tqdm(df.iterrows(), total=len(df), desc="Loading items")
        ]

        self.title_texts = [str(r["name"]) for _, r in df.iterrows()]
        self.all_texts = [
            to_index_text(
                str(r["name"]),
                str(r["description"])
                if "description" in df.columns and pd.notna(r.get("description"))
                else "",
            )
            for _, r in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing texts")
        ]

        self._titles_fuzzy = [str(t).lower() for t in self.title_texts]
        print("Building FAISS index...")
        self._faiss_title = FaissTfidfIndex.build(self.title_texts, dims=256)
        print("FITTED")
        return self

    def _candidate_ids(
        self, query: str, k_title: int = 600, k_all: int = 150, k_faiss: int = 300
    ) -> List[int]:
        cand_ids: List[int] = []
        if self._faiss_title is not None:
            _, idx3 = self._faiss_title.score_query(query, top_k=k_faiss)
            cand_ids.extend(idx3.tolist())
        seen, unique = set(), []
        for cid in cand_ids:
            if cid not in seen:
                unique.append(cid)
                seen.add(cid)
        return unique

    def _fuzzy_title_score(self, query: str, doc_idx: int) -> float:
        if not self._titles_fuzzy:
            return 0.0
        q = (query or "").lower()
        return float(fuzz.token_set_ratio(q, self._titles_fuzzy[doc_idx]) / 100.0)

    def _faiss_title_score(self, query: str, doc_idx: int) -> float:
        if self._faiss_title is None:
            return 0.0
        return self._faiss_title.score_doc(query, doc_idx)

    def match_one(
        self,
        ingredient_name: str,
        *,
        top_k: int = 5,
        k_faiss: int = 300,
        min_title_signal: float = 0.02,
        min_fuzzy: float = 0.7,
        min_vec_sim=0.18,
        w_fuzzy: float = 0.8,
        penalty_mismatch: float = 0.8,
    ) -> List[Dict]:
        query = ingredient_name or ""

        raw_query = ingredient_name or ""
        rewrite = _rewrite_special_ingredients(raw_query)
        query = _normalize_query_for_matching(rewrite or raw_query)

        q_tokens_all = tokenize_lower(query)
        if not q_tokens_all:
            return []
        q_main = q_tokens_all
        first_key = q_main[0]

        cand_ids = self._candidate_ids(query, k_faiss=k_faiss)
        if not cand_ids:
            return []

        results: List[Tuple[float, int, Dict]] = []

        for cid in cand_ids:
            it = self.items[cid]
            title_tokens = tokenize_lower(it.title)

            title_l = it.title.lower()

            allow_by_word = any(has_whole_word_ci(t, it.title) for t in q_main)
            title_overlap = jaccard(q_main, title_tokens)
            vec_sim = self._faiss_title_score(query, cid)
            fuzzy = self._fuzzy_title_score(query, cid)

            score = 0.0

            # Кускус: требуем "кускус" в title или очень высокий fuzzy; штрафуем "Напитки"/"вкус"
            if "кускус" in q_main:
                if "кускус" not in title_tokens and fuzzy < 0.85:
                    continue
                if "вкус" in title_l:
                    score -= 1.5

            # Яйца: при запросе "яйца" (в т.ч. после переписывания желтков) — бонус за куриные, штраф за копчёные/перепелиные
            if ("яйца" in q_main) or ("яйцо" in q_main):
                if "курин" in title_l:
                    score += 0.6
                if "перепел" in title_l:
                    score -= 1.0
                if "копчен" in title_l:
                    score -= 2.5

            fuzzy = self._fuzzy_title_score(query, cid)

            # Гейт: целое слово ИЛИ пересечение токенов ИЛИ char‑симиляр ИЛИ fuzzy
            if (
                not allow_by_word
                and title_overlap < min_title_signal
                and vec_sim < min_vec_sim
                and fuzzy < min_fuzzy
            ):
                continue

            score += 1.0 * title_overlap
            score += 1.8 * vec_sim
            score += 1.5 * fuzzy

            # Слишком общие слова без уточнений — штраф
            if (first_key in _GENERIC_RU or q_main[-1] in _GENERIC_RU) and len(
                q_main
            ) >= 2:
                if len(set(q_main[1:]) & set(title_tokens)) == 0:
                    score -= 1.2

            if first_key not in title_tokens and vec_sim < 0.25:
                score -= penalty_mismatch

            reasons = {
                "title_overlap": round(float(title_overlap), 4),
                "vec_sim": round(float(vec_sim), 4),
                "fuzzy": round(float(fuzzy), 4),
            }
            results.append((score, cid, reasons))

        # Безопасный fallback: только если есть положительный сигнал
        if not results and cand_ids:
            best = None
            best_score = -1e9
            for cid in cand_ids[:200]:
                c = self._faiss_title_score(query, cid)
                fuzzy = self._fuzzy_title_score(query, cid)
                sc = 1.0 * c + w_fuzzy * fuzzy
                if sc > best_score:
                    best_score = sc
                    best = cid
            if best is not None and best_score > 0.25:
                it = self.items[best]
                return [
                    {
                        "score": round(float(best_score), 4),
                        "product_id": it.doc_id,
                        "title": it.title,
                        "description": it.description,
                        "reasons": {"fallback": True},
                    }
                ][:top_k]

            return []

        if not results:
            return []

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:top_k]

        out: List[Dict] = []
        for score, cid, reasons in top:
            it = self.items[cid]
            out.append(
                {
                    "score": round(float(score), 4),
                    "product_id": it.doc_id,
                    "title": it.title,
                    "description": it.description,
                    "reasons": reasons,
                }
            )

        return out

    def match_many(
        self,
        ingredients: Union[Iterable[str], pd.DataFrame],
        top_k: int = 1,
        ingredient_col: str = "cooker_vals",
        id_col: str = "cooker_id",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Match multiple ingredients to catalog items.

        Parameters:
        -----------
        ingredients : Union[Iterable[str], pd.DataFrame]
            Either an iterable of ingredient names, or a DataFrame with ingredient info
        top_k : int
            Number of top matches to return per ingredient
        ingredient_col : str
            Column name for ingredient names if DataFrame is provided (default: "cooker_vals")
        id_col : str
            Column name for ingredient IDs if DataFrame is provided (default: "cooker_id")
        **kwargs : additional arguments passed to match_one

        Returns:
        --------
        pd.DataFrame with columns: cooker_id, ingredient, product_id, match_title, match_url,
                                    match_category, match_subcategory, score, reasons
        """
        rows = []

        # Handle DataFrame input
        if isinstance(ingredients, pd.DataFrame):
            df = ingredients
            if ingredient_col not in df.columns:
                raise ValueError(f"Column '{ingredient_col}' not found in DataFrame")

            has_id = id_col in df.columns

            for idx, row in tqdm.tqdm(df.iterrows()):
                ing = row[ingredient_col]
                ing_id = row[id_col] if has_id else idx

                matches = self.match_one(ing, top_k=top_k, **kwargs)
                if not matches:
                    rows.append(
                        {
                            "cooker_id": ing_id,
                            "ingredient": ing,
                        }
                    )
                    continue
                for m in matches:
                    rows.append(
                        {
                            "cooker_id": ing_id,
                            "ingredient": ing,
                            "product_id": m["product_id"],
                            "match_title": m["title"],
                            "score": m["score"],
                            "reasons": m["reasons"],
                        }
                    )
        else:
            # Handle iterable of strings
            # for idx, ing in enumerate(ingredients):
            for idx, ing in enumerate(
                tqdm.tqdm(ingredients, desc="Matching ingredients")
            ):
                matches = self.match_one(ing, top_k=top_k, **kwargs)
                if not matches:
                    rows.append(
                        {
                            "cooker_id": idx,
                            "ingredient": ing,
                            "product_id": None,
                            "match_title": None,
                            "score": None,
                            "reasons": {},
                        }
                    )
                    continue
                for m in matches:
                    rows.append(
                        {
                            "cooker_id": idx,
                            "ingredient": ing,
                            "product_id": m["product_id"],
                            "match_title": m["title"],
                            "score": m["score"],
                            "reasons": m["reasons"],
                        }
                    )

        return pd.DataFrame(rows)


if __name__ == "__main__":
    import json

    df_product = pd.read_csv("products_final_categories2.csv")

    with open(
        "D:/gleb/Recsys_projeccts/sirius_seminars/vkusvill_case/recsys_tests/items_dict.json",
        "r",
        encoding="utf-8",
    ) as f:
        id_to_name = json.load(f)

    cooker_df = pd.DataFrame(
        {"cooker_id": list(id_to_name.keys()), "cooker_vals": list(id_to_name.values())}
    )
    idx = VVCatalogIndex(penalize_ready_to_eat=True).fit(
        df_product[["id", "name", "description", "composition"]]
    )

    df_matches = idx.match_many(
        cooker_df,
        top_k=1,
        min_title_signal=0.02,
        penalty_mismatch=0.6,
    )
    print(df_matches.head(100).to_string(index=False))
    df_matches.to_csv("df_matches.csv")
