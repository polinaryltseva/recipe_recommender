# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List
# from icecream import ic

# import pandas as pd
# from whoosh import scoring
# from whoosh.analysis import StemmingAnalyzer
# from whoosh.analysis import LanguageAnalyzer
# from whoosh.fields import ID, TEXT, Schema
# from whoosh.filedb.filestore import RamStorage
# from whoosh.qparser import MultifieldParser, OrGroup, AndGroup

# from whoosh.query import And, FuzzyTerm, Or, Term


# @dataclass(frozen=True)
# class MatchResult:
#     pov_id: int
#     vv_id: int
#     ingredient: str
#     queries: List[str]


# class WhooshCatalogMatcher:
#     """Lightweight in-memory Whoosh matcher for arbitrary catalogues."""

#     def __init__(self, df: pd.DataFrame, id_col: str = "id", text_col: str = "name"):
#         normalized = (
#             df[[id_col, text_col]]
#             .dropna()
#             .drop_duplicates(subset=[id_col])
#             .rename(columns={id_col: "id", text_col: "name"})
#         )
#         normalized["id"] = normalized["id"].astype(int)
#         normalized["name"] = normalized["name"].astype(str)
#         self.df = normalized.reset_index(drop=True)
#         self.schema = (
#             Schema(  # схема для индекса (как именно будут храниться данные в индексе)
#                 id=ID(stored=True, unique=True),
#                 name=TEXT(stored=True, analyzer=StemmingAnalyzer()),
#                 name_exact=TEXT(stored=True),
#             )
#         )
#         self._index = self._build_index()

#     def _build_index(self):  # создаёт индекс для поиска
#         storage = RamStorage()
#         idx = storage.create_index(self.schema)
#         writer = idx.writer()
#         for row in self.df.itertuples():
#             writer.add_document(id=str(row.id), name=row.name, name_exact=row.name)
#         writer.commit()
#         return idx

#     def search(
#         self, query: str, limit: int = 10
#     ) -> pd.DataFrame:  # поиск по запросу (запрос - это название ингредиента)
#         query = (query or "").strip()
#         print("QUERY: ", query)
#         if not query:
#             return self.df.head(0)

#         terms = [token for token in query.lower().split() if token]
#         if not terms:
#             return self.df.head(0)
#         print("TERMS: ", terms)
#         # ic(terms)
#         subqueries = []
#         # создаём подзапросы для каждого токена (каждого слова в запросе)
#         for token in terms:
#             if len(token) < 3:
#                 subqueries.append(Or([Term("name", token), Term("name_exact", token)]))
#             else:
#                 subqueries.append(
#                     Or(
#                         [
#                             FuzzyTerm("name", token, maxdist=1, prefixlength=2),
#                             Term("name_exact", token),
#                         ]
#                     )
#                 )
#         print("SUBQUERIES: ", subqueries)
#         whoosh_query = And(subqueries)  # это общий запрос по всем подзапросам
#         print("WHOOSH QUERY: ", whoosh_query)
#         rows: List[Dict[str, str]] = []
#         # создаём список для хранения результатов
#         with self._index.searcher(weighting=scoring.BM25F()) as searcher:
#             hits = searcher.search(whoosh_query, limit=limit)
#             ic(hits)
#             for hit in hits:
#                 # ic(hit)
#                 rows.append({"id": int(hit["id"]), "name": hit["name"]})
#         print("ROWS: ", rows)
#         if not rows:
#             print("FALLBACK БЛЯТЬ")
#             return self.df.head(0)
#         return pd.DataFrame(rows)

    
#     # def search(
#     #     self, query: str, limit: int = 10
#     # ) -> pd.DataFrame:
#     #     query = (query or "").strip()
#     #     ic(query)
#     #     if not query:
#     #         return self.df.head(0)

#     #     # Указываем, в каких полях искать.
#     #     # 'name' будет использоваться для поиска по основе слова (стемминг).
#     #     # 'name_exact' - для поиска по точному слову.
#     #     # Оператор по умолчанию между словами запроса - AND.
#     #     parser = MultifieldParser(
#     #     ["name", "name_exact"], schema=self.schema, group=AndGroup
#     # )

#     #     # Парсер сам обработает строку запроса:
#     #     # токенизирует, приведет к нижнему регистру и применит стемминг для поля 'name'.
#     #     whoosh_query = parser.parse(query)
#     #     print("PARSED WHOOSH QUERY: ", whoosh_query)

#     #     rows: List[Dict[str, str]] = []
#     #     with self._index.searcher(weighting=scoring.BM25F()) as searcher:
#     #         hits = searcher.search(whoosh_query, limit=limit)
#     #         ic(hits)
#     #         for hit in hits:
#     #             ic(hit)
#     #             rows.append({"id": int(hit["id"]), "name": hit["name"]})

#     #     if not rows:
#     #         print("FALLBACK")
#     #         return self.df.head(0)
#     #     ic(rows)
#     #     return pd.DataFrame(rows)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from whoosh import scoring
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.filedb.filestore import RamStorage
from whoosh.query import And, FuzzyTerm, Or, Term


@dataclass(frozen=True)
class MatchResult:
    pov_id: int
    vv_id: int
    ingredient: str
    queries: List[str]


class WhooshCatalogMatcher:
    """Lightweight in-memory Whoosh matcher with progressive fallback strategy."""

    def __init__(self, df: pd.DataFrame, id_col: str = "id", text_col: str = "name"):
        normalized = (
            df[[id_col, text_col]]
            .dropna()
            .drop_duplicates(subset=[id_col])
            .rename(columns={id_col: "id", text_col: "name"})
        )
        normalized["id"] = normalized["id"].astype(int)
        normalized["name"] = normalized["name"].astype(str)
        self.df = normalized.reset_index(drop=True)
        self.schema = Schema(
            id=ID(stored=True, unique=True),
            name=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            name_exact=TEXT(stored=True),
        )
        self._index = self._build_index()

    def _build_index(self):
        """Create in-memory Whoosh index from DataFrame."""
        storage = RamStorage()
        idx = storage.create_index(self.schema)
        writer = idx.writer()
        for row in self.df.itertuples():
            writer.add_document(id=str(row.id), name=row.name, name_exact=row.name)
        writer.commit()
        return idx

    def search(self, query: str, limit: int = 10) -> pd.DataFrame:
        """
        Search with progressive fallback strategy:
        1. AND matching (all terms must match) - most precise
        2. OR matching (any term matches) - medium recall
        3. Single-word matching (most specific word) - maximum recall
        """
        query = (query or "").strip()
        if not query:
            return self.df.head(0)

        terms = [token for token in query.lower().split() if token]
        if not terms:
            return self.df.head(0)

        # Strategy 1: Strict AND matching (all terms must match)
        results = self._search_and(terms, limit)
        if not results.empty:
            return results

        # Strategy 2: OR matching (any term matches)
        results = self._search_or(terms, limit)
        if not results.empty:
            return results

        # Strategy 3: Single most-specific word (usually the last one)
        results = self._search_single_word(terms, limit)
        if not results.empty:
            return results

        return self.df.head(0)

    def _build_term_query(self, token: str) -> Or:
        """Build a query for a single token with fuzzy + exact matching."""
        if len(token) < 3:
            return Or([Term("name", token), Term("name_exact", token)])
        else:
            return Or([
                FuzzyTerm("name", token, maxdist=1, prefixlength=2),
                Term("name_exact", token),
            ])

    def _search_and(self, terms: List[str], limit: int) -> pd.DataFrame:
        """Search requiring ALL terms to match."""
        subqueries = [self._build_term_query(token) for token in terms]
        whoosh_query = And(subqueries)
        return self._execute_search(whoosh_query, limit)

    def _search_or(self, terms: List[str], limit: int) -> pd.DataFrame:
        """Search requiring ANY term to match."""
        subqueries = [self._build_term_query(token) for token in terms]
        whoosh_query = Or(subqueries)
        return self._execute_search(whoosh_query, limit)

    def _search_single_word(self, terms: List[str], limit: int) -> pd.DataFrame:
        """
        Search with single most-specific word.
        Strategy: try last word first (usually more specific than descriptors),
        then try each word in reverse order.
        
        Examples:
        - "зелень петрушка" → tries "петрушка" first (the ingredient)
        - "грибы шампиньоны" → tries "шампиньоны" first (more specific than "грибы")
        - "перец черный" → tries "черный" first, then "перец"
        """
        for token in reversed(terms):
            whoosh_query = self._build_term_query(token)
            results = self._execute_search(whoosh_query, limit)
            if not results.empty:
                return results

        return self.df.head(0)

    def _execute_search(self, whoosh_query, limit: int) -> pd.DataFrame:
        """Execute a Whoosh query and return results as DataFrame."""
        rows: List[Dict[str, str]] = []
        with self._index.searcher(weighting=scoring.BM25F()) as searcher:
            hits = searcher.search(whoosh_query, limit=limit)
            for hit in hits:
                rows.append({"id": int(hit["id"]), "name": hit["name"]})

        if rows:
            return pd.DataFrame(rows)
        else:
            return self.df.head(0)