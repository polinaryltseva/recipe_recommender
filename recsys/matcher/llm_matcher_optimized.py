from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

from .ease_scorer import EaseOfflineScorer
from .whoosh_matcher import WhooshCatalogMatcher
from ..models.ease_popular.actual_popular import DbTopPopular
from icecream import ic 



class LLMMatcher:
    """
    Optimized LLM matcher for production use.
    - Uses dictionaries instead of DataFrames
    - Singleton pattern for Whoosh indices
    - Precomputed product metadata
    - Simple API: List[str] -> List[int]
    """

    # Top popular VV product IDs (fallback when EASE fails or results insufficient)
    

    # Class-level cache for indices and metadata (singleton pattern)
    _pov_searcher = None
    _vv_searcher = None
    _pov_to_vv = None
    _vv_metadata = None
    _items_dict = None
    _ease_scorer = None

    def __init__(self, project_root: Path | None = None):
        """
        Initialize the matcher. Heavy resources are loaded once and cached.

        Args:
            project_root: Root directory containing data files.
                         Defaults to src/new_matcher directory.
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parent

        self.project_root = Path(project_root)
        toppopular = DbTopPopular(k=50)
        self.TOP_POPULAR_VV_IDS = toppopular.get_popular_products()
        # Default reranking weights
        self.DEFAULT_WEIGHTS = {"ease": 0.6, "rating": 0.25, "price": 0.15}

        # Load resources (singleton pattern - only loaded once)
        self._ensure_resources_loaded()

    @classmethod
    def _ensure_resources_loaded(cls):
        """Load all resources once and cache them at class level."""
        if cls._pov_searcher is not None:
            # Already loaded
            return

        project_root = Path(__file__).resolve().parent

        # 1. Load items_dict (POV ID -> ingredient name)
        items_dict_path = project_root.parent.parent / "models/matcher/items_dict.json"
        with open(items_dict_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
            cls._items_dict = {int(k): v for k, v in raw_items.items()}

        # 2. Load POV to VV mapping

        pov_mapping_path = project_root.parent.parent / "models/matcher/items_rows_pov2vv.json"
        with open(pov_mapping_path, "r", encoding="utf-8") as f:
            pov_records = json.load(f)

        # Build dictionary: pov_id -> list of (vv_id, ingredient_name)
        cls._pov_to_vv = {}
        for record in pov_records:
            # Skip records with missing data
            if record.get("pov_id") is None or record.get("vv_id") is None:
                continue

            pov_id = int(record["pov_id"])
            vv_id = int(record["vv_id"])
            ingredient = record["ingredient"]

            if pov_id not in cls._pov_to_vv:
                cls._pov_to_vv[pov_id] = []
            cls._pov_to_vv[pov_id].append((vv_id, ingredient))

        # 3. Build POV catalog for Whoosh search
        pov_catalog = [
            {"id": pov_id, "name": ingredient_name}
            for pov_id, ingredient_name in cls._items_dict.items()
        ]
        # ic()
        # Convert to simple format for Whoosh (pandas only used here for Whoosh compatibility)
        import pandas as pd

        pov_catalog_df = pd.DataFrame(pov_catalog)
        ic(pov_catalog_df)
        cls._pov_searcher = WhooshCatalogMatcher(
            pov_catalog_df, id_col="id", text_col="name"
        )

        # 4. Load VV catalog from preprocessed JSON (much faster than CSV)
        vv_catalog_path = project_root.parent.parent / "models/matcher/vv_catalog.json"
        with open(vv_catalog_path, "r", encoding="utf-8") as f:
            vv_catalog_dict = json.load(f)

        # Convert to list format for Whoosh
        vv_catalog = [
            {"id": int(vv_id), "name": name} for vv_id, name in vv_catalog_dict.items()
        ]
        vv_catalog_df = pd.DataFrame(vv_catalog)
        cls._vv_searcher = WhooshCatalogMatcher(
            vv_catalog_df, id_col="id", text_col="name"
        )

        # 5. Load product metadata from shop.db

        shop_db_path = project_root.parent.parent /  "db/shop.db"
        # TODO import db
        # shop_db_path = "se/src/new_matcher/shop.db"
        cls._vv_metadata = cls._load_product_metadata(shop_db_path)

        # 6. Load EASE model

        ease_model_path = project_root.parent.parent / "models/ease_popular/ease_model.pkl"
        # TODO: import ease_model
        # ease_model_path = "D:/gleb/Recsys_projeccts/sirius_seminars/vkusvill_case/src/new_matcher/ease_model.pkl"
        cls._ease_scorer = EaseOfflineScorer(ease_model_path)

    @staticmethod
    def _load_product_metadata(db_path: Path) -> Dict[int, Dict[str, float]]:
        """
        Load product metadata from SQLite into a dictionary.

        Returns:
            {vv_id: {"price": float, "rating": float, "category_id": int}}
        """
        metadata = {}
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, price, rating, category_id FROM product")
            for row in cursor.fetchall():
                vv_id, price, rating, category_id = row
                metadata[vv_id] = {
                    "price": price if price else 0.0,
                    "rating": rating if rating else 0.0,
                    "category_id": category_id if category_id else 0,
                }

        # Fill missing prices with category medians
        category_prices = {}
        for vv_id, meta in metadata.items():
            cat_id = meta["category_id"]
            if meta["price"] > 0:
                if cat_id not in category_prices:
                    category_prices[cat_id] = []
                category_prices[cat_id].append(meta["price"])

        category_medians = {
            cat_id: sorted(prices)[len(prices) // 2]
            for cat_id, prices in category_prices.items()
            if prices
        }

        # Apply medians
        global_median = sorted(
            [m["price"] for m in metadata.values() if m["price"] > 0]
        )[len(metadata) // 2]

        for vv_id, meta in metadata.items():
            if meta["price"] <= 0:
                cat_id = meta["category_id"]
                meta["price"] = category_medians.get(cat_id, global_median)

        return metadata

    def match_llm_output(
        self,
        llm_outputs: List[str],
        current_cart: List[int],
        top_k: int = 20,
    ) -> List[int]:
        """
        Main API: Match LLM outputs to VkusVill product IDs.

        Args:
            llm_outputs: List of ingredient names from LLM (e.g., ["Масло сливочное", "Вода"])
            current_cart: List of POV IDs currently in user's cart (internal IDs 0-972)
            top_k: Number of VV product IDs to return

        Returns:
            List of VV product IDs (outer IDs from shop.db), length <= top_k
        """
        try:
            # Step 1: Map LLM outputs to POV IDs using Whoosh search
            pov_matches = self._map_llm_to_pov_ids(llm_outputs, per_query_limit=3)
            print(pov_matches)
            print( self._map_llm_to_pov_ids(llm_outputs, per_query_limit=top_k))

            if not pov_matches:
                # Fallback to top popular
                return self.TOP_POPULAR_VV_IDS[:top_k]

            # Step 2: Get EASE recommendations based on cart
            ease_scores = self._get_ease_scores(current_cart, pov_matches)
            # Step 3: Map POV IDs to VV product candidates
            vv_candidates = self._map_pov_to_vv_candidates(pov_matches)
            if not vv_candidates:
                return self.TOP_POPULAR_VV_IDS[:top_k]

            # Step 4: Rerank candidates using EASE + metadata
            reranked = self._rerank_candidates(vv_candidates, ease_scores, llm_outputs)

            # Step 5: Diversify by ingredient and extract top-K
            final_vv_ids = self._diversify_and_select(reranked, top_k)

            # Step 6: Fill with top popular if needed
            if len(final_vv_ids) < top_k:
                final_vv_ids = self._fill_with_popular(final_vv_ids, top_k)
            return final_vv_ids[:top_k]

        except Exception as e:
            # Fallback on any error
            print(f"Error in match_llm_output: {e}")
            return self.TOP_POPULAR_VV_IDS[:top_k]

    def _map_llm_to_pov_ids(
        self, llm_outputs: List[str], per_query_limit: int = 3
    ) -> List[Tuple[int, str, List[str]]]:
        """
        Map LLM outputs to POV IDs using Whoosh search.

        Returns:
            List of (pov_id, ingredient_name, [llm_queries])
        """
        matches = []
        for query in llm_outputs:
            if not query or not query.strip():
                continue

            # Search POV catalog
            results = self._pov_searcher.search(query, limit=per_query_limit)
            if results.empty:
                continue
            for _, row in results.iterrows():
                pov_id = int(row["id"])
                ingredient_name = row["name"]
                matches.append((pov_id, ingredient_name, [query]))

        return matches

    def _get_ease_scores(
        self, cart_pov_ids: List[int], pov_matches: List[Tuple[int, str, List[str]]]
    ) -> Dict[int, float]:
        """
        Get EASE scores for POV IDs based on cart context.

        Returns:
            {pov_id: ease_score}
        """
        if not cart_pov_ids:
            # No cart context, return minimal scores
            return {pov_id: 0.001 for pov_id, _, _ in pov_matches}

        try:
            # Get EASE recommendations
            ease_recommendations = self._ease_scorer.recommend(cart_pov_ids, top_k=80)

            # Convert to dict
            ease_scores = {pov_id: score for pov_id, score in ease_recommendations}

            # Add minimal scores for matched items not in EASE results
            matched_pov_ids = {pov_id for pov_id, _, _ in pov_matches}
            for pov_id in matched_pov_ids:
                if pov_id not in ease_scores:
                    ease_scores[pov_id] = 0.001

            return ease_scores

        except Exception as e:
            print(f"Error in EASE scoring: {e}")
            return {pov_id: 0.001 for pov_id, _, _ in pov_matches}

    def _map_pov_to_vv_candidates(
        self, pov_matches: List[Tuple[int, str, List[str]]]
    ) -> List[Dict]:
        """
        Map POV IDs to VV product candidates.

        Returns:
            List of {pov_id, vv_id, ingredient_name, llm_queries}
        """
        candidates = []
        for pov_id, ingredient_name, llm_queries in pov_matches:
            # Look up in precomputed mapping
            if pov_id in self._pov_to_vv:
                for vv_id, vv_ingredient in self._pov_to_vv[pov_id]:
                    candidates.append(
                        {
                            "pov_id": pov_id,
                            "vv_id": vv_id,
                            "ingredient_name": ingredient_name,
                            "llm_queries": llm_queries,
                        }
                    )
                ic(candidates)
            else:
                # Secondary search in VV catalog using ingredient name
                vv_results = self._vv_searcher.search(ingredient_name, limit=10)
                ic(vv_results)
                if not vv_results.empty:
                    for _, row in vv_results.iterrows():
                        vv_id = int(row["id"])
                        candidates.append(
                            {
                                "pov_id": pov_id,
                                "vv_id": vv_id,
                                "ingredient_name": ingredient_name,
                                "llm_queries": llm_queries,
                            }
                        )

        return candidates

    def _rerank_candidates(
        self,
        candidates: List[Dict],
        ease_scores: Dict[int, float],
        llm_outputs: List[str],
    ) -> List[Dict]:
        """
        Rerank VV candidates using EASE scores + product metadata.

        Returns:
            Sorted list of candidates with rerank_score
        """
        enriched = []
        ic(len(candidates))
        for candidate in candidates:
            pov_id = candidate["pov_id"]
            vv_id = candidate["vv_id"]

            # Get EASE score
            ease_score = ease_scores.get(pov_id, 0.001)

            # Get product metadata
            meta = self._vv_metadata.get(vv_id, {"price": 0.0, "rating": 0.0})
            price = meta["price"]
            rating = meta["rating"]

            enriched.append(
                {
                    **candidate,
                    "ease_score": ease_score,
                    "price": price,
                    "rating": rating,
                }
            )

        # Normalize scores
        if enriched:
            ease_values = [c["ease_score"] for c in enriched]
            price_values = [c["price"] for c in enriched]
            rating_values = [c["rating"] for c in enriched]

            ease_norm = self._min_max_normalize(ease_values)
            price_norm = self._min_max_normalize(price_values)
            rating_norm = [min(r / 5.0, 1.0) for r in rating_values]

            for i, candidate in enumerate(enriched):
                candidate["ease_norm"] = ease_norm[i]
                candidate["price_norm"] = price_norm[i]
                candidate["rating_norm"] = rating_norm[i]

                # Weighted score
                candidate["rerank_score"] = (
                    self.DEFAULT_WEIGHTS["ease"] * ease_norm[i]
                    + self.DEFAULT_WEIGHTS["rating"] * rating_norm[i]
                    + self.DEFAULT_WEIGHTS["price"] * price_norm[i]
                )

        # Sort by rerank_score descending
        enriched.sort(key=lambda x: x["rerank_score"], reverse=True)
        # len()
        return enriched

    def _diversify_and_select(self, reranked: List[Dict], top_k: int) -> List[int]:
        """
        Diversify results by ingredient and select top-K VV IDs.

        Returns:
            List of VV product IDs
        """
        # Group by base ingredient (pov_id)
        ic(reranked)
        ic(len(reranked))
        grouped = {}
        for candidate in reranked:
            pov_id = candidate["pov_id"]
            if pov_id not in grouped:
                grouped[pov_id] = []
            grouped[pov_id].append(candidate)

        # Round-robin selection
        result = []
        while grouped and len(result) < top_k:
            empty_keys = []
            for pov_id, items in list(grouped.items()):
                if not items:
                    empty_keys.append(pov_id)
                    continue

                item = items.pop(0)
                vv_id = item["vv_id"]

                # Avoid duplicates
                if vv_id not in result:
                    result.append(vv_id)

                if len(result) == top_k:
                    break

            for key in empty_keys:
                grouped.pop(key, None)
        ic(result)
        ic(len(result))
        return result

    def _fill_with_popular(self, current_vv_ids: List[int], top_k: int) -> List[int]:
        """
        Fill remaining slots with top popular products.

        Returns:
            Extended list of VV IDs
        """
        result = current_vv_ids.copy()
        used = set(current_vv_ids)

        for pop_vv_id in self.TOP_POPULAR_VV_IDS:
            if len(result) >= top_k:
                break
            if pop_vv_id not in used:
                result.append(pop_vv_id)
                used.add(pop_vv_id)

        return result

    @staticmethod
    def _min_max_normalize(values: List[float]) -> List[float]:
        """Min-max normalization to [0, 1]."""
        if not values:
            return []
        min_val = min(values)
        max_val = max(values)
        if math.isclose(min_val, max_val):
            return [1.0] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]
