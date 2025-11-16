"""
EASE Model Wrapper with Name Mapping
Provides recommendation interface that works with ingredient names
"""

import pickle
from pathlib import Path
from typing import List, Union

import numpy as np


class EASERecommenderWithNames:
    """
    EASE Recommender that can work with both item IDs and ingredient names.

    Usage:
        # Initialize
        recommender = EASERecommenderWithNames(
            model_path='../recsys_tests/ease_model.pkl',
            names_path='../recsys_tests/items_dict.json'
        )

        # Get recommendations by item IDs
        recs = recommender.recommend([0, 1, 2], top_k=10)

        # Get recommendations by ingredient names
        recs = recommender.recommend(['Молоко', 'Яйцо куриное', 'Масло сливочное'], top_k=10)

        # Get recommendations with names
        recs = recommender.recommend_with_names(['Молоко', 'Чеснок'], top_k=10)
    """

    def __init__(self, model_path: str):
        """
        Initialize the recommender with model and name mappings.

        Args:
            model_path: Path to the pickled EASE model
            names_path: Path to the JSON file with id->name mappings
        """
        self.model_path = Path(model_path)

        # Load model
        # print(f"Loading EASE model from {self.model_path}...")
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model_weights = model_data["model_weights"]
        self.item2id = model_data["item2id"]
        self.id2item = model_data["id2item"]
        self.n_items = model_data["n_items"]
        self.n_users = model_data.get("n_users", 0)

        # print(f"   Model loaded: {self.n_items} items, {self.n_users} users")
        # print(f"   Weights shape: {self.model_weights.shape}")

    def recommend(
        self,
        user_items: List[Union[int, str]],
        top_k: int = 20,
        exclude_seen: bool = True,
    ) -> List[int]:
        """
        Generate top-K item recommendations based on user's interaction history.

        Args:
            user_items: List of items (IDs or names) the user has interacted with
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude items the user has already seen

        Returns:
            List of recommended item IDs
        """
        # Convert to IDs
        # item_ids = self._convert_to_ids(user_items)
        # ic(item_ids)
        if not user_items:
            # print(
            #     "Warning: No valid items provided, returning empty recommendations, NEED TO APPLY TOPPOP"
            # )
            return []

        # Create user vector
        user_vector = np.zeros(self.n_items, dtype=np.float32)
        user_vector[user_items] = 1.0

        # Compute scores
        scores = user_vector @ self.model_weights

        # Exclude already seen items
        if exclude_seen:
            scores[user_items] = -np.inf

        # Get top-K items
        top_indices = np.argsort(-scores)[:top_k]

        return [int(idx) for idx in top_indices]

    def batch_recommend(
        self,
        user_items_list: List[List[Union[int, str]]],
        top_k: int = 20,
        exclude_seen: bool = True,
    ) -> List[List[int]]:
        """
        Generate recommendations for multiple users at once.

        Args:
            user_items_list: List of interaction lists for each user
            top_k: Number of recommendations per user
            exclude_seen: Whether to exclude already seen items

        Returns:
            List of recommendation lists (one per user)
        """
        return [self.recommend(items, top_k, exclude_seen) for items in user_items_list]
