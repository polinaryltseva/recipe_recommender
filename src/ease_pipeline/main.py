from main_ease import EASERecommendationSystem
from top_popular import TopPopular  # noqa: F401 - needed for pickle

system = EASERecommendationSystem()


user_activity = ...  # Список List["str"]

user_activity = [446, 4188, 10124]
recommendations = system.get_recommendations(user_activity, top_k=10)
print(recommendations)
