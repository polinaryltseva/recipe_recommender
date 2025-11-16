from recsys.registry import get_recommender_for_user

def test_get_recommender():
    rec = get_recommender_for_user(1)
    assert rec is not None
