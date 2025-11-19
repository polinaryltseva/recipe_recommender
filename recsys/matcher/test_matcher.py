"""
Simple test script for the optimized LLM matcher.
"""

from llm_matcher_optimized import LLMMatcher


def test_basic_matching():
    """Test basic matching functionality."""
    print("=" * 60)
    print("Testing LLM Matcher")
    print("=" * 60)

    # Initialize matcher (resources loaded once)
    print("\n[1] Initializing matcher...")
    matcher = LLMMatcher()
    print("[OK] Matcher initialized")

    # Test case 1: Simple ingredients
    print("\n[2] Test case 1: Simple ingredients")
    llm_outputs = ["Масло сливочное", "Ванильный сахар", "Вода"]
    current_cart = [15, 66, 24]  # Internal POV IDs

    print(f"   LLM outputs: {llm_outputs}")
    print(f"   Current cart (POV IDs): {current_cart}")

    result = matcher.match_llm_output(llm_outputs, current_cart, top_k=10)

    print(f"   Result (VV IDs): {result}")
    print(f"   Result length: {len(result)}")

    # Test case 2: Empty cart
    print("\n[3] Test case 2: Empty cart")
    llm_outputs = ["Молоко", "Яйцо куриное", "Сахар"]
    current_cart = []

    print(f"   LLM outputs: {llm_outputs}")
    print(f"   Current cart: {current_cart}")

    result = matcher.match_llm_output(llm_outputs, current_cart, top_k=10)

    print(f"   Result (VV IDs): {result}")
    print(f"   Result length: {len(result)}")

    # Test case 3: Unknown ingredients (fallback test)
    print("\n[4] Test case 3: Unknown ingredients")
    llm_outputs = ["Неизвестный продукт XYZ", "Абракадабра"]
    current_cart = [0, 1, 2]

    print(f"   LLM outputs: {llm_outputs}")
    print(f"   Current cart: {current_cart}")

    result = matcher.match_llm_output(llm_outputs, current_cart, top_k=6)

    print(f"   Result (VV IDs): {result}")
    print(f"   Result length: {len(result)}")
    print(f"   Expected fallback: {LLMMatcher.TOP_POPULAR_VV_IDS[:6]}")

    # Test case 4: Many ingredients
    print("\n[5] Test case 4: Many ingredients")
    llm_outputs = [
        "Молоко",
        "Масло сливочное",
        "Мука",
        "Яйцо",
        "Сахар",
        "Соль",
        "Вода",
    ]
    current_cart = [0, 15]

    print(f"   LLM outputs: {llm_outputs}")
    print(f"   Current cart: {current_cart}")

    result = matcher.match_llm_output(llm_outputs, current_cart, top_k=20)

    print(f"   Result (VV IDs): {result}")
    print(f"   Result length: {len(result)}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


def test_singleton_pattern():
    """Test that singleton pattern works (resources loaded only once)."""
    print("\n[6] Testing singleton pattern...")

    matcher1 = LLMMatcher()
    matcher2 = LLMMatcher()

    # Both should share the same class-level resources
    assert matcher1._pov_searcher is matcher2._pov_searcher
    assert matcher1._vv_searcher is matcher2._vv_searcher
    assert matcher1._pov_to_vv is matcher2._pov_to_vv
    assert matcher1._vv_metadata is matcher2._vv_metadata

    print("[OK] Singleton pattern verified: Resources shared between instances")


if __name__ == "__main__":
    test_basic_matching()
    test_singleton_pattern()
