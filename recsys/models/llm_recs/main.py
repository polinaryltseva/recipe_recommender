# recsys/models/llm_recs/main.py
from .vllm_server_sync import vllm_recomender
from .internal_id_names import InternalIDs2Names
# from .llm_output_mapper import LLMOutputSearchDB
from ...matcher.llm_matcher_optimized import LLMMatcher
from pathlib import Path
# import recipe_recommender.recsys as recsys_pkg
RECSYS_ROOT = Path(__file__).resolve().parents[2]

class LLMWrapped:
    def __init__(self):
        self.int_ids = InternalIDs2Names()
        self.out_ids = LLMMatcher(project_root=RECSYS_ROOT)
        self.model = vllm_recomender()

    def recommend(self, input_idxs, user_id, k=10):
        names = self.int_ids.ids_to_names(input_idxs)
        if not names:
            return []
        cart_str = "; ".join(names)
        llm_raw = self.model.get_recs(cart_str, user_id, k)
        # TODO Apply 
        recs = self.out_ids.match_llm_output(llm_raw, input_idxs, top_k=k)
        print("LLM WRAPPED RECS: ", recs)
        return recs
