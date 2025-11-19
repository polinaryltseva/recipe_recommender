# recsys/models/llm_recs/internal_id_names.py

import json
from pathlib import Path
from typing import List


class InternalIDs2Names:
    def __init__(self, mapping_path: str | None = None):
        if mapping_path is None:
            root_dir = Path(__file__).resolve().parents[3]
            mapping_path = root_dir / "models" / "llm_recs" / "vv_id2names.json"
            print("!!!!! ", root_dir)
        else:
            mapping_path = Path(mapping_path)

        with open(mapping_path, "r", encoding="utf-8") as f:
            id_to_name = json.load(f)

        # ключи в json — строки, приводим к int
        self.internal2vkusvill = {int(k): v for k, v in id_to_name.items()}

    def ids_to_names(self, cart_items: List[int]) -> List[str]:
        vkussvil_names = []
        for id_ in cart_items:
            if id_ in self.internal2vkusvill:
                vkussvil_names.append(self.internal2vkusvill[id_])
            else:
                # можно заменить на logger.warning
                print(f"Warning: Internal ID '{id_}' not found in mappings")
        return vkussvil_names
