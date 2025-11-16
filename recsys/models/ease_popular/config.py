from pathlib import Path

# Папка с ЭТИМ файлом (recsys/models/ease_popular)
BASE_DIR = Path(__file__).resolve().parent

# Папка с данными модели (models/ease_popular)
DATA_DIR = Path(__file__).resolve().parents[3] / "models" / "ease_popular"
# объяснение:
# __file__ -> recsys/models/ease_popular/config.py
# parents[3] -> корень проекта (RecSys_case_final_arc)

EASEMODEL = DATA_DIR / "ease_model.pkl"
MAPPING = DATA_DIR / "items_dict.json"
POV2VV = DATA_DIR / "items_dict_pov2vv_str.json"
VV2POV = DATA_DIR / "items_dict_vv2pov_str.json"

# пока не используем отдельный pickle-топпоп, поэтому просто оставим None
TOPPOPULAR = None
