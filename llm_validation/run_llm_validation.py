import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine

from validate import RecommendationValidator
from llm_validator import create_llm_predictor_with_auto_user_id
import pandas as pd


def main():
    validator = RecommendationValidator(
        data_path='../recipes_normalized.csv', 
        k_core=5,
        test_size=0.2,
        val_size=0.15,
        seed=42,
        verbose=True
    )
    
    # Ограничиваем размер тестового сета для быстрой оценки случайной выборкой
    MAX_TEST_USERS = 6000
    test_split_df = validator.get_grouped_split('test')
    if len(test_split_df) > MAX_TEST_USERS:
        test_split_df = test_split_df.sample(n=MAX_TEST_USERS, random_state=42).reset_index(drop=True)
        test_mask = validator.full_grouped_data['split'] == 'test'
        validator.full_grouped_data = pd.concat([
            validator.full_grouped_data[~test_mask],
            test_split_df
        ]).reset_index(drop=True)
    
    model_path = "Qwen/Qwen3-14b"
    
    engine_args = EngineArgs(
        model=model_path,
        gpu_memory_utilization=0.9,  
        max_model_len=2048,
        trust_remote_code=True
    )
    engine = LLMEngine.from_engine_args(engine_args)
    tokenizer = engine.tokenizer

    llm_predictor = create_llm_predictor_with_auto_user_id(
        engine=engine,
        tokenizer=tokenizer,
        validator=validator,
        split='test',
        context_window=5,
        matching_threshold=0.6,
        verbose=True
    )
    
    result = validator.evaluate_split(
        split='test',
        predictor=llm_predictor,
        preds_col='llm_preds',
        top_k=5,
        show_progress=True
    )
    
    for metric, value in result['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    import json
    results_file = 'llm_validation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

