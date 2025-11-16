from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def gen_from(path):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
    t = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    prompt = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = t(prompt, return_tensors="pt").to(m.device)
    out = m.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=t.eos_token_id)
    resp = t.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return resp

print("OUTPUT_DIR:", gen_from(output_dir))
print("MERGED:", gen_from(merged_model_path))
