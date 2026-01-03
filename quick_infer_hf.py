from transformers import AutoTokenizer, AutoModelForCausalLM

path = "hf_export"
tok = AutoTokenizer.from_pretrained(path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(path)  # CPU by default
model.eval()

prompt = "BIANCA:"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
