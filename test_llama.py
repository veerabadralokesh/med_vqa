# https://huggingface.co/docs/transformers/model_doc/llama
import os; print(os.environ['HF_TOKEN']) 
import torch
import transformers as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_url = 'meta-llama/Llama-2-7b-hf'
tokenizer = T.AutoTokenizer.from_pretrained(model_url)
model = T.LlamaForCausalLM.from_pretrained(model_url).to(device)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

generate_ids = model.generate(inputs, max_length=30)
output = tokenizer.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]
print(output)
