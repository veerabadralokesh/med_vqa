# https://huggingface.co/docs/transformers/model_doc/llama
import sys, os; print(os.environ['HF_TOKEN']) 
import torch
import transformers as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_url = 'meta-llama/Llama-2-7b-hf'
tokenizer = T.AutoTokenizer.from_pretrained(model_url)
model = T.LlamaForCausalLM.from_pretrained(model_url).to(device)

prompt, max_length = sys.argv[1:3]
max_length = int(max_length)
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

generate_ids = model.generate(inputs, max_length=max_length)
output = tokenizer.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]
print(output)
