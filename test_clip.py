# https://huggingface.co/docs/transformers/model_doc/clip
import requests, PIL
import torch
import transformers as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_url = 'openai/clip-vit-base-patch32'
preprocess = T.CLIPProcessor.from_pretrained(model_url)
model = T.CLIPModel.from_pretrained(model_url).to(device)

image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = PIL.Image.open(requests.get(image_url, stream=True).raw)
text = [
    'a photo of three cats',
    'a photo of two cats',
    'a photo of a cat',
    'a photo of a dog',
]
inputs = preprocess(
    text=text,
    images=image,
    return_tensors='pt',
    padding=True
).to(device)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu()

for t, p in zip(text, probs):
   print(t, p.item())
