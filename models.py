import torch, requests, PIL
import transformers as T

model_urls = {
    'CLIP': ...,
    'LLaMa': ...,
}


TODO = NotImplementedError('TODO')


class VQAModel(torch.nn.Module):

    def __init__(self, image_encoder, text_encoder, text_decoder, device):
        if image_encoder == 'CLIP':
            # https://huggingface.co/docs/transformers/model_doc/clip
            preprocess = T.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = T.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = PIL.Image.open(requests.get(url, stream=True).raw)
            inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            raise TODO
        else:
            raise TODO

        if text_encoder == 'LLaMA':
            # https://huggingface.co/docs/transformers/main/en/model_doc/llama
            tokenizer = T.LlamaTokenizer.from_pretrained("meta-llama/llama-2-7b-hf")
            model = T.LlamaForCausalLM.from_pretrained("meta-llama/llama-2-7b-hf")
            raise TODO
        else:
            raise TODO

        raise TODO

    def forward(self, images, questions):
        image_feats = self.image_encoder(images)
        text_feats  = self.text_encoder(questions)
        fused_feats = self.fusion_module(image_feats, text_feats)
        answer_logits = self.text_decoder(fused_feats)
        return answer_logits
