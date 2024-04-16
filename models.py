import torch, requests, PIL
import transformers as T
import open_clip


class ImageEncoder(torch.nn.Module):
    
    @classmethod
    def from_name(cls, name, **kwargs):
        if name == 'CLIP':
            url = 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K' # (256, 1024)
        elif name == 'PMC-CLIP':
            url = 'ryanyip7777/pmc_vit_l_14'

        model, train_preprocess, val_preprocess = \
            open_clip.create_model_and_transforms(f'hf-hub:{url}', device=torch.device('cuda'))
    
        return cls(model.visual, train_preprocess, val_preprocess, n_patches=256, embed_size=1024, **kwargs)

    def __init__(self, vit, train_preprocess, val_preprocess, n_patches, embed_size, embed_type):
        super().__init__()

        self.train_preprocess = train_preprocess
        self.val_preprocess = val_preprocess

        self.vit = vit
        self.vit.output_tokens = True
        self.vit.proj = None
    
        self.embed_type = embed_type
        self.embed_size = embed_size
        
        if embed_type == 'both':
            self.seq_length = n_patches + 1
        elif embed_type == 'patch':
            self.seq_length = n_patches
        elif embed_type == 'global':
            self.seq_length = 1
        
    def forward(self, images):
        global_embeddings, patch_embeddings = self.vit(images)
        global_embeddings = global_embeddings.unsqueeze(1)
        
        if self.embed_type == 'global':
            image_embeddings = global_embeddings
        elif self.embed_type == 'patch':
            image_embeddings = patch_embeddings  
        elif self.embed_type == 'both':
            image_embeddings = torch.cat([global_embeddings, patch_embeddings], dim=1)
        
        assert image_embeddings.shape[1:] == (self.seq_length, self.embed_size), image_embeddings.shape
        return image_embeddings
        

class TextDecoder(torch.nn.Module):
    
    @classmethod
    def from_name(cls, name):
        if name == 'LLaMA':
            url = 'meta-llama/Llama-2-7b-hf'
        elif name == 'PMC-LLaMA':
            url = 'chaoyi-wu/PMC_LLAMA_7B'
        elif name == ' LLaVA':
            url = 'liuhaotian/llava-v1.6-vicuna-7b'
        
        llm = T.LlamaForCausalLM.from_pretrained(url, device_map='auto', cache_dir='models')
        tokenizer = T.LlamaTokenizer.from_pretrained(url, cache_dir='models')
        
        return cls(llm, tokenizer, max_length=512, embed_size=4096)

    def __init__(self, llm, tokenizer, max_length, embed_size):
        super().__init__()

        self.llm = llm
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.embed_size = embed_size
        
    def forward(self, input_embeddings, mask, labels):
        assert input_embeddings.shape[1] <= self.max_length, input_embeddings.shape
        assert input_embeddings.shape[2] == self.embed_size, input_embeddings.shape

        return self.llm.forward(
            inputs_embeds=input_embeddings,
            attention_mask=mask,
            labels=labels
        )

    def generate(self, input_embeddings, mask, **kwargs):
        assert input_embeddings.shape[1] <= self.max_length, input_embeddings.shape
        assert input_embeddings.shape[2] == self.embed_size, input_embeddings.shape

        return self.llm.generate(
            inputs_embeds=input_embeddings,
            attention_mask=mask,
            **kwargs
        )


class MultimodalFusion(torch.nn.Module):
    
    def __init__(self, image_embed_size, text_embed_size, device):
        super().__init__()
        self.image_embed_size = image_embed_size
        self.text_embed_size = text_embed_size

        self.project_image = torch.nn.Linear(
            image_embed_size, text_embed_size, device=device
        )
        
    def forward(self, image_embeddings, text_embeddings, mask):
        batch_size, image_length = image_embeddings.shape[:2]

        image_embeddings = self.project_image(image_embeddings)
        combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)

        image_mask = torch.ones((batch_size, image_length), device=mask.device)
        combined_mask = torch.cat([image_mask, mask], dim=1)

        return combined_embeddings, combined_mask


class VQAModel(torch.nn.Module):
    
    def __init__(self, image_encoder, text_encoder, text_decoder):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder  = text_encoder
        self.text_decoder  = text_decoder

        self.fusion_module = MultimodalFusion(
            image_encoder.embed_size,
            text_decoder.embed_size,
            device=text_decoder.llm.device
        )
        
    def combine_multimodal_inputs(self, images, padded_tokens, mask):   
        image_embeddings = self.image_encoder(images)
        text_embeddings  = self.text_encoder(padded_tokens)

        combined_embeddings, combined_mask = self.fusion_module(
            image_embeddings, text_embeddings, mask
        )
        return combined_embeddings, combined_mask
    
    def forward(self, images, padded_tokens, mask):       
        input_embeddings, mask = self.combine_multimodal_inputs(
            images, padded_tokens, mask
        )
        dummy_tokens = torch.zeros(
            (images.shape[0], self.image_encoder.seq_length),
            dtype=padded_tokens.dtype,
            device=padded_tokens.device
        )
        labels = torch.cat([dummy_tokens, padded_tokens], dim=1)
        
        output = self.text_decoder.forward(input_embeddings, mask, labels)
        return output

    def generate(self, images, padded_tokens, mask, **kwargs):
        input_embeddings, mask = self.combine_multimodal_inputs(
            images, padded_tokens, mask
        )
        output = self.text_decoder.generate(input_embeddings, mask, **kwargs)
        return output
