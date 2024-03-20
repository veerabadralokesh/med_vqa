import torch
import clip

TODO = NotImplementedError('TODO')


class VQAModel(torch.nn.Module):

    def __init__(self, image_encoder, text_encoder, text_decoder, device):
        if image_encoder == 'CLIP':
            clip_model, preprocess = clip.load('ViT-B/32', device)
            self.image_preprocess = preprocess
            self.image_model = clip_model
            self.image_encoder = lambda image: self.image_model.encode_image(self.image_preprocess(image).unsqueeze(0).to(device))
            self.clip_text_encoder = clip_model.encode_text
            raise TODO
        else:
            raise TODO

        if text_encoder == 'LLaMA':
            raise TODO
        elif text_encoder == 'CLIP' and image_encoder == 'CLIP':
            self.text_encoder = self.clip_text_encoder
        else:
            raise TODO

        # self.fusion_module = 

        if text_decoder is not None:
            raise TODO
        else:
            self.text_decoder = torch.nn.Identity

        raise TODO

    def forward(self, images, questions):
        image_feats = self.image_encoder(images)
        text_feats  = self.text_encoder(questions)
        fused_feats = self.fusion_module(image_feats, text_feats)
        answer_logits = self.text_decoder(fused_feats)
        return answer_logits
