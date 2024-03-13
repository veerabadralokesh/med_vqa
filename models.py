import torch
import clip

TODO = NotImplementedError('TODO')


class VQAModel(torch.nn.Module):

    def __init__(self, image_encoder, text_encoder, text_decoder, device):
        if image_encoder == 'CLIP':
            clip_model = clip.load('ViT-B/32', device)
            raise TODO
        else:
            raise TODO

        if text_encoder == 'LLaMA':
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
