import sys, os, argparse, time, tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import datasets, peft, evaluate
bleu_score_metric = evaluate.load('bleu')
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

import data, models


class Timer(object):

    def __init__(self):
        self.t_start = time.time()

    def tick(self):
        t_stop = time.time()
        t_delta = t_stop - self.t_start
        print(f'Done ({t_delta:.4f} s)', flush=True)
        self.t_start = t_stop
        return t_delta


def compute_metrics(input_tokens, output_tokens, answer_mask, tokenizer):
    batch_size = input_tokens.shape[0]

    m = defaultdict(list)
    true_answers = []
    pred_answers = []

    for i in range(batch_size):
        true_answer_tokens = input_tokens[i,1:][answer_mask[i,1:].bool()]
        pred_answer_tokens = output_tokens[i,:-1][answer_mask[i,1:].bool()]
        
        true_answer = tokenizer.decode(true_answer_tokens)
        pred_answer = tokenizer.decode(pred_answer_tokens)
        if not pred_answer.strip():
            pred_answer = '_'

        similarity = SequenceMatcher(a=true_answer, b=pred_answer).ratio()
        
        m['exact_match'].append(int(true_answer == pred_answer))
        m['similarity'].append(similarity)
        
        true_answers.append(true_answer)
        pred_answers.append(pred_answer)

    m = {k: np.mean(v) for k, v in m.items()}
    
    try:
        bleu_result = bleu_score_metric.compute(
            predictions=pred_answers, references=true_answers
        )
    except ZeroDivisionError:
        print((true_answers, pred_answers))
        raise
    m['bleu_score'] = bleu_result['bleu']
    m['precision1'] = bleu_result['precisions'][0]
    m['precision2'] = bleu_result['precisions'][1]
    m['precision3'] = bleu_result['precisions'][2]
    m['precision4'] = bleu_result['precisions'][3]
    return m


def count_parameters(model):
    n_params = 0
    for k, v in model.named_parameters():
        if v.requires_grad:
            n_params += np.prod(v.shape)
    return n_params


def save_model_state(model, save_path):
    text_decoder_state_dict = {
        k: v.to('cpu')
            for k, v in model.text_decoder.llm.named_parameters() 
            if v.requires_grad
    }
    torch.save({
        'fusion_module': model.fusion_module.state_dict(),
        'text_decoder': text_decoder_state_dict,
    }, save_path)


def load_model_state(model, save_path):
    checkpoint = torch.load(save_path)
    model.fusion_module.load_state_dict(checkpoint['fusion_module'])
    model.text_decoder.llm.load_state_dict(checkpoint['text_decoder'], strict=False)


def training_plot(metrics):
    fig, ax = plt.subplots(1, 3, figsize=(9,4))

    ax[0].set_ylabel('loss')
    ax[1].set_ylabel('similarity')
    ax[2].set_ylabel('exact_match')

    metrics = metrics.reset_index().groupby(['phase', 'epoch']).mean()
    train_metrics = metrics.loc['train']
    val_metrics = metrics.loc['val']

    for phase in ['train', 'val']:
        m = metrics.loc[phase]
        ax[0].plot(m.index, m.loss, label=phase)
        ax[1].plot(m.index, m.similarity, label=phase)
        ax[2].plot(m.index, m.exact_match, label=phase)
    
    for ax_i in ax:
        ax_i.set_axisbelow(True)
        ax_i.grid(linestyle=':')
        ax_i.set_xlabel('epoch')
        ax_i.legend(frameon=False)

    fig.tight_layout()
    return fig


def train(
    dataset: str,
    image_encoder: str,
    text_encoder: str,
    text_decoder: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    lora_rank: int
):
    timer = Timer()

    assert torch.cuda.is_available(), 'No GPU devices available'

    print('Available GPU devices:')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_properties(i).name}')

    print(f'Loading {image_encoder} image encoder...', end='', flush=True)
    image_encoder = models.ImageEncoder.from_name(image_encoder, embed_type='both')
    image_length = image_encoder.seq_length # number of image tokens
    timer.tick()

    print(f'Loading {text_decoder} text decoder...')
    text_decoder = models.TextDecoder.from_name(text_decoder)
    text_encoder = text_decoder.llm.model.embed_tokens
    timer.tick()

    print(f'Creating VQA model...', end='', flush=True)
    model = models.VQAModel(image_encoder, text_encoder, text_decoder)
    timer.tick()

    print(f'Loading {dataset} dataset...', end='', flush=True)
    train_set, val_set, test_set = data.VQADataset.from_name(
        dataset,
        train_preprocess=image_encoder.train_preprocess,
        val_preprocess=image_encoder.val_preprocess,
        tokenizer=text_decoder.tokenizer,
        image_length=image_encoder.seq_length,
        max_length=text_decoder.max_length,
        device='cuda'
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True)
    timer.tick()

    print(f'Testing forward pass...', flush=True)
    images, input_tokens, prompt_mask, answer_mask = next(iter(train_loader))
    output = model.forward(images, input_tokens, prompt_mask)
    compute_metrics(input_tokens, output_tokens, answer_masks, text_decoder.tokenizer)
    timer.tick()

    print(f'Trainable parameters:', end='', flush=True)
    print(f'  before LORA: {count_parameters(model):e} params')
    peft_config = peft.LoraConfig(
        r=lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', \
            'gate_proj', 'up_proj', 'down_proj', 'lm_head'
        ],
        bias='none',
        task_type='CAUSAL_LM'
    )
    model.text_decoder.llm = peft.get_peft_model(model.text_decoder.llm, peft_config)
    print(f'  after LORA: {count_parameters(model):e} params')

    learnable_params = torch.nn.ParameterList()
    learnable_params.extend(model.fusion_module.parameters())
    learnable_params.extend(model.text_decoder.llm.parameters())
    optimizer = torch.optim.AdamW(learnable_params, lr=learning_rate)
    timer.tick()

    metrics_df = pd.DataFrame(columns=['epoch', 'phase', 'step'])
    metrics_df.set_index(['epoch', 'phase', 'step'], inplace=True)

    print('Start training loop')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch} / {num_epochs}')

        epoch_train_loss = 0
        epoch_val_loss = 0

        print('Training...')
        model.train()
        for step, batch in enumerate(pbar:=tqdm.tqdm(train_loader)):
            images, input_tokens, prompt_mask, answer_mask = batch
            output = model(images, input_tokens, prompt_mask)

            loss = output.loss.detach().cpu().item()
            output_tokens = torch.argmax(output.logits, dim=-1)[:,image_length:]
            m = compute_metrics(
                input_tokens, output_tokens, answer_mask, text_decoder.tokenizer
            )
            metrics_df.loc[(epoch, 'train', step), 'loss'] = loss
            metrics_df.loc[(epoch, 'train', step), 'exact_match'] = m['exact_match']
            metrics_df.loc[(epoch, 'train', step), 'similarity'] = m['similarity']
            metrics_df.loc[(epoch, 'train', step), 'bleu_score'] = m['bleu_score']
            metrics_df.loc[(epoch, 'train', step), 'precision1'] = m['precision1']
            metrics_df.loc[(epoch, 'train', step), 'precision2'] = m['precision2']
            metrics_df.loc[(epoch, 'train', step), 'precision3'] = m['precision3']
            metrics_df.loc[(epoch, 'train', step), 'precision4'] = m['precision4']

            epoch_train_loss += loss
            pbar.set_description(f'train loss = {loss}')
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_train_loss /= len(train_loader)
        print(f'Epoch {epoch} train loss = {epoch_train_loss}')

        print('Evaluating...')
        model.eval()
        for step, batch in enumerate(pbar:=tqdm.tqdm(val_loader)):
            images, input_tokens, prompt_mask, answer_mask = batch
            output = model(images, input_tokens, prompt_mask)
            
            loss = output.loss.detach().cpu().item()
            output_tokens = torch.argmax(output.logits, dim=-1)[:,image_length:]
            m = compute_metrics(
                input_tokens, output_tokens, answer_mask, text_decoder.tokenizer
            )
            metrics_df.loc[(epoch, 'val', step), 'loss'] = loss
            metrics_df.loc[(epoch, 'val', step), 'exact_match'] = m['exact_match']
            metrics_df.loc[(epoch, 'val', step), 'similarity'] = m['similarity']
            metrics_df.loc[(epoch, 'val', step), 'bleu_score'] = m['bleu_score']
            metrics_df.loc[(epoch, 'val', step), 'precision1'] = m['precision1']
            metrics_df.loc[(epoch, 'val', step), 'precision2'] = m['precision2']
            metrics_df.loc[(epoch, 'val', step), 'precision3'] = m['precision3']
            metrics_df.loc[(epoch, 'val', step), 'precision4'] = m['precision4']

            epoch_val_loss += loss
            pbar.set_description(f'val loss = {loss}')

        epoch_val_loss /= len(val_loader)
        print(f'Epoch {epoch} val loss = {epoch_val_loss}')

        print('Saving model...')
        save_model_state(model, f'{out_name}_epoch_{epoch}.pt')
        training_plot(metrics_df).savefig(
            f'{out_name}_training.png', bbox_inches='tight'
        )
        metrics_df.to_csv(f'{out_name}_metrics.csv')

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='VQA-RAD')
    parser.add_argument('--image_encoder', type=str, default='CLIP')
    parser.add_argument('--text_encoder', type=str, default=None)
    parser.add_argument('--text_decoder', type=str, default='LLaMA')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lora_rank', type=int, default=8)
    kwargs = vars(parser.parse_args())
    train(**kwargs)
