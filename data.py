import pathlib
import gdown
import zipfile
from PIL import Image
import torch
import datasets

data_urls = {
    'VQA-RAD': 'flaviagiammarino/vqa-rad',
    'SLAKE': 'BoKelvin/SLAKE',
    'PMC-VQA': 'xmcmic/PMC-VQA'
}

def load_dataset(
    data_name,
    preprocess,
    tokeinizer,
    device='cuda',
    cache_dir='data'
):
    '''
    Args:
        data_name: 'VQA-RAD', 'SLAKE', or 'PMC-VQA'
        preprocess: Image preprocessing function
        tokenizer: Text tokenization function
        device: default is 'cuda'
        cache_dir: Data download directory (default: 'data')
    Returns:
        train_set, val_set, test_set: VQADataset instances
    '''
    dataset = datasets.load_dataset(data_urls[data_name], cache_dir=cache_dir)

    if data_name == 'VQA-RAD':
        train_set = VQARADDataset(dataset['train'], preprocess, tokenizer)
        test_set = VQARADDataset(dataset['test'], preprocess, tokenizer)
        train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1])

    elif data_name == 'SLAKE':

        data_root = pathlib.Path(cache_dir) / 'BoKelvin___slake'
        image_root = data_root / 'imgs'

        if not image_root.is_dir(): # need to download and unzip the images
            image_url = 'https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip?download=true'
            image_zip = data_root / 'imgs.zip'
            gdown.download(image_url, output=str(image_zip), quiet=False)
            with zipfile.ZipFile(image_zip, 'r') as f:
                f.extractall(path=data_root)

        dataset = dataset.filter(lambda x: x['q_lang'] == 'en')
        train_set = SLAKEDataset(dataset['train'], image_root, preprocess, tokenizer)
        val_set = SLAKEDataset(dataset['validation'], image_root, preprocess, tokenizer)
        test_set = SLAKEDataset(dataset['test'], image_root, preprocess, tokenizer)

    elif data_name == 'PMC-VQA':
        train_set = PMCVQADataset(dataset['train'], preprocess, tokenizer)
        test_set = PMCVQADataset(dataset['test'], preprocess, tokenizer)
        train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1])
        # features: ['Figure_path', 'Question', 'Answer', ...]

    return train_set, val_set, test_set


class VQADataset(torch.utils.data.Dataset):
    
    @classmethod
    def from_name(cls, name, train_preprocess, val_preprocess, **kwargs):

        if name == 'VQA-RAD':
            url = 'flaviagiammarino/vqa-rad'
            val_split = 'test'
        elif name == 'SLAKE':
            url = 'BoKelvin/SLAKE'
            val_split = 'validation'

        ds = datasets.load_dataset(url, cache_dir='data')
        
        train_set = cls(ds['train'], train_preprocess, **kwargs)
        val_set = cls(ds[val_split], val_preprocess, **kwargs)
        test_set = cls(ds['test'], val_preprocess, **kwargs)

        return train_set, val_set, test_set
            
    def __init__(self, ds, image_preprocess, tokenizer, image_length, max_length, device):
        super().__init__()
        self.ds = ds
        
        # image preprocessor
        self.image_preprocess = image_preprocess
        
        # text tokenizer
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token

        assert max_length > image_length
        self.image_length = image_length
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        raw_image = self.ds[idx]['image']
        question = self.ds[idx]['question']
        answer = self.ds[idx]['answer']
        
        image = self.image_preprocess(raw_image)

        prompt = f'[INST]\nPlease answer the following question based on the provided image.\n[/INST]\nQ: {question}\nA:'
                
        prompt_tokens = self.tokenizer.encode(prompt)
        answer_tokens = self.tokenizer.encode(answer)[1:]
        
        padded_tokens, prompt_mask, answer_mask = self.pad_tokens(prompt_tokens, answer_tokens)

        return (
            torch.as_tensor(image, device=self.device),
            torch.as_tensor(padded_tokens, device=self.device),
            torch.as_tensor(prompt_mask, device=self.device),
            torch.as_tensor(answer_mask, device=self.device)
        )
    
    def pad_tokens(self, prompt_tokens, answer_tokens):
        pad = self.tokenizer.pad_token_id
        tokens = prompt_tokens + answer_tokens
        prompt_mask = [1 for i in prompt_tokens] + [0 for i in answer_tokens]
        answer_mask = [0 for i in prompt_tokens] + [1 for i in answer_tokens]
        padding = self.max_length - self.image_length - len(tokens)
        if padding > 0:
            tokens = tokens + [pad for i in range(padding)]
            prompt_mask = prompt_mask + [0 for i in range(padding)]
            answer_mask = answer_mask + [0 for i in range(padding)]
        elif padding < 0:
            tokens = tokens[:self.max_length - self.image_length]
            prompt_mask = prompt_mask[:self.max_length - self.image_length]
            answer_mask = answer_mask[:self.max_length - self.image_length]
        return tokens, prompt_mask, answer_mask
