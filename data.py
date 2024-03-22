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

def load_dataset(data_name, device='cuda', cache_dir='data'):
    '''
    Args:
        data_name: 'VQA-RAD', 'SLAKE', or 'PMC-VQA'
        device: default is 'cuda'
        cache_dir: Data download directory (default: 'data')
    Returns:
        train_set, val_set, test_set: VQADataset instances
    '''
    dataset = datasets.load_dataset(data_urls[data_name], cache_dir=cache_dir)

    if data_name == 'VQA-RAD':
        train_set = VQARADDataset(dataset['train'])
        test_set = VQARADDataset(dataset['test'])
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
        train_set = SLAKEDataset(dataset['train'], image_root)
        val_set = SLAKEDataset(dataset['validation'], image_root)
        test_set = SLAKEDataset(dataset['test'], image_root)

    elif data_name == 'PMC-VQA':
        train_set = PMCVQADataset(dataset['train'])
        test_set = PMCVQADataset(dataset['test'])
        train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1])
        # features: ['Figure_path', 'Question', 'Answer', ...]

    return train_set, val_set, test_set


class VQADataset(torch.utils.data.Dataset):
    '''
    Args:
        images: list of PIL.Image
        questions: list of str
        answers: list of str
    '''
    def __init__(self, images, questions, answers):
        self.images = images
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.images[idx], self.questions[idx], self.answers[idx]


class VQARADDataset(VQADataset):

    def __init__(self, data):
        images = data['image']
        questions = data['question']
        answers = data['answer']
        super().__init__(images, questions, answers)


class SLAKEDataset(VQADataset):

    def __init__(self, data, image_root):
        self.image_names = data['img_name'] # e.g. 'xmlab99/source.jpg'
        self.questions = data['question']
        self.answers = data['answer']

        self.image_root = pathlib.Path(image_root)
        self.images = {}
        for image_name in self.image_names:
            if image_name not in self.images:
                image_file = self.image_root / image_name
                with Image.open(image_file) as image:
                    image.load()
                self.images[image_name] = image

    def __getitem__(self, idx):
        image = self.images[self.image_names[idx]]
        return image, self.questions[idx], self.answers[idx]


class PMCVQADataset(VQADataset):

    def __init__(self, data):
        raise TODO
