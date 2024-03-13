import torch
import datasets

data_urls = {
    'VQA-RAD': 'flaviagiammarino/vqa-rad',
    'SLAKE': 'BoKelvin/SLAKE',
    'PMC-VQA': 'xmcmic/PMC-VQA'
}

def load_dataset(data_name, device):

    dataset = datasets.load_dataset(data_urls[data_name])
    dataset = dataset.with_format('torch', device=device)

    if data_name == 'VQA-RAD':
        train_set = dataset['train']
        test_set = dataset['test']
        train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1])
        # features: ['image', 'question', 'answer']

    elif data_name == 'SLAKE':
        train_set = dataset['train']
        val_set = dataset['validation']
        test_set = dataset['test']
        # features: ['img_name', 'question', 'answer', ...]

    elif data_name == 'PMC-VQA':
        train_set = dataset['train']
        test_set = dataset['test']
        train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1])
        # features: ['Figure_path', 'Question', 'Answer', ...]

    # TODO: the datasets have different columns and preprocessing steps,
    # so it would be better to implement separate torch Datasets for each

    return train_set, val_set, test_set


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, images, questions, answers):
        self.images = images
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.questions[idx], self.answers[idx]
