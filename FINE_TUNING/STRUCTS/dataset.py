import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyth_imports import *
from utils import *
from tempfile import mkdtemp

class AnalogiesData(Dataset):
    def __init__(self, filename_analogies, tokenizer = None):
        self.targets, self.data = load_tokenized_data(filename_analogies)
        self.targets = torch.tensor(self.targets, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        targets_ = self.targets[index].clone().detach()
        return {'data': self.data[index], 'targets': targets_}