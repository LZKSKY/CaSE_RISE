from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]























