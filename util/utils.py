from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        row = self.data[idx]
        x = row[0: 3]
        label = row[3: 6]

        return x, label

    def __len__(self):
        return len(self.data)


