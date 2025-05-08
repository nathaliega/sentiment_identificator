from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['review']
        label = row['rating']
        return label, text

    def __len__(self):
        return len(self.dataframe)
