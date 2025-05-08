from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    """Custom PyTorch Dataset for loading text reviews and their associated labels.

    Args:
        dataframe (pandas.DataFrame): A DataFrame containing at least two columns:
            - 'review': The text of the review.
            - 'rating': The corresponding label or rating.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, idx):
        """Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Any, str]: A tuple containing:
                - label: The rating associated with the review.
                - text: The review text.
        """
        row = self.dataframe.iloc[idx]
        text = row['review']
        label = row['rating']
        return label, text

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataframe)
