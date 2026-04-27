from torch.utils.data import Dataset


class CLADataset(Dataset):
    """Custom dataset class for image classification task.

    Parameters:
        - transforms (callable): Transforms to be applied to the image sample.
        - images (list): List of image samples.
        - labels (list, optional): List of corresponding labels. Default is None.

    """
    def __init__(self, transforms, images, labels=None):
        """Initialize CLADataset with images and optional labels."""
        self.samples = images
        self.transform = transforms
        self.labs = labels

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Parameters:
            - idx (int): Index of the sample to retrieve.

        Returns:
            - tuple or tensor: Transformed image and label (if available).

        """
        sample = self.samples[idx]
        if self.labs is not None:
            label = self.labs[idx]

        # Apply transformations to the image
        transformed_img = self.transform(image=sample)

        # Return transformed image and original label
        if self.labs is not None:
            return transformed_img['image'], label

        return transformed_img['image']
