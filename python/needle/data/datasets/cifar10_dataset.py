import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.base_folder = base_folder
        self.transforms = transforms 
        self.split = train 
        self.p = p

        self.images = []
        self.labels = []
        self.train_data = sorted([file for file in os.listdir(base_folder) if "data_batch" in file])
        self.test_data = sorted([file for file in os.listdir(base_folder) if "test_batch" in file])

        data_split = self.train_data if self.split else self.test_data

        for files in data_split:
            with open(os.path.join(base_folder,files),"rb") as f:
                batch_data = pickle.load(f,encoding="latin1")
                self.images.append(batch_data["data"])
                self.labels.extend(batch_data["labels"])
        
        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images / 255.0

        # path = os.path.join(self.base_folder, "batches.meta")
        
        # with open(path,"rb") as infile:
        #     data = pickle.load(infile,encoding="latin1")
        #     self.classes = data["label_names"]
        #     self.classes_to_idx = {_class:i for i,_class in enumerate(self.classes)}


        self.images = self.images.reshape(-1, 3, 32, 32)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        image = self.images[index]
        
        if self.transforms is not None:
            image = self.apply_transforms(image)
        return image, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION
