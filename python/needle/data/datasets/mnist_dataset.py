from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        import gzip 
        import struct 

        with gzip.open(image_filename, 'rb') as f:
            magic_number = struct.unpack('>I', f.read(4))[0]
            num_img = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]

            img_data = f.read()
            images = np.frombuffer(img_data, dtype=np.uint8).reshape(num_img, num_rows * num_cols).astype(np.float32)
            images = images / 255.0

        with gzip.open(label_filename, 'rb') as f:
            magic_number = struct.unpack('>I', f.read(4))[0]
            num_img = struct.unpack('>I', f.read(4))[0]

            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)

        self.transforms = transforms
        
        images = images.reshape(num_img, num_rows, num_cols, 1)
        
        self.images = images
        self.labels = labels

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image = self.images[index]
        
        if self.transforms is not None:
            image = self.apply_transforms(image)
        return image, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION