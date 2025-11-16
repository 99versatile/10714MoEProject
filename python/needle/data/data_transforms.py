import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        # pad all sides of img
        padded_img = np.zeros([img.shape[0] + 2*self.padding, img.shape[1] + 2*self.padding, img.shape[2]])
        padded_img[self.padding:self.padding+img.shape[0], self.padding:self.padding+img.shape[1], :] = img

        crop_start_x = self.padding + shift_x
        crop_start_y = self.padding + shift_y
        crop_end_x = crop_start_x + img.shape[0]
        crop_end_y = crop_start_y + img.shape[1]
        
        return padded_img[crop_start_x:crop_end_x, crop_start_y:crop_end_y, :]
        ### END YOUR SOLUTION
