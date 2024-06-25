import os
import numpy as np
import albumentations
import bisect
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import cv2

prob_resize = 0.3
prob_rgbshift = 0.3
prob_colorshift = 0.3
interpolates = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

class ImagePaths(Dataset):
    def __init__(self, paths, dirpath, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.dirpath = dirpath

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
        self.hflip = albumentations.HorizontalFlip()
        self.vflip = albumentations.VerticalFlip()
        self.recolor = albumentations.ChannelShuffle(p=prob_rgbshift)
        self.colorshift = albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=prob_colorshift)

        
    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(os.path.join(self.dirpath, image_path))
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # preporess
        h, w = image.shape[0], image.shape[1]
        interpolate = random.choice(interpolates)
        if 0.8*h > self.size and 0.8*w > self.size:
            self.rescaler = albumentations.RandomScale(scale_limit=[-0.2,1], interpolation=interpolate, always_apply=False, p=prob_resize)
        else:
            self.rescaler = albumentations.RandomScale(scale_limit=[0,1], interpolation=interpolate, always_apply=False, p=prob_resize)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper, self.hflip, self.vflip, self.recolor, self.colorshift])

        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, dirpath):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, dirpath=dirpath, size=size, random_crop=True)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, dirpath):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, dirpath=dirpath, size=size, random_crop=True)


class Eval_ImagePaths(Dataset):
    def __init__(self, paths, dirpath, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.dirpath = dirpath

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        
    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(os.path.join(self.dirpath, image_path))
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image
    
    def __getitem__(self, i):
        image = self.preprocess_image(self.labels["file_path_"][i])
        image = image.transpose((2,0,1))
        return image
    
def load_data(args):
    with open(args.txt_file, "r") as f:
        paths = f.read().splitlines()
    train_data = Eval_ImagePaths(paths=paths, dirpath=args.dataset_path, size=args.image_size, random_crop=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader
