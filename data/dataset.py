from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2





def cifar10_dataset(location='../data', train=True, download=True, transform=None):
    return datasets.CIFAR10(location, train=train, download=download, transform=transform)

''' using pytorch albumenations'''

'''def train_tarnsform():
    A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ]
)'''''

    
    
def transformations(augmentation=True, rotation=3.0):
        transforms_list = [
        A.SmallestMaxSize(max_size=160),
        #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
        ]
    
    return A.Compose(transforms_list)

'''
def transformations(augmentation=False, rotation=3.0):
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    if augmentation:
        transforms_list = [
                              transforms.RandomRotation((-rotation, rotation), fill=(1,))
                          ] + transforms_list

    return transforms.Compose(transforms_list)'''
