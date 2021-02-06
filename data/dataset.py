from torchvision import datasets, transforms


def cifar10_dataset(location='../data', train=True, download=True, transform=None):
    return datasets.CIFAR10(location, train=train, download=download, transform=transform)


def transformations(augmentation=False, rotation=3.0):
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    if augmentation:
        transforms_list = [
                              transforms.RandomRotation((-rotation, rotation), fill=(1,))
                          ] + transforms_list

    return transforms.Compose(transforms_list)
