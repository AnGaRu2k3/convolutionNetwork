from torchvision import datasets, transforms
from PIL import Image
class RGB(object):
    def __call__(self, image: Image.Image):
        if image.mode == "L":
            image = image.convert("RGB")
        return image
def get_dataset(name):
    if name == "MNIST":
        transform = transforms.Compose([RGB(), transforms.ToTensor()])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        valid_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif name == "FashionMNIST":
        transform = transforms.Compose([RGB(),transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        valid_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif name == "Caltech101":
        transform = transforms.Compose([transforms.Resize((256, 256)), RGB(), transforms.ToTensor()])
        train_dataset = datasets.Caltech101(root='./data', download=True, transform=transform)
        valid_dataset = datasets.Caltech101(root='./data', download=True, transform=transform)
        num_classes = 101
    elif name == "Caltech256":
        transform = transforms.Compose([transforms.Resize((256, 256)), RGB(), transforms.ToTensor()])
        train_dataset = datasets.Caltech256(root='./data', download=True, transform=transform)
        valid_dataset = datasets.Caltech256(root='./data', download=True, transform=transform)
        num_classes = 256
    return num_classes, train_dataset, valid_dataset
