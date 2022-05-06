from torchvision import transforms
from .config import config

transform = transforms.Compose([
    transforms.Resize(size= config["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])