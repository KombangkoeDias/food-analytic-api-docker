import timm
import torch
from torch import nn
from .config import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class foodNet(nn.Module):
  def __init__(self):
    super(foodNet, self).__init__()
    self.pretrained_model = timm.create_model(config["model_name"], pretrained=False)
    self.pretrained_model.head.fc = nn.Linear(self.pretrained_model.head.fc.in_features, len(config["idx_to_class"]))

  def forward(self, input):
    input = input.to(device)
    x = self.pretrained_model(input)
    return x

model = foodNet()
model.to(device)
model.load_state_dict(torch.load(config["model_weight_filepath"], map_location=device))
model.eval()