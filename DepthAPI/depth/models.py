import numpy as np
import cv2
import torch
from .dpt.models import DPTDepthModel
from .configs import dpt_config
from torchvision.transforms import Compose, ToTensor
from .dpt.transforms import Resize, NormalizeImage, PrepareForNet


class DepthInferenceWrapper:
    def __init__(self, model, transform):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.transform = transform

    @torch.no_grad()
    def predict(self, image):
        image = self.transform(image)
        image = image.to(self.device)
        inv_depth = self.model(image)
        inv_depth = inv_depth.cpu().detach().numpy()
        depth = 1 / inv_depth
        depth = depth[0].tolist()
        return depth


class DPTInferenceWrapper(DepthInferenceWrapper):
    def __init__(self):
        super().__init__(
            DPTDepthModel(
                path=dpt_config["model_path"],
                scale=dpt_config["scale"],
                shift=dpt_config["shift"],
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            ),
            Compose(
                [
                    lambda x: {"image": x / 255.0},
                    Resize(
                        dpt_config["image_size"][0],
                        dpt_config["image_size"][1],
                        resize_target=True,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="minimal",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    PrepareForNet(),
                    lambda x: x["image"],
                    lambda x: torch.unsqueeze(torch.as_tensor(x), 0),
                ]
            ),
        )
