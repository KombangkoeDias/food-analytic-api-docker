import numpy as np
import cv2
import torch
from .dpt.models import DPTDepthModel
from .configs import dpt_config
from torchvision.transforms import Compose
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
        depth = self.model(image)
        depth = depth.cpu().detach().numpy()
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
                    lambda x: np.expand_dims(x.transpose(2, 0, 1), axis=0),
                    lambda x: (x / 255.0).astype(np.float32),
                    lambda x: {"image": x},
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
                ]
            ),
        )
