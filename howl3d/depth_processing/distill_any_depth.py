import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from torchvision.transforms import Compose
from tqdm import tqdm

from howl3d.depth_processing.base import BaseDepthProcessor
from howl3d.utils.directories import ensure_directory
from thirdparty.distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from thirdparty.distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from thirdparty.distillanydepth.modeling.archs.dam.dam import DepthAnything

dad_model_configs = {
    "vits": {"encoder": 'vits', "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": 'vitb', "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {'encoder': "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024], "use_bn": False, "use_clstoken": False, "max_depth": 150.0, "mode": 'disparity', "pretrain_type": 'dinov2', "del_mask_token": False},
}

# Adapted from DistillAnyDepth's app.py -- https://github.com/Westlake-AGI-Lab/Distill-Any-Depth/blob/main/app.py
class DistillAnyDepthProcessor(BaseDepthProcessor):
    def __init__(self, config):
        super().__init__(config, "dad_depth_dir")

    def get_depth_normalization_params(self, depth):
        d_min = depth.min()
        d_max = depth.max()
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        return depth_norm

    def compute_depths(self, frame_idx, model):
        frame_path = self.config["frames_output_path"] / f"frame_{frame_idx:06d}.png"

        # Load image
        image = cv2.imread(str(frame_path))
        image_np = np.array(image)[..., ::-1] / 255
        transform = Compose([Resize(700, 700, resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC), NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), PrepareForNet()])
        image_tensor = transform({'image': image_np})['image']
        image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(self.config["device"])

        # Process through depth model
        depth, _ = model(image_tensor)

        # Extract depth from the prediction, resizing it to the original dimensions
        height, width = image_np.shape[:2]
        depth_map = depth.detach().cpu().numpy()[0, 0, :, :]
        depth_map = cv2.resize(depth_map, (width, height), cv2.INTER_LINEAR)

        # Save depth map to disk
        depth_path = self.config["depths_output_path"] / f"depth_{frame_idx:06d}.npy"
        np.save(str(depth_path), depth_map)

    def process(self):
        if self.should_process("dad_depth_dir"):
            # Load DistillAnyDepth model
            dad_model = self.config["dad_model"]
            print(f"Loading DistillAnyDepth model, {dad_model} variant")
            distill_any_depth = DepthAnything(**dad_model_configs[dad_model]) if dad_model == "vitl" else DepthAnythingV2(**dad_model_configs[dad_model])
            distill_any_depth.load_state_dict(load_file(f"models/distill_any_depth/{dad_model}.safetensors"))
            distill_any_depth = distill_any_depth.to(self.config["device"]) # TODO: DepthAnythingV2 has an eval() here, is that really unnecessary?

            print(f"Computing depths for {self.config['video_info']['frames']} frames")

            # Ensure depth output directory exists, cleaning up existing contents if they exist
            ensure_directory(self.config["depths_output_path"])

            # Construct a manually updated progress bar
            pbar = tqdm(range(self.config["video_info"]["frames"]))

            # Compute depth for each frame
            for i in range(self.config["video_info"]["frames"]):
                self.compute_depths(i, distill_any_depth)
                pbar.update(1)
                pbar.refresh()

            # Cleanup model from GPU
            del distill_any_depth
            torch.cuda.empty_cache()
        else:
            print("Depths already exported, skipping depth computation")