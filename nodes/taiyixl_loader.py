import os
import folder_paths
from ..src.taiyi_sd import taiyi_load_checkpoint_guess_config
from ..taiyi_download import taiyi_model_check, taiyi_download

class TaiYiXLCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ckpt_name": ("STRING", {"default": "taiyi_diffusion_xl.safetensors"}),
                }
            }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "TaiyiXL"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = os.path.join(folder_paths.models_dir, "checkpoints", ckpt_name)
        if not taiyi_model_check(ckpt_path):
            taiyi_download(ckpt_path)
        out = taiyi_load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]
    
NODE_CLASS_MAPPINGS = {
    "TaiyiXLCheckpointLoader": TaiYiXLCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TaiyiXLCheckpointLoader": "TaiyiXLCheckpointLoader"
}
