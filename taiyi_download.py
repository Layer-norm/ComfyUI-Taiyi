import os
import json
import hashlib
from torch.hub import download_url_to_file

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "taiyilink.json"), 'r') as config_file:
    link_dict = json.load(config_file)

def sha_check(ckpt_path):
    sha256_hash = hashlib.sha256()
    with open(ckpt_path,"rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
        if sha256_hash.hexdigest() == link_dict["SHA256"]:
            return True
        else:
            return False

def taiyi_model_check(ckpt_path):
    return os.path.exists(ckpt_path)

def taiyi_download(ckpt_path):
    hf_endpoint = link_dict["HF_ENDPOINT"]
    taiyi_url = link_dict["TaiyiXL_link"]
    taiyi_url = taiyi_url.replace("{HF_ENDPOINT}", hf_endpoint)
    print(taiyi_url)
    download_url_to_file(url=taiyi_url, dst=ckpt_path)

#model SHA256 check
if __name__ == '__main__':
    # put your models dir here
    models_dir = 'I:/ComfyUI/ComfyUI/models'
    ckpt_path = os.path.join(models_dir, "checkpoints", "taiyi_diffusion_xl.safetensors")
    if not sha_check(ckpt_path):
        print("SHA256 check fail")
        taiyi_download(ckpt_path)
    else:
        print("SHA256 check pass")