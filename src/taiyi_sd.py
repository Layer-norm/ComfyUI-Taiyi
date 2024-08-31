"""
    This file is modified from ComfyUI/comfy/sd.py
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import torch

from comfy import utils
from comfy import model_management

from comfy import clip_vision
from comfy.sd import VAE, CLIP
from comfy.model_detection import unet_prefix_from_state_dict
from comfy.model_patcher import ModelPatcher

from . import taiyi_detection

def load_model_weights(model, sd):
    # print(f"sd:{sd['cond_stage_model.clip_l.transformer.text_model.embeddings.token_embedding.weight'].size()}")
    # sd:torch.Size([57475, 768])
    m, u = model.load_state_dict(sd, strict=False)
    m = set(m)
    unexpected_keys = set(u)

    k = list(sd.keys())
    for x in k:
        if x not in unexpected_keys:
            w = sd.pop(x)
            del w
    if len(m) > 0:
        print("missing", m)
    return model

def taiyi_load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}):
    sd = utils.load_torch_file(ckpt_path)

    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    diffusion_model_prefix = unet_prefix_from_state_dict(sd)

    parameters = utils.calculate_parameters(sd, "model.diffusion_model.")

    weight_dtype = utils.weight_dtype(sd, diffusion_model_prefix)

    load_device = model_management.get_torch_device()

    model_config = taiyi_detection.model_config_from_unet(sd, "model.diffusion_model.")
    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if weight_dtype is not None:
        unet_weight_dtype.append(weight_dtype)

    model_config.custom_operations = model_options.get("custom_operations", None)

    unet_dtype = model_options.get("weight_dtype", None)

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        offload_device = model_management.unet_offload_device()
        model = model_config.get_model(sd, "model.diffusion_model.", device=inital_load_device)
        model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae_sd = utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd)

    if output_clip:
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                clip = CLIP(clip_target, embedding_directory=embedding_directory)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    print("clip missing:", m)

                if len(u) > 0:
                    print("clip unexpected:", u)
            else:
                print("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    if output_model:
        model_patcher = ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
        if inital_load_device != torch.device("cpu"):
            print("loaded straight to GPU")
            model_management.load_models_gpu([model_patcher], force_full_load=True)

    return (model_patcher, clip, vae, clipvision)