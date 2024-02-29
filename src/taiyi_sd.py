import torch

from comfy import utils
from comfy import model_management

from comfy import clip_vision
# from comfy.sd import VAE, CLIP, load_model_weights
from comfy.sd import VAE, CLIP
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

def taiyi_load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True):
    sd = utils.load_torch_file(ckpt_path)
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    parameters = utils.calculate_parameters(sd, "model.diffusion_model.")
    load_device = model_management.get_torch_device()

    model_config = taiyi_detection.model_config_from_unet(sd, "model.diffusion_model.")    
    unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))

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
        model_patcher = ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device(), current_device=inital_load_device)
        if inital_load_device != torch.device("cpu"):
            print("loaded straight to GPU")
            model_management.load_model_gpu(model_patcher)

    return (model_patcher, clip, vae, clipvision)