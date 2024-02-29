import torch
from comfy import model_base
from comfy import utils
from comfy import supported_models_base
from comfy import latent_formats
from comfy import diffusers_convert

# from comfy.supported_models import models

from . import taiyi_encode

class TaiYiXL(supported_models_base.BASE):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

    latent_format = latent_formats.SDXL

    def model_type(self, state_dict, prefix=""):
        if "v_pred" in state_dict:
            return model_base.ModelType.V_PREDICTION
        else:
            return model_base.ModelType.EPS

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.SDXL(self, model_type=self.model_type(state_dict, prefix), device=device)
        if self.inpaint_model():
            out.set_inpaint()
        return out

    def process_clip_state_dict(self, state_dict):
        keys_to_replace = {}
        replace_prefix = {}

        replace_prefix["conditioner.embedders.0.transformer.text_model"] = "cond_stage_model.clip_l.transformer.text_model"
        state_dict = utils.transformers_convert(state_dict, "conditioner.embedders.1.model.", "cond_stage_model.clip_g.transformer.text_model.", 32)
        keys_to_replace["conditioner.embedders.1.model.text_projection"] = "cond_stage_model.clip_g.text_projection"
        keys_to_replace["conditioner.embedders.1.model.text_projection.weight"] = "cond_stage_model.clip_g.text_projection"
        keys_to_replace["conditioner.embedders.1.model.logit_scale"] = "cond_stage_model.clip_g.logit_scale"

        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)
        state_dict = utils.state_dict_key_replace(state_dict, keys_to_replace)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        keys_to_replace = {}
        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")
        for k in state_dict:
            if k.startswith("clip_l"):
                state_dict_g[k] = state_dict[k]

        state_dict_g["clip_l.transformer.text_model.embeddings.position_ids"] = torch.arange(512).expand((1, -1))
        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]
        for p in pop_keys:
            if p in state_dict_g:
                state_dict_g.pop(p)

        replace_prefix["clip_g"] = "conditioner.embedders.1.model"
        replace_prefix["clip_l"] = "conditioner.embedders.0"
        state_dict_g = utils.state_dict_prefix_replace(state_dict_g, replace_prefix)
        return state_dict_g

    def clip_target(self):
        return supported_models_base.ClipTarget(taiyi_encode.TaiYiXLTokenizer, taiyi_encode.TaiYiXLClipModel)

models = [TaiYiXL]
