"""
    This file is modified from ComfyUI/comfy/sdxl_clip.py
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

import os
import torch
from comfy import sd1_clip
from .taiyi_clip_model import TaiYiCLIPTextModel

TAIYI_TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tokenizer")
TAIYI_TOKENIZER2_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tokenizer2")

class TaiYiXLClip(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", max_length=225, freeze=True, layer="penultimate", layer_idx=None, dtype=None, model_options={}):
        if layer == "penultimate":
            layer="hidden"
            layer_idx=-2

        taiyi_textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_clip_config_taiyi.json")
        super().__init__(device=device, freeze=freeze, layer=layer, layer_idx=layer_idx, textmodel_json_config=taiyi_textmodel_json_config, dtype=dtype,
                         model_class=TaiYiCLIPTextModel, special_tokens={"start": 57473, "end": 57474, "pad": 0}, layer_norm_hidden_state=False, return_projected_pooled=True, model_options=model_options)
    
    def load_sd(self, sd):
        return super().load_sd(sd)

class TaiYiXLClipTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, tokenizer_path=TAIYI_TOKENIZER_PATH, embedding_directory=None, tokenizer_data={}):
        super().__init__(tokenizer_path, max_length=512, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=768, embedding_key='clip_l')



class TaiYiXLClipG(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", max_length=225, freeze=True, layer="penultimate", layer_idx=None, dtype=None, model_options={}):
        if layer == "penultimate":
            layer="hidden"
            layer_idx=-2

        taiyi_textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_config_bigg_taiyi.json")
        super().__init__(device=device, freeze=freeze, layer=layer, layer_idx=layer_idx, textmodel_json_config=taiyi_textmodel_json_config, dtype=dtype,
                         model_class=TaiYiCLIPTextModel, special_tokens={"start": 57473, "end": 57474, "pad": 0}, layer_norm_hidden_state=False, return_projected_pooled=True, model_options=model_options)

    def load_sd(self, sd):
        return super().load_sd(sd)

class TaiYiXLClipGTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, tokenizer_path=TAIYI_TOKENIZER2_PATH, embedding_directory=None, tokenizer_data={}):
        super().__init__(tokenizer_path, max_length=512, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=1280, embedding_key='clip_g')


class TaiYiXLTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = TaiYiXLClipTokenizer(embedding_directory=embedding_directory)
        self.clip_g = TaiYiXLClipGTokenizer(embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)
    
    def state_dict(self):
        return {}

class TaiYiXLClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.clip_l = TaiYiXLClip(device=device, dtype=dtype, model_options=model_options)
        self.clip_g = TaiYiXLClipG(device=device, dtype=dtype, model_options=model_options)
        self.dtypes = set([dtype])

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.clip_g.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_g.reset_clip_options()
        self.clip_l.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return torch.cat([l_out, g_out], dim=-1), g_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
            return self.clip_g.load_sd(sd)
        else:
            return self.clip_l.load_sd(sd)
