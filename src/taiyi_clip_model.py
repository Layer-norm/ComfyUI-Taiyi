"""
    This file is modified from ComfyUI/comfy/clip_model.py
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
from comfy.clip_model import CLIPEmbeddings, CLIPEncoder

class TaiYiCLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        vocab_size = config_dict["vocab_size"]
        num_positions = config_dict["max_position_embeddings"]
        self.eos_token_id = config_dict["eos_token_id"]

        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, vocab_size=vocab_size, num_positions=num_positions, dtype=torch.float32, device=device, operations=operations)
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations)
        self.final_layer_norm = operations.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, attention_mask=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=torch.float32):
        x = self.embeddings(input_tokens, dtype=dtype)
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), float("-inf"))

        causal_mask = torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device).fill_(float("-inf")).triu_(1)
        if mask is not None:
            mask += causal_mask
        else:
            mask = causal_mask

        x, i = self.encoder(x, mask=mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)

        pooled_output = x[torch.arange(x.shape[0], device=x.device), input_tokens.to(dtype=torch.int, device=x.device).argmax(dim=-1),]
        return x, i, pooled_output


class TaiYiCLIPTextModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = TaiYiCLIPTextModel_(config_dict, dtype, device, operations)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])