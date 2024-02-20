# Comfyui TaiYi
TaiYiXLCheckpointLoader: An unoffical node support Taiyi-Diffusion-XL(Taiyi-XL) Chinese-English bilingual language text-to-image model
一个自制的 [Taiyi-Diffusion-XL](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B) 模型加载节点

## Install
- Download [Taiyi-Diffusion-XL-3.5B](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B/resolve/main/taiyi_diffusion_xl.safetensors) model to `/ComfyUI/models/checkpoints` folder
- Navigate to `/ComfyUI/custom_nodes/` folder 
- Run \
`git clone https://github.com/Layer-norm/ComfyUI-TaiYi.git` 
- Restart ComfyUI

## 安装
- 把 [Taiyi-Diffusion-XL-3.5B](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B/resolve/main/taiyi_diffusion_xl.safetensors) 模型下载到 `/ComfyUI/models/checkpoints` 文件夹
- 进入 `/ComfyUI/custom_nodes/`文件夹
- 终端运行 \
 `git clone https://github.com/Layer-norm/ComfyUI-TaiYi.git`
- 重启 ComfyUI

## Examples（示例）
<p align="center">
    <img src="example\purplebottle.png" width="800">
    <img src="example\whitecat.png" width="800">
</p>

## Matters needing attention（注意事项）
Because the original model was mainly tested on Fooocus, the performance on comfyui may differ from the original results.
In addition, the effect of Chinese prompts may be slightly inferior to English, and some misunderstandings is acceptable.
因为原模型主要在Fooocus上测试，在comfyui上的表现会与原始结果可能会有差异
此外中文提示词效果可能会略逊于英文，产生理解偏差是正常现象

## Acknowledgments（感谢）
[ComfyUI](https://github.com/comfyanonymous/ComfyUI.git) The most powerful and modular stable diffusion GUI and backend. 最好用的sd GUI框架 \
[Taiyi-Diffusion-XL](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B): Advancing Bilingual Text-to-Image Generation with Large Vision-Language Model Support[[1]](https://arxiv.org/abs/2401.14688)
[Fooocus-Taiyi-XL](https://github.com/IDEA-CCNL/Fooocus-Taiyi-XL.git) Taiyi-XL Deployment Webui. 官方TaiyiXL的foocus推理部署
