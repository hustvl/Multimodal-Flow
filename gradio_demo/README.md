# MF-1 Gradio demo

A web UI for [Multimodal Flow](https://github.com/hustvl/Multimodal-Flow) (MF-1). It has three tabs:
text to image (with a live denoising preview), image to text (captioning and questions), and text
continuation. It runs on the same GPU as the model, one request at a time.

## Requirements
- An NVIDIA GPU with native bf16 (Ampere or newer, such as L4, A10, A100, RTX 30/40, or H100).
- The Multimodal Flow environment from the repository root (`environment.yaml`), then `pip install -e .`.
- `pip install -r gradio_demo/requirements.txt` (pins Gradio 5.49.1).

## Assets
1. The checkpoint (Hugging Face release, documentation only; the code does not download it):
   ```bash
   huggingface-cli download hustvl/Multimodal-Flow --local-dir ./MF_weights
   ```
   Use `./MF_weights/MF/sft` as the checkpoint.
2. The Scale RAE image decoder, needed for image generation and captioning:
   ```bash
   huggingface-cli download nyu-visionx/siglip2_decoder --local-dir ./assets/scale_rae_decoder
   ```
3. Point `MF_ASSETS_ROOT` at the folder that contains `scale_rae_decoder/` (for example `./assets`).

## Run
```bash
export MF_CHECKPOINT=$PWD/MF_weights/MF/sft
export MF_ASSETS_ROOT=$PWD/assets
python gradio_demo/app.py                 # http://localhost:7860
python gradio_demo/app.py --share         # also a temporary public link
python gradio_demo/app.py --mock          # UI preview with no GPU or model
```
Options: `--port N`. Environment variables: `MF_DEVICE` (default `cuda`), `MF_WEIGHTS` (`ema` or `raw`),
`MF_LIVE_PREVIEW` (`0` turns the live preview off).

## Tests
Install the test dependencies first: `pip install pytest gradio_client`.

```bash
python -m pytest gradio_demo/tests -q
```
