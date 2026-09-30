# Inference

Activate the Conda environment from the main [README](../README.md) before
running the commands below.

The same inference commands work with the released MF checkpoint and with a
checkpoint produced by your own training run. Point `--checkpoint` at a
complete checkpoint directory; no source-code changes are needed.

The released Hugging Face package keeps shared assets separate:

~~~text
MF/pretrain/
MF/sft/
Text Decoder/
Vision statistics/
~~~

Point the checkpoint argument to either MF/pretrain or MF/sft. The model
config resolves the shared text decoder and vision statistics automatically.

Text inference only needs the model package. Captioning and image generation
also need the matching Scale RAE decoder assets. Set `MF_ASSETS_ROOT` to a
directory containing `scale_rae_decoder/` before running those commands.

## Text continuation

~~~bash
mf infer --checkpoint <CHECKPOINT> --weights ema text \
  --prompt "A short language model can"
~~~

## Image understanding

~~~bash
mf infer --checkpoint <CHECKPOINT> --weights ema caption \
  --image /path/to/image.jpg \
  --prompt "Describe this image."
~~~

## Text-to-image generation

~~~bash
mf infer --checkpoint <CHECKPOINT> --weights ema image \
  --prompt "A quiet observatory above the clouds." \
  --output outputs/sample.png
~~~

The default sampler uses 16 flow steps for text and 64 flow steps for image
generation. These are resolved from the checkpoint configuration; inference is
not split into separate user-managed stages. Use `--weights raw` when raw
weights are preferred over EMA weights.

Inference includes sequential chunk generation and chunk-level key/value
caching. The command accepts either the released model layout or the matching
layout produced by the training configs.
