# Data Preparation

MF keeps data and model assets outside the source checkout. Download the
assets you need, place them in a stable local layout, and point the selected
configuration at those paths.

## Public data

| Source | Used for | Reference |
| --- | --- | --- |
| Ultra-FineWeb-L3 | Text-only pretraining | [Dataset](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3) |
| GPIC | Image understanding and image generation | [Dataset](https://huggingface.co/datasets/stanford-vision-lab/gpic) |
| BLIP3o-60k | SFT text-to-image | [Dataset](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) |
| DALL-E3 | SFT text-to-image | [Dataset](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset) |
| ShareGPT-4o Image | SFT text-to-image | [Dataset](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image) |
| LLaVA-1.5 Instruct-150K | SFT VQA | [Official project](https://github.com/haotian-liu/LLaVA) |


## Model assets

| Asset | Role | Reference |
| --- | --- | --- |
| T5-small | Text encoder and tokenizer | [Model](https://huggingface.co/t5-small) |
| SigLIP2 | Vision encoder | [google/siglip2-so400m-patch14-224](https://huggingface.co/google/siglip2-so400m-patch14-224) |
| Scale RAE | Vision codec and decoder | [nyu-visionx/siglip2_decoder](https://huggingface.co/nyu-visionx/siglip2_decoder) |
| MF text decoder | Text reconstruction and generation | [Multimodal-Flow release](https://huggingface.co/hustvl/Multimodal-Flow) |
| Vision statistics | Latent normalization | [Multimodal-Flow release](https://huggingface.co/hustvl/Multimodal-Flow) |

The paths are configuration values. The default local layout is:

~~~text
Multimodal-Flow/
  assets/
    t5-small/
    siglip2-so400m-patch14-224/
    scale-rae-decoder/
    text_decoder.pt
    vision_stats.pt
  data/
    gpic/
    ultrafineweb-l3/
    text_to_image_sft/
    llava-1.5-instruct/
    coco/
~~~

For Hugging Face repositories, download snapshots with the Hugging Face CLI
and place each snapshot under the corresponding configured directory. The
repository does not require a particular cache location.

## Local quickstart bundle

The quickstart configs use a small local bundle so the first training run does
not depend on a dataset-specific loader. Its default layout is:

~~~text
data/
  train.jsonl
  images/
    train/
      000000.png
~~~

Each JSONL record may contain text, an image, a caption, and optional
question-answer pairs:

~~~json
{"id":"image-0001","image":"train/000000.png","caption":"A green circle.","qa":[{"question":"What color is the circle?","answer":"green"}],"text":null}
{"id":"text-0001","image":null,"caption":null,"qa":[],"text":"A short text example."}
~~~

The image path is relative to the configured bundle root (data/images by
default), not the JSONL file. Include both captioned images and text records
for the default mixed-task quickstart. The same format is
enough for a small smoke run; larger projects can point the YAML paths at GPIC,
Ultra-FineWeb-L3, or an adapter for another source.

## Configuration paths

Pretraining uses the configured `data.image_text` and `data.text` roots. SFT
uses `sft.text_to_image.root`, `sft.vqa.train_json`, and `sft.vqa.image_root`.
Model paths are under `codecs.*`, `model.text_decoder`, and `latent_stats.*`.
Change these values in YAML; no source edit is required.

The official VQA reader expects records with `image` and `conversations`
fields, for example:

~~~json
{"image": "train2017/000000000001.jpg", "conversations": [{"from": "human", "value": "<image>\\nWhat is shown?"}, {"from": "gpt", "value": "A group of people outdoors."}]}
~~~

## Your own data

You may use another dataset by converting it to the documented record format
or by providing a data adapter. Keep the adapter at the data boundary and
return MF task samples with their sequence, modality, and supervision fields.
See [Architecture](ARCHITECTURE.md) for the extension boundary.

Every input path must be readable before launch. Keep dataset revisions,
asset revisions, and preprocessing choices with the run output.
