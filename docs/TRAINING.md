# Training

Create the Conda environment and install MF as described in the main
[README](../README.md), prepare the data and model assets from
[Data Preparation](DATA.md), then choose a YAML
configuration. The shipped configs are complete recipes; users should change
paths and ordinary runtime options in YAML rather than edit Python code.

## Configurations

The following are the defaults in the repository:

| Config | Use | Steps | Save interval | World | Global batch | Pack length | GC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `quickstart-0.6b.yaml` | 0.6B local-bundle quickstart | 100K | 20K | 4 | 4 | 8192 | on |
| `quickstart-1.6b.yaml` | 1.6B local-bundle quickstart | 100K | 20K | 4 | 4 | 8192 | on |
| `pretrain.yaml` | 1.6B full pretraining recipe | 100K | 20K | 4 | 4 | 8192 | off |
| `sft.yaml` | Text-to-image and VQA SFT | 20K | 10K | 4 | 4 | 8192 | on |

Both quickstart configs use `data/train.jsonl` and `data/images/` for a small
local bundle. The full pretraining recipe uses the external datasets described
in `DATA.md`. The SFT mixture is 30% text-to-image and 70% VQA.

## Pretraining

Run locally or with a distributed launcher:

~~~bash
torchrun --standalone --nproc-per-node=4 -m mf train \
  --config configs/quickstart-1.6b.yaml

torchrun --standalone --nproc-per-node=4 -m mf train \
  --config configs/pretrain.yaml
~~~

The launcher process count must match `distributed.world_size`; the global
batch must match the world size, per-rank batch, and accumulation settings.
The Muon shard group must divide the world size. For a single GPU:

~~~bash
mf train --config configs/quickstart-0.6b.yaml \
  --set distributed.world_size=1 \
  --set distributed.global_batch_size=1 \
  --set optimizers.muon_shard_group_size=1
~~~

Override paths and runtime values with dotted `--set` options:

~~~bash
torchrun --standalone --nproc-per-node=4 -m mf train --config configs/pretrain.yaml \
  --set data.image_text.root=/path/to/gpic \
  --set data.text.ultrafineweb_multi_style.root=/path/to/ultrafineweb/multi_style \
  --set data.text.ultrafineweb_qa.root=/path/to/ultrafineweb/qa
~~~

The quickstart and SFT configs enable `model.gradient_checkpointing` for
lower-memory development-machine runs. It is implemented in the MF block
stack, not just accepted by the schema. When enabling it in another config,
also set `model.compile_packed_blocks=false`; the two execution modes are
intentionally mutually exclusive.

~~~bash
torchrun --standalone --nproc-per-node=4 -m mf train --config configs/pretrain.yaml \
  --set model.gradient_checkpointing=true \
  --set model.compile_packed_blocks=false
~~~

## SFT

The public SFT recipe reads Hugging Face dataset snapshots and the official
LLaVA-1.5 Instruct format:

~~~bash
torchrun --standalone --nproc-per-node=4 -m mf sft --config configs/sft.yaml \
  --init-from outputs/mf_pretrain/checkpoints/step_100000
~~~

Use `--resume` for an interrupted SFT run. Use `--init-from` when starting a
new SFT trajectory from a completed pretraining checkpoint.

## Training outputs

Each run writes its concise progress summary and machine-readable metrics under
the configured logging output directory:

~~~text
outputs/<run>/logs/
  train.log       # step, loss, learning rate, grad norm, speed, ETA
  metrics.jsonl   # complete metrics for scripts and later analysis
  events.out...   # TensorBoard event files
~~~

The shipped recipes use TensorBoard logging. Start a local dashboard with:

~~~bash
tensorboard --logdir outputs/<run>/logs
~~~

train.log is intentionally short. Detailed timing and task-level metrics stay
in metrics.jsonl and the TensorBoard curves, so the console remains readable
while the full run record remains available.

## Resume and outputs

~~~bash
torchrun --standalone --nproc-per-node=4 -m mf train --config configs/pretrain.yaml \
  --resume outputs/mf_pretrain/checkpoints/step_020000
~~~

`--resume` restores model, optimizer, scheduler, EMA, data cursor, and step
state when present. A complete checkpoint contains:

~~~text
outputs/<run>/checkpoints/step_<N>/
  COMPLETED
  run_manifest.json
  trainer_state.json
~~~

Only use a checkpoint after `COMPLETED` and `run_manifest.json` are present.
The resolved configuration, config hash, metrics, and checkpoint manifests are
written under the configured output directory.
