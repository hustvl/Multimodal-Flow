# Architecture and Extensions

MF is a configuration-driven framework. The paper setup is one resolved recipe
inside the framework, not a separate launcher or model path.

## Core dataflow

```text
ModalitySpec -> SemanticSequence -> TaskDefinition -> CompiledSequence
             -> PhysicalLayout -> ModelInput -> Objective -> ModelOutput
             -> Trainer / Sampler / Decoder
```

The configuration selects codecs, task mixtures, data sources, model size,
and runtime behavior. The resolved configuration and extension manifest are
recorded with the checkpoint.

## Sequence-first contracts

`MultimodalSequence` is the semantic representation of an ordered sample. A
sequence contains named chunks with modality, condition/target role, token
budget, temporal metadata, and output slots. `CompiledSequence` is the single
authoritative hand-off to the physical model: it produces token spans,
attention visibility, flow targets, decoder targets, and output positions.

Planners, collators, attention, objectives, and decoders consume the compiled
sequence instead of inferring meaning from chunk order. This is the boundary
for video, image editing, and future multimodal sequences.

## Modular layers

| Layer | Responsibility |
| --- | --- |
| `mf.data` | Read records and preserve resumable data cursors |
| `mf.contracts` | Define modalities, sequences, tasks, geometry, and physical layouts |
| `mf.codecs` | Encode inputs and decode model states for each modality |
| `mf.modeling` | Compile physical layouts, attention, backbone, and output heads |
| `mf.training` | Compose objectives, optimize, log, and save checkpoints |
| `mf.inference` | Load checkpoints and run physical forward or generation loops |
| `mf.registries` | Coordinate extension registration and lifecycle |

## Extension lifecycle

Extensions are loaded before configuration validation and frozen before model,
data, or inference objects are constructed:

```python
from mf.extensions import freeze_extensions, load_extensions

load_extensions(("my_project.mf_extension",))
# resolve and validate the YAML configuration
freeze_extensions()
# construct the model, data pipeline, trainer, or inference bundle
```

The frozen manifest covers modality, task, codec, physical layout, objective,
output head, sampler, decoder, and generation definitions. Late registration
is rejected so that model heads, packing, objectives, and inference cannot
silently disagree.

## Adding a modality

An extension should register one coherent modality contract and provide the
pieces needed by its physical path:

1. a modality definition and codec;
2. a sequence-to-physical layout adapter;
3. task and sampler definitions;
4. output head and objective components;
5. a decoder and, when generation is required, a generation loop.

The generic codec registry is used for audio, depth, video, or other future
modalities. The built-in vision and text registries are typed views for the
paper recipe, not separate namespaces. A codec name must resolve to a codec
whose declared modality matches the sequence definition.

## Data and inference adapters

A custom data factory returns the standard task-batch contract and implements
`state_dict()` / `load_state_dict()` for exact cursor restoration. It should
translate source-specific records into semantic sequences at the data boundary.

Inference uses the same extension loader as training and evaluation. A new
generation mode registers a complete denoise/decode loop and consumes the
physical pipeline; it does not add another image/text branch to the trainer.
The built-in text continuation, image captioning, and text-to-image commands
are convenience recipes over this shared path.
