# Third-party notices

This repository includes adaptations from the following open-source work.
Each component retains its original license and attribution; these notices are
separate from the Apache-2.0 license of the MF project.
External model weights and datasets are not redistributed.

- **ELF / UniFlow** (MIT, copyright 2026 ELF authors):
  the T5-small encoder and latent text decoder design in
  `src/mf/codecs/t5.py` and `src/mf/codecs/text_decoder.py` were adapted from
  the UniFlow implementation.
- **RAE** (MIT, copyright 2025 Boyang Zheng):
  the DINO image codec and Muon optimizer implementation were adapted from
  [RAE](https://github.com/bytetriper/RAE). The source files inspected for
  this adaptation have SHA-256 prefixes `6564d285518c` (RAE model),
  `7bb854b24312` (DINO encoder), and `0d0ee0b16a91` (Muon).
  The decoder shell also retains Hugging Face/Facebook 2022 Apache-2.0
  provenance; see `LICENSES/Apache-2.0.txt`.
- **Hugging Face Transformers** (Apache-2.0):
  used as a runtime dependency; no Transformers source is copied here.

The MF project license is in `LICENSE`; third-party attribution is listed above.
