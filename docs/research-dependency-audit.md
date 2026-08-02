# Research dataset and teacher-model dependency audit

Audit date: 2026-07-30
Target: Ubuntu 26.04 LTS, x86-64, Intel i5-8400 (6 CPU cores), Intel UHD
630 only, 16 GiB installed RAM, 4 GiB swap, approximately 824 GiB free disk.

This document records the source audit, provisioned dependency state, and
verification procedure. All upstream revisions below are pinned so an update
does not silently change underneath Lumen.

## Executive conclusions

1. The dataset adapters and the neural teachers must be separate layers.
   EDM-98, Harmonix Set, CCMusic, and SALAMI annotations can be normalized
   without importing a neural runtime into Lumen's DMX process.
2. Lumen's core environment cannot contain the official SongFormer stack.
   Lumen requires Python 3.12+ and NumPy `>=2.3,<3`; SongFormer explicitly
   specifies Python 3.10 and pins NumPy `1.25.0`.
3. EDMFormer can be provisioned in a separate Python 3.12 environment because
   the EDM-98 package declares Python 3.10 through 3.12 and exposes a CPU
   device. CPU support is implemented, but upstream provides no i5-8400
   performance claim. It must be treated as an offline worker until measured.
4. The official SongFormer repository's batch inference path is CUDA-only in
   practice: it creates `cuda:<rank>` devices and its published speed was
   measured on an NVIDIA L40. The Hugging Face wrapper is less hard-coded and
   may be made to execute on CPU, but CPU is neither documented nor tested
   upstream. This computer has no CUDA-capable GPU.
5. Downloading all SongFormer training data is inappropriate for the first
   implementation pass. SongFormDB reports about 262 GB of repository storage;
   its JSONL labels are only a few megabytes and can be fetched selectively.
6. CCMusic `song_structure` is currently gated. The user must accept its terms
   with a Hugging Face account. Its dataset license is CC-BY-NC-ND-4.0, while
   the underlying audio rights are explicitly not granted by the dataset
   publisher.
7. Public annotations do not provide a general right to fetch copyrighted
   recordings. Importers should accept audio the user already possesses and
   validate/alignment-match it. No YouTube, Deezer, NetEase, or Spotify
   scraping should be made a hidden dependency.

## Host audit

Observed on this machine:

- `/etc/os-release`: Ubuntu 26.04 LTS.
- Shell `python3`: 3.14.6.
- Available pyenv interpreter: 3.12.8.
- GPU: Intel CoffeeLake-S GT2/UHD Graphics 630; no NVIDIA GPU.
- Available tools: `git`, `gcc`, `g++`, and `make`.
- Not currently found on `PATH`: `git-lfs`, `ffmpeg`, `sox`, and `cmake`.
- Lumen currently declares Python `>=3.12` and NumPy `>=2.3,<3`.

Recommended host packages before model provisioning:

```text
git-lfs
ffmpeg
libsndfile1
build-essential
pkg-config
cmake
```

Optional, only for legacy SALAMI/Harmonix alignment or vocoder work:

```text
sox
libsox-fmt-all
ninja-build
```

`ffmpeg` is the canonical external decoder/resampler for MP3, M4A, and other
compressed captures. `libsndfile1` supports SoundFile-compatible WAV/FLAC
decoding. `git-lfs` is required because EDM-98 checkpoints are Git LFS
pointers in a normal source checkout.

## 1. EDM-98 and EDMFormer

### Pinned sources

- Repository: <https://github.com/25ohms/EDM-98>
- Audited commit:
  [`2dd942f2f9e71ffd826346828eeaba1dd3ece56a`](https://github.com/25ohms/EDM-98/tree/2dd942f2f9e71ffd826346828eeaba1dd3ece56a)
- Package definition:
  <https://github.com/25ohms/EDM-98/blob/2dd942f2f9e71ffd826346828eeaba1dd3ece56a/pyproject.toml>
- Installer:
  <https://github.com/25ohms/EDM-98/blob/2dd942f2f9e71ffd826346828eeaba1dd3ece56a/scripts/install_inference_deps.sh>
- Inference pipeline:
  <https://github.com/25ohms/EDM-98/blob/2dd942f2f9e71ffd826346828eeaba1dd3ece56a/src/edm98/inference/pipeline.py>
- Paper: <https://arxiv.org/abs/2603.08759>

Transitive source repositories used by the official installer:

- MuQ commit
  [`28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a`](https://github.com/tencent-ailab/MuQ/tree/28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a)
- MusicFM commit
  [`b83ebedb401bcef639b26b05c0c8bee1dc2dfe71`](https://github.com/minzwon/musicfm/tree/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71)

The EDM-98 installer currently installs MuQ and clones MusicFM from unpinned
default branches. Lumen must substitute the commit pins above.

### Declared Python dependencies

EDM-98 declares Python `>=3.10` and explicitly classifies 3.10, 3.11, and
3.12. Dataset-only installation has no runtime dependencies.

The `inference` extra declares:

```text
PyYAML>=6.0
torch>=2.4.0
librosa>=0.11.0
omegaconf>=2.3.0
safetensors>=0.5.3
x-transformers>=2.4.14
scipy>=1.15.0
```

MuQ 0.1.0 declares Python `>=3.8` and:

```text
einops
librosa
nnAudio
numpy
soundfile
torch
torchaudio
tqdm
transformers
easydict
x_clip
```

MusicFM is source-only: it has no `pyproject.toml` or `setup.py`. Its inference
imports require at least:

```text
torch
torchaudio
numpy
einops
transformers
```

EDM-98 therefore needs both an exact MusicFM checkout on `PYTHONPATH` (or the
official `MUSICFMPATH` integration) and the Python packages above.

### Assets and cache behavior

EDMFormer consumes four representations for each 420-second block:

- MuQ, native 420-second context.
- MuQ, concatenated 30-second windows.
- MusicFM, native 420-second context.
- MusicFM, concatenated 30-second windows.

All audio is decoded/resampled to mono 24 kHz. MuQ explicitly requires 24 kHz
and recommends FP32 because reduced precision can produce NaNs.

Required assets:

| Asset | Size | Integrity/source |
|---|---:|---|
| EDMFormer `model.pt` | 417,979,984 bytes | SHA-256 `1412e207645e9a71adc09777714dd251ce7805cada9bf19518d2e455a977e165`; Git LFS in EDM-98 |
| MusicFM `pretrained_msd.pt` | 1,316,802,088 bytes | SHA-256 `218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded`; Git LFS/Hugging Face |
| MusicFM `msd_stats.json` | about 2.3 KB | EDM-98 repository or MusicFM model repository |
| MuQ `model.safetensors` | 1,333,825,096 bytes | SHA-256 `273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6`; `OpenMuQ/MuQ-large-msd-iter` |
| EDMFormer config | text | `configs/edmformer.yaml` at the pinned EDM-98 commit |

MuQ model revision audited:
[`0562a57814f6f8bbd9fdea0a25921a2fce1a841a`](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/tree/0562a57814f6f8bbd9fdea0a25921a2fce1a841a).

MusicFM model revision audited:
[`4513b38bc25ad1d227b1980819b9691ba97f4d87`](https://huggingface.co/minzwon/MusicFM/tree/4513b38bc25ad1d227b1980819b9691ba97f4d87).

Do not download both MuQ `pytorch_model.bin` and `model.safetensors`; they are
duplicate serializations of the same model at approximately 1.334 GB each.
The minimum teacher asset footprint is approximately 3.1 GB before Python
packages and caches.

MusicFM constructs a Wav2Vec2-Conformer configuration from
`facebook/wav2vec2-conformer-rope-large-960h-ft`; that Hugging Face
configuration is another first-run network/cache dependency even when the
MusicFM checkpoint is local.

### Dataset format and audio access

The package contains 98 JSONL records and split files:

- Train: 78 IDs.
- Validation: 10 IDs.
- Test: 10 IDs.

Each record contains:

```json
{
  "id": "Deezer track ID",
  "labels": [[0.054, "intro"], [35.942, "buildup"], [247.0, "end"]],
  "file_path": "original provenance filename.mp3"
}
```

The functional labels are `intro`, `buildup`, `drop`, `breakdown`, `outro`,
and `silence`. Time points are strictly increasing and terminate in `end`.

The audio is not included. The documented contract is to acquire it
separately, name it `<deezer_id>.<ext>`, and join it to the metadata by ID.
EDM-98 supplies no audio downloader. See:
<https://github.com/25ohms/EDM-98/blob/2dd942f2f9e71ffd826346828eeaba1dd3ece56a/README.md#accessing-the-dataset>.

### License

- Repository code/model-related materials: CC-BY-4.0.
- Packaged metadata and split files: MIT.
- MuQ source code: MIT.
- MuQ released weights: CC-BY-NC-4.0.
- MusicFM source: MIT, with its bundled Hugging Face-derived flash-conformer
  file under Apache-2.0.
- Song recordings: not licensed or distributed by EDM-98.

The noncommercial MuQ weight license is compatible with the owner's stated
private, personal use, but its provenance must remain attached to any derived
teacher/student artifact.

### Target-machine constraints

The EDM-98 CLI supports `--device cpu`, but the published stack contains
roughly 3.1 GB of FP32 model weights and repeatedly evaluates two large
foundation models over 420-second and 30-second windows. No upstream CPU
latency figure is published. It must run as a low-priority offline job and
must never execute in Lumen's audio callback or DMX loop.

## 2. Harmonix Set

### Pinned sources

- Repository: <https://github.com/urinieto/harmonixset>
- Audited commit:
  [`64abeb509429e73d74559fb98e621dac866efea1`](https://github.com/urinieto/harmonixset/tree/64abeb509429e73d74559fb98e621dac866efea1)
- README:
  <https://github.com/urinieto/harmonixset/blob/64abeb509429e73d74559fb98e621dac866efea1/README.md>
- Original requirements:
  <https://github.com/urinieto/harmonixset/blob/64abeb509429e73d74559fb98e621dac866efea1/requirements.txt>

### Format and access

The repository contains 912 records in each of these forms:

- `beats_and_downbeats/*.txt`: tab-separated beat time in seconds, beat
  position in bar, and bar number.
- `segments/*.txt`: boundary time in seconds followed by the functional
  segment label.
- `jams/*.jams`: JAMS 0.3.3 beat, downbeat, segmentation, and metadata.
- `metadata.csv`: title, artist, release, duration, BPM, meter/genre,
  MusicBrainz ID, and AcoustID where available.
- `youtube_urls.csv` and `youtube_alignment_scores.csv`: external candidate
  locations and DTW alignment quality, not audio licenses.

No source audio is distributed. A roughly 1.2 GB mel-spectrogram archive is
linked from Dropbox. It contains its own license and `info.json`; that archive
license must be accepted and recorded separately before use.

### Dependencies and incompatibilities

The annotation text and CSV can be parsed with the Python standard library.
Lumen must not install the historical research environment merely to import
annotations.

The upstream 2019/2020 requirements are:

```text
numpy==1.17.2
madmom==0.16.1
librosa==0.7.0
mutagen==1.42.0
pandas==0.24.2
numba==0.48
Cython==0.29.21
argparse~=1.4.0
tqdm~=4.51.0
joblib~=0.14.1
```

Those pins predate Python 3.12 and conflict with Lumen's NumPy 2.3 floor.
`madmom` and the old NumPy/Numba/Pandas pins are not a viable Python 3.12
installation. The legacy notebook/result reproduction environment, if ever
needed, must be a frozen older container, not a Lumen dependency.

The included `download_youtube_mp3s.py` uses obsolete `youtube-dl`, constructs
shell commands, and warns against its own use. It is not an acceptable Lumen
acquisition mechanism. Audio should be user-supplied and aligned by
fingerprint/duration where permitted.

### License

The repository declares MIT. Audio and the separate mel archive have distinct
rights; the MIT repository license must not be represented as licensing
commercial song recordings.

## 3. CCMusic `song_structure`

### Pinned sources

- Hugging Face dataset:
  <https://huggingface.co/datasets/ccmusic-database/song_structure>
- Audited dataset revision:
  [`be72c4d67e0c99c8b68a37eb1df649c40ea8e4e3`](https://huggingface.co/datasets/ccmusic-database/song_structure/tree/be72c4d67e0c99c8b68a37eb1df649c40ea8e4e3)
- Dataset card at the audited revision:
  <https://huggingface.co/datasets/ccmusic-database/song_structure/blob/be72c4d67e0c99c8b68a37eb1df649c40ea8e4e3/README.md>
- Official access instructions:
  <https://ccmusic-database.github.io/en/download.html>
- Zenodo record 1.1: <https://doi.org/10.5281/zenodo.5676893>

There is also a same-named Hugging Face **model** repository at revision
`09ea9c573b959e1c9c464f93a2a2b63ab78b9f18`. It contains only a README and
is not a downloadable trained structure model. Its MIT tag does not override
the dataset's license.

### Format, size, and access

The current Hugging Face dataset is automatic-gated and requires an
authenticated account after the user personally accepts all gate fields.
Authentication may come from `HF_TOKEN` or the standard cached credential
created by `huggingface-cli login`; the environment variable is not mandatory.
Code must never print or persist either credential.

It reports:

- 300 pop songs in one `train` split.
- Five Xet/LFS-backed Arrow shards. Their current exact sizes are:
  - `data-00000-of-00005.arrow`: 459,036,352 bytes
  - `data-00001-of-00005.arrow`: 437,979,680 bytes
  - `data-00002-of-00005.arrow`: 507,293,416 bytes
  - `data-00003-of-00005.arrow`: 481,762,440 bytes
  - `data-00004-of-00005.arrow`: 435,470,784 bytes
- Current Arrow total: 2,321,542,672 bytes.
- Download size: 2,308,839,939 bytes.
- Hugging Face repository storage: approximately 4.64 GB.
- Audio feature declared at 22,050 Hz.
- A mel image.
- A sequence of `onset_time:uint32`, `offset_time:uint32`, and
  `structure:string`.
- Labels including intro/re-intro, verse, pre-chorus, chorus, post-chorus,
  bridge, interlude, and ending.

The card is internally inconsistent: it describes MP3 audio fields while also
stating that only frame features are provided because of copyright. The
importer must inspect actual decoded feature types and never assume full MP3
content.

Official full-dataset usage declares:

```python
from datasets import load_dataset
load_dataset(
    "ccmusic-database/song_structure",
    name="default",
    split="train",
    token="user-owned-token",
)
```

That route necessarily resolves the large Arrow shards and is not Lumen's
annotation-only route. The dataset-viewer `/rows` route was tested and rejected:
the provider tries to scan a 457,892,166-byte shard before selecting a row and
fails its own 300 MB processing limit. `/first-rows` is useful only as an
authorized-access check because it exposes rows 0 through 99, not all 300.

Lumen instead pins Hugging Face's converted Parquet revision
`6ac1a082ca649072518d9fcd7fbf448a1e844266` and verifies its three-file
inventory before reading. DuckDB 1.5.5 with the pinned `httpfs` extension uses
HTTP byte ranges and projects only the nested `label` column. Parquet metadata
reports 34,197 compressed bytes for that column across the three files. The
query returns all 300 label timelines without materializing any Parquet, Arrow,
audio, mel-image, raw provider response, or media-link file locally. The
credential is passed as a parameter-bound, in-memory DuckDB secret and is never
inserted into SQL text, output, or the manifest.

The pinned converted inventory is:

- `0000.parquet`: 894,708,606 bytes
- `0001.parquet`: 505,166,036 bytes
- `0002.parquet`: 915,092,957 bytes
- Total: 2,314,967,599 bytes

The locally isolated DuckDB `httpfs` extension is 21,570,542 bytes with SHA-256
`887c392b1e49128d11667c81e3698d8b00dfdeb456771acf66d05a0f74f7b7d8`.
The fetcher validates its version, size, and hash before any authenticated
projection.

The resulting local contract is:

```text
sources/ccmusic/authorized-annotations/
  manifest.json
  row-000000.txt
  ...
  row-000299.txt
```

Each text row is `onset_time<TAB>offset_time<TAB>structure`, preserving the
official unsigned integer values in centiseconds. The manifest pins revision,
row index, filename, segment count, and SHA-256 and explicitly declares that
audio/mel fields and links were excluded. The completed export contains 300
rows and 2,918 source segments.

The pinned source has two malformed structure strings (rows 21 and 92) where a
valid numeric TSV record is embedded after a newline. The exporter recovers
only records matching the exact `onset<TAB>offset<TAB>label` grammar and records
both repairs in `source_repairs`; arbitrary multiline content fails closed.
Thirteen other source discontinuities remain: eleven gaps and two overlaps.
Their exact row, segment, endpoints, delta, and kind are preserved in
`timeline_discontinuities` rather than silently changing ground truth.

No Python versions are pinned by CCMusic itself. Lumen's minimal path locks
`huggingface-hub==0.30.1` and `duckdb==1.5.5`; the latter is isolated in the
EDMFormer research environment. A future full-Arrow path would additionally
require:

```text
datasets
huggingface-hub
pyarrow
soundfile and/or the decoder required by the selected datasets release
pillow
```

MP3 fallback decoding requires host `ffmpeg`.

### License and acquisition limitation

The current Hugging Face dataset declares CC-BY-NC-ND-4.0 and noncommercial
use only. The gate further states that the dataset does not own the linked
audio and that rights remain with the creators/channel owners. The website's
complete-database route additionally requires an application by email and
manual evaluation for research use.

Therefore:

- annotation ingestion can be implemented after explicit user acceptance;
- the token must remain outside source control and exported manifests;
- underlying recordings cannot be treated as CC-BY-NC-ND merely because the
  dataset wrapper carries that tag;
- no trained artifact based on this data should be redistributed without a
  separate license review, especially because `ND` and third-party audio are
  involved.

## 4. SALAMI

### Pinned sources

- Public annotations:
  <https://github.com/DDMAL/salami-data-public>
- Audited commit:
  [`8e4f95d18a3ab628c53011fa5a43e9d3be27965d`](https://github.com/DDMAL/salami-data-public/tree/8e4f95d18a3ab628c53011fa5a43e9d3be27965d)
- README:
  <https://github.com/DDMAL/salami-data-public/blob/8e4f95d18a3ab628c53011fa5a43e9d3be27965d/readme.md>
- Optional matching resources:
  [`jblsmith/matching-salami@c93d581563a381cde6353c3cba68d7c95d9b6573`](https://github.com/jblsmith/matching-salami/tree/c93d581563a381cde6353c3cba68d7c95d9b6573)

### Format and access

The audited tree contains:

- 1,359 song-ID annotation directories.
- 2,243 raw annotator text files.
- Parsed uppercase, lowercase, and functional layers for each raw annotation.
- 1,446 metadata rows, including entries with discarded/private flags.
- Source-specific metadata for Codaich, Internet Archive, RWC, and Isophonics.

Each piece has one or two annotator files in normal cases. The raw format is
documented in the included `SALAMI Annotator Guide.pdf`. Parsed files are
plain timestamp/label text and require no third-party Python dependency.

SALAMI does not distribute the audio. Some Internet Archive source material is
independently accessible; other records require separately held audio. The
matching repository supplies candidate YouTube IDs and alignment offsets for
part of the set, but explicitly leaves obtaining audio to the reader.

The optional matching repository declares old, unpinned tooling:

```text
pytube
apiclient
jsonschema
matplotlib
dataset
youtube-dl
mutagen
```

That stack is not needed to import SALAMI and should not enter a production
environment. Its `align_audio.py` also depends on SoX and an external
fingerprinting workflow.

### License

The annotation and metadata repository is CC0/public-domain dedication.
Audio retains the license of its source collection or recording and is not
covered by SALAMI's CC0 dedication.

## 5. SongFormer, SongFormDB, and SongFormBench

### Pinned sources

- GitHub repository: <https://github.com/ASLP-lab/SongFormer>
- Audited GitHub commit:
  [`139b2aa3b14bd1c6d961d0994e9fc975f1ef7fd5`](https://github.com/ASLP-lab/SongFormer/tree/139b2aa3b14bd1c6d961d0994e9fc975f1ef7fd5)
- Requirements:
  <https://github.com/ASLP-lab/SongFormer/blob/139b2aa3b14bd1c6d961d0994e9fc975f1ef7fd5/requirements.txt>
- Inference implementation:
  <https://github.com/ASLP-lab/SongFormer/blob/139b2aa3b14bd1c6d961d0994e9fc975f1ef7fd5/src/SongFormer/infer/infer.py>
- Hugging Face model revision:
  [`a75880ed1b7375ac71860ec6c4fc9c899cf99515`](https://huggingface.co/ASLP-lab/SongFormer/tree/a75880ed1b7375ac71860ec6c4fc9c899cf99515)
- SongFormDB revision:
  [`4fc4b4709ceb020616431b3e9ebce84519a37326`](https://huggingface.co/datasets/ASLP-lab/SongFormDB/tree/4fc4b4709ceb020616431b3e9ebce84519a37326)
- SongFormBench revision:
  [`acd574ecbf666be535b0d051b71936f6ec9956ec`](https://huggingface.co/datasets/ASLP-lab/SongFormBench/tree/acd574ecbf666be535b0d051b71936f6ec9956ec)

The Git submodules at the audited SongFormer commit are the same pinned MuQ
and MusicFM revisions listed in the EDM-98 section.

### Exact official Python environment

Upstream says Python 3.10 and Ubuntu 22.04.1. Its exact requirements file is:

```text
torch==2.4.0
torchaudio==2.4.0
lightning==2.5.1.post0
transformers==4.51.1
accelerate==1.5.2
datasets==3.6.0
tokenizers==0.21.1
huggingface-hub==0.30.1
safetensors==0.5.3
numpy==1.25.0
scipy==1.15.2
scikit-learn==1.6.1
pandas==2.2.3
librosa==0.11.0
audioread==3.0.1
soundfile==0.13.1
pesq==0.0.4
auraloss==0.4.0
nnAudio==0.3.3
julius==0.2.7
soxr==0.5.0.post1
mir_eval==0.8.2
jams==0.3.4
msaf==0.1.80
matplotlib==3.10.1
seaborn==0.13.2
tensorboard==2.19.0
wandb==0.19.8
gpustat==1.1.1
hydra-core==1.3.2
omegaconf==2.3.0
fire==0.7.1
click==8.1.8
einops==0.8.1
einx==0.3.0
x-transformers==2.4.14
x-clip==0.14.4
ema-pytorch==0.7.7
schedulefree==1.4.1
torchmetrics==1.7.1
h5py==3.13.0
pyarrow==19.0.1
pillow==11.1.0
ftfy==6.3.1
regex==2024.11.6
pypinyin==0.54.0
textgrid==1.6.1
pylrc==0.1.2
modelscope==1.27.1
tqdm==4.67.1
loguru==0.7.3
joblib==1.4.2
easydict==1.13
addict==2.4.0
beartype==0.21.0
triton==3.0.0
muq==0.1.0
vmo==0.30.5
gradio
einops
beartype
blessed
PyPDF2
```

The upstream checkpoint downloader additionally imports `requests` but does
not declare it directly; it normally arrives transitively. Lumen's lock should
declare it explicitly.

`pesq`, `vmo`, and `msaf` are source distributions for relevant Python
versions and can require a working compiler/toolchain. MSAF also pulls
`cvxopt`, `enum34`, and other legacy dependencies not obvious in SongFormer's
top-level file. These packages are primarily training/evaluation dependencies,
not necessary for a purpose-built inference worker.

### Critical version/runtime incompatibilities

- Lumen: Python >=3.12, NumPy >=2.3.
- SongFormer: Python 3.10, NumPy ==1.25.0.
- Harmonix historical stack: NumPy 1.17.2 and other Python-3.12-incompatible
  packages.

They must be separate processes/environments connected by versioned JSON or
SQLite job records, never imports across environments.

The SongFormer repository inference worker sets `device = f"cuda:{rank}"`.
Its shell interface is organized around GPU count and processes per GPU.
Training uses Accelerate's single-GPU configuration and NVIDIA environment
variables. Upstream reports 2–4 second whole-song inference on an NVIDIA L40,
not CPU.

PyTorch `2.4.0` from the ordinary Linux PyPI index pulls CUDA 12.1 component
packages on x86-64. On this non-NVIDIA host, install the matching official CPU
wheel explicitly; otherwise many gigabytes of unusable CUDA dependencies may
be installed. `triton` and `gpustat` can be present but provide no functional
benefit without NVIDIA hardware.

The repository notes that newer PyTorch behavior may require changing
MusicFM's load to `torch.load(..., weights_only=False)`. EDM-98 already wraps
this compatibility issue. A SongFormer adapter must do so explicitly and test
checkpoint loading; it must not patch a vendored checkout by hand at runtime.

### Model assets

There are two alternative deployment layouts:

1. Standalone GitHub flow:
   - SongFormer head `SongFormer.safetensors`: 104,468,437 bytes,
     SHA-256 `87f17bfbed37014c6af4314abd9eb6971a94e3a95e9fc70f9e5ee33bdacb487b`.
   - MusicFM MSD checkpoint: 1,316,802,088 bytes.
   - MuQ weights from its Hugging Face cache: 1,333,825,096 bytes.
2. Hugging Face remote-code flow:
   - Combined `model.safetensors`: 2,755,035,132 bytes,
     SHA-256 `8dcabf4ea19973edd51b9e5794004775fa7e8de3ecfa07eb1dbce00f516ce7f7`.
   - Accompanying source/config/stat files.

Do not provision both layouts. The Hugging Face path executes repository
custom code via `trust_remote_code=True`; it must be pinned to the audited
revision and reviewed before loading. The model input is mono 24 kHz audio and
output is timed functional sections.

### SongFormDB

SongFormDB declares CC-BY-4.0 and contains four subsets:

- HX: 712 rule-corrected Harmonix records.
- Ext: 4,314 records.
- Hook: 5,933 records.
- Gem: 4,387 Gemini-annotated multilingual records.

Its JSONL records include IDs, durations, split/subset data, paths, labels,
and YouTube URLs where applicable. The complete repository contains about
24,810 files and reports approximately 262 GB storage, primarily `.npy`
features/mels. The important label files are only approximately:

```text
HX JSONL       0.54 MB
Ext JSONL      3.10 MB
Hook JSONL     2.57 MB
Gem JSONL      4.06 MB
```

The first implementation should download only README, JSONL, and split/label
metadata by explicit allow-list. It should not clone/snapshot all 262 GB.
Audio is not included; the card suggests YouTube sources or audio
reconstruction from mels.

### SongFormBench

SongFormBench declares CC-BY-4.0 and contains:

- 200 HarmonixSet benchmark songs.
- 100 Chinese-pop benchmark songs.
- A 7-class functional scheme plus preserved pre-chorus.

It reports approximately 5.33 GB storage across 911 files, including labels,
mels, and evaluation material. Fetch labels first; acquire mels only for a
specific evaluation test.

### Optional BigVGAN reconstruction

SongFormDB/Bench documents BigVGAN reconstruction from mel spectrograms.
Audited optional upstream:
[`NVIDIA/BigVGAN@7d2b454564a6c7d014227f635b7423881f14bdac`](https://github.com/NVIDIA/BigVGAN/tree/7d2b454564a6c7d014227f635b7423881f14bdac).

BigVGAN is tested upstream on Python 3.10, PyTorch 2.3.1, CUDA 11.8/12.1 and
NVIDIA hardware. A non-fused PyTorch path exists, but CPU reconstruction is
not the intended deployment. BigVGAN is not needed for annotation ingestion,
teacher inference on user-supplied recordings, or student training from
precomputed features. Do not install it initially on this machine.

### License

- SongFormer repository: CC-BY-4.0 according to its README/LICENSE.
- SongFormer Hugging Face card has no structured license value in its current
  API metadata, although its README badge says CC-BY-4.0. Preserve the
  repository declaration and model revision in provenance.
- SongFormDB and SongFormBench: CC-BY-4.0.
- MuQ weights embedded in or required by the stack: CC-BY-NC-4.0.
- MusicFM source/model repository: MIT as declared upstream.
- Referenced YouTube/source audio: separately copyrighted and not conveyed by
  the dataset license.

## Recommended isolated provisioning order

This is the required order for a reproducible implementation.

### Phase 0: preserve and inventory

1. Create and checksum a complete Lumen backup before implementation.
2. Record the current Git diff and untracked training work.
3. Create a machine-readable research lock containing every repository/model
   revision and asset checksum listed in this audit.

### Phase 1: host decoding and asset transport

1. Install `git-lfs`, `ffmpeg`, `libsndfile1`, `build-essential`,
   `pkg-config`, and `cmake`.
2. Run `git lfs install` for the user.
3. Verify compressed-audio decoding and 24 kHz resampling with a synthetic or
   user-owned fixture.

### Phase 2: dependency-free annotation import

1. Import pinned EDM-98 metadata/splits.
2. Import pinned Harmonix text/CSV/JAMS metadata without its legacy
   requirements.
3. Import pinned SALAMI raw/parsed annotations without its matching stack.
4. Fetch only SongFormDB/SongFormBench JSONL labels and metadata.
5. Leave CCMusic in a visible `awaiting_user_license_acceptance` state until
   the user accepts the gate; then fetch its Arrow shards with a user-owned
   token.
6. Normalize all labels into Lumen's independent functional, energy, content,
   timing, confidence, and provenance axes.

### Phase 3: EDMFormer CPU worker

1. Create a dedicated Python 3.12.8 environment outside Lumen's core venv.
2. Install a pinned PyTorch 2.4 CPU build first, with matching Torchaudio.
3. Install EDM-98 at the audited commit with inference dependencies.
4. Install MuQ from the audited commit, not default branch.
5. Check out MusicFM at the audited commit and expose it only to the worker.
6. Fetch one serialization of every required checkpoint into a Lumen-specific
   model cache; verify size and SHA-256 before use.
7. Test cache-warming, offline startup, 30-second input, full-song input,
   corrupt checkpoint behavior, process cancellation, and memory ceiling.
8. Benchmark wall time and peak RSS on the i5-8400 before connecting the
   worker to Lumen jobs.

### Phase 4: SongFormer reference worker

1. Install a current Python 3.10 patch in a second isolated environment.
2. Begin with an inference-only lock derived from the official requirements,
   not the complete training/evaluation list.
3. Install PyTorch/Torchaudio 2.4 CPU wheels explicitly.
4. Pin the SongFormer, MuQ, MusicFM, and Hugging Face model revisions.
5. Select exactly one checkpoint layout and verify all hashes.
6. Run a CPU feasibility test under a hard timeout and memory limit.
7. If CPU performance is unacceptable, retain SongFormer as a dataset/model
   comparison tool and use EDMFormer/cached labels as teachers. Do not block
   Lumen's live operation on it.

### Phase 5: training environment

Only after inference and data normalization pass:

1. Create a third, disposable Python 3.10 training environment using the full
   official SongFormer list.
2. Resolve and lock native-source packages (`pesq`, `vmo`, `msaf`, `cvxopt`)
   separately.
3. Keep W&B disabled/offline unless the user explicitly chooses otherwise.
4. Train only the small CPU-capable causal student locally. Full SongFormer
   retraining is not a realistic i5-8400 workload.

### Phase 6: Lumen process boundary

1. Communicate with workers through queued jobs and versioned result files or
   database records.
2. Never import Torch, Transformers, MuQ, MusicFM, or SongFormer in the live
   audio/DMX process.
3. Make missing teachers nonfatal: the existing live analyzer remains the
   first-play fallback.
4. Cache normalized timelines by stable recording identity and model revision.

## Required verification gates

Provisioning is not complete until all applicable gates pass:

1. Every dataset importer validates counts, monotonic boundaries, end markers,
   duration bounds, duplicate IDs, and source provenance.
2. Every download verifies pinned revision, byte size, and SHA-256 where
   available.
3. All models start with network disabled after cache warming.
4. Corrupt/missing assets produce explicit health errors rather than heuristic
   predictions labeled as neural output.
5. EDMFormer and SongFormer workers can be terminated without leaving child
   processes or locks.
6. Full-song reconstruction from Lumen's 60-second capture segments is
   sample-contiguous before teacher analysis.
7. Train/validation/test splitting is by stable song/recording identity, never
   by 60-second segment.
8. Teacher jobs do not alter DMX timing, audio callback latency, feedback
   persistence, or shutdown behavior.
9. Dataset/model status and license/access blockers are visible in the Lumen
   interface.
10. The full Lumen test suite passes three complete times after integration,
    followed by three independent code/runtime review passes as requested.

## Implementation decision

Use all of the audited sources, but not in the same way:

- EDM-98/EDMFormer: primary EDM structural teacher.
- Harmonix Set: primary broad beat/downbeat/functional annotation source.
- SALAMI: broad hierarchical structure and vocabulary source.
- CCMusic: gated supplemental pop annotation source after explicit access.
- SongFormDB/Bench: selectively downloaded labels and evaluation material.
- SongFormer: heavyweight offline reference/teacher only if the CPU
  feasibility gate passes.
- Lumen's captures, context annotations, preferred-action sequences, feedback,
  and DMX state: the separate personalization/choreography dataset.

This preserves the intended two-model architecture: public research teaches
Lumen what is happening musically; local Lumen sessions teach what the lights
should do about it.

## Provisioning result — 2026-07-31

The dependency and asset layer described above is now provisioned on the target
i5-8400 machine. `config/research/research-lock.json` is the machine-readable
source of truth and `scripts/setup-research` is the idempotent provisioning and
verification entry point. Fully resolved transitive package locks are stored in
`config/research/edmformer-freeze.txt` and
`config/research/songformer-inference-freeze.txt`.

### Installed isolated environments

- EDMFormer: Python 3.12.13, NumPy 2.2.6, Torch 2.4.0+cpu, Torchaudio
  2.4.0+cpu, Torchvision 0.19.0+cpu. MuQ and EDM-98 are editable installs from
  the exact pinned local revisions.
- SongFormer inference: Python 3.10.20, NumPy 1.25.0, Torch 2.4.0+cpu,
  Torchaudio 2.4.0+cpu, and Torchvision 0.19.0+cpu. The actual model import
  exposed an undocumented runtime dependency on MSAF; `msaf==0.1.80` and its
  resolved dependency tree are now included in the freeze.
- `pip check` reports no broken requirements in either environment.
- CUDA is unavailable and both verification paths assert CPU-only execution.

### Verified model assets

| Asset | Bytes | SHA-256 |
|---|---:|---|
| EDMFormer `model.pt` | 417,979,984 | `1412e207645e9a71adc09777714dd251ce7805cada9bf19518d2e455a977e165` |
| MusicFM MSD checkpoint | 1,316,802,088 | `218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded` |
| MusicFM statistics | 2,277 | `c36c61ab10ca4d2e7fdfefc3fcc15205316bec276a06a47baa3641a62c546f22` |
| MusicFM Wav2Vec2 architecture config | 2,239 | `7a63cb5706c9a37483f1973a3c226d54eb504ce15cf62cb52637019540c8a75d` |
| MuQ large MSD safetensors | 1,333,825,096 | `273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6` |
| MuQ config | 3,133 | `237335ee27d8fb951ce778701a12a79e06c51ae636dd786f97e45f51ce532543` |
| SongFormer EMA head | 104,468,437 | `87f17bfbed37014c6af4314abd9eb6971a94e3a95e9fc70f9e5ee33bdacb487b` |

The Wav2Vec2 config is required because MusicFM calls Transformers for the
architecture definition even though it initializes no Wav2Vec2 pretrained
weights and subsequently loads the complete pinned MSD state dict. It has been
pinned and cached explicitly so offline startup succeeds.

### Selective dataset result

- Pinned EDM-98, Harmonix, and SALAMI annotation repositories are present.
- SongFormDB label-only manifests contain 712 HX, 4,308 Ext, 9,498 Hook, and
  4,137 Gem records.
- SongFormBench contains 300 label records.
- All five JSONL manifests parse completely, and every locked file passes exact
  byte-size and SHA-256 validation.
- The reported 262,081,980,546-byte SongFormDB feature repository and
  copyrighted song audio were deliberately not downloaded.
- The CCMusic gate was accepted by the user. A standard cached Hugging Face
  login is currently valid; `HF_TOKEN` remains an optional non-persisted
  override. No credential value is recorded in Lumen's files or output.
- `scripts/setup-research ccmusic-metadata` fetches only four small repository
  metadata files. `scripts/setup-research ccmusic-annotations` performs the
  authenticated, revision-pinned Parquet range projection described above and
  exports only the 300 label timelines. Neither command permits Arrow, audio,
  or mel downloads.

### Runtime verification

`scripts/setup-research verify-models` performs real model work rather than
only checking that files exist:

1. Strictly loads the 26,115,201-parameter EDMFormer checkpoint.
2. Loads MuQ (333,401,472 parameters) and MusicFM (328,932,480 parameters) with
   network access disabled.
3. Runs a deterministic two-second CPU tensor through both feature teachers.
   Both return aligned `(1, 50, 1024)` embeddings.
4. Fuses the four production feature axes into `(1, 50, 4096)` and runs the
   EDMFormer head, producing `(1, 16, 128)` function logits and `(1, 16)`
   boundary logits.
5. Strictly reconstructs the SongFormer EMA checkpoint and runs its head on
   CPU, producing the same logit dimensions.

The combined persistent EDMFormer/MuQ/MusicFM load peaked at approximately
4,299,080 KiB RSS. This establishes feasibility for an isolated offline worker;
it does not imply that these teachers should ever run in Lumen's live DMX
process.

Three consecutive runs of `scripts/setup-research verify` completed
successfully on 2026-07-31 before the authorized CCMusic export. Each pass
rechecked all repository revisions,
asset sizes and hashes, both package freezes and `pip check`, all selective
label manifests, offline feature-model inference, strict EDMFormer and
SongFormer checkpoint loading, and CPU head inference. The CCMusic gate and
label-only export have since been completed and are included in subsequent
verification. A subsequent full dependency/model pass succeeded with DuckDB
1.5.5 in the frozen EDMFormer environment and revalidated the 300-row CCMusic
package. The complete Lumen unit suite then passed three consecutive 110-test
runs.

### Commands

```text
scripts/setup-research provision
scripts/setup-research verify
scripts/setup-research verify-models
scripts/setup-research status
scripts/setup-research ccmusic-status
scripts/setup-research ccmusic-metadata
scripts/setup-research ccmusic-annotations
```

`status` is the lighter non-mutating readiness report. `verify` is intentionally
heavy because it includes actual checkpoint inference and may use about
4.2 GiB of RAM.
