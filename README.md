# EchoCLIP: A Multimodal Foundation Model For Echocardiography

EchoCLIP is a multimodal foundation model for echocardiography. It is finetuned from CLIP weights on a dataset of >1M pairs of echocardiogram images and their associated expert interpretation text. It can be used for semantic search amongst echo videos as well as zero-shot prediction on a wide range of clinically relevant tasks. For more details, see our paper:

(link will be added once preprint is released)
<!-- [Multimodal Foundation Models For Echocardiogram Interpretation](https://arxiv.org/abs/) -->

## Quickstart

This repo contains example code for loading and using EchoCLIP and its long-context variant, EchoCLIP-R. To get started, clone this repo and navigate into it. Then, create a new `conda` environment and install the required packages:

```
git clone https://github.com/echonet/echo_CLIP
cd echo_CLIP
conda env create -n echo-clip
conda activate echo-clip
python -m pip install -r requirements.txt
```
You should now be able to run `embedding_example.py` and `zero_shot_example.py`.

## Probe training on frozen EchoCLIP embeddings

You can train attentive probes (from `attentive_pooler.py`) on top of frozen video embeddings using `probe_train_mr.py`.
The script supports initializing multiple probe heads from a YAML config and training them in parallel (same batches, separate optimizers).

Expected CSV columns:

- `video_path`
- `label` (text or numeric class compatible with `zero_shot_mr.py` normalization)
- `subject_id` (optional but recommended for subject-level validation metrics)

Grid-search config example:

- `configs/probe_grid_mr.yaml`

Example run with YAML config:

```bash
python probe_train_mr.py --config configs/probe_grid_mr.yaml
```

Single-head CLI run (no YAML):

```bash
python probe_train_mr.py \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/val.csv \
  --video-path-col video_path \
  --label-col label \
  --subject-id-col subject_id \
  --device cuda \
  --epochs 20 \
  --batch-size 8 \
  --output-dir probe_mr_outputs
```

Optional Weights & Biases logging:

```bash
python -m pip install wandb
python probe_train_mr.py --train-csv /path/to/train.csv --val-csv /path/to/val.csv --use-wandb
```

Artifacts written to `--output-dir`:

- `checkpoints/latest.pt`
- `checkpoints/epoch_XXX.pt` (per-epoch probe checkpoints)
- `checkpoints/best.pt` (best by subject-level validation accuracy)
- `probe_training_history.json`
- `probe_head_summary.json` (per-head final/best subject accuracy and selected best head)
- per-class metric CSVs and confusion plots from final validation pass

## Repo contents

* `embedding_example.py` demonstrates how to load EchoCLIP-R's weights and use them to calculate the similarity between an example echocardiogram and example report text.
* `zero_shot_example.py` demonstrates how to load EchoCLIP's weights and use them to perform zero-shot pacemaker identification and zero-shot ejection fraction prediction.
* `utils.py` contains implementations of our methods for performing zero-shot binary classification and zero-shot regression. The functions used in `zero_shot_example.py` are defined in this file. The prompts we use for the zero-shot tasks in our paper are all available here. Additionally, this file contains regexes for cleaning and preparing report text before it is tokenized.
* `template_tokenizer.py` contains the implementation of our custom echocardiography report tokenizer, which is designed to compress Cedars-Sinai echo reports into a small number of tokens.
* `template_vocab.txt` contains a vocabulary of 770 words and phrases constructed from the template file our cardiologists use to create their reports. This vocabulary is used by our template tokenizer to efficiently tokenize long reports.
* `blank_wordpiece.tokenizer` is a default config file for initializing a WordPiece tokenizer using HuggingFace's `tokenizers` library. We use it to initialize our custom tokenizer.
