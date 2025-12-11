## FORSAtic (forca-hack) — quickstart (Colab)

This repo contains preprocessing + training scripts for the hackathon.

### Folder layout (expected)
- **datasets (local/Drive)**:
  - `clients-satisfaction/train.csv`
  - `clients-satisfaction/test_file.csv`
  - `call-center/train.csv`
  - `call-center/test.csv`
- **code**:
  - `forca-hack/src/`
  - `forca-hack/scripts/`

### Colab steps (Challenge 1 — Social comments baseline)

#### 0) (Once) clone your repo

```bash
!git clone <YOUR_GITHUB_REPO_URL>
%cd <REPO_FOLDER>
```

#### 1) (Every run) pull latest code + install deps

```bash
!git pull
!pip -q install -r forca-hack/requirements.txt
```

#### 2) Run baseline training + create submission

Adjust the CSV paths if your Drive folder differs.

```bash
!python forca-hack/scripts/train_social_tfidf.py \
  --train_csv clients-satisfaction/train.csv \
  --test_csv clients-satisfaction/test_file.csv \
  --out_dir forca-hack/outputs
```

#### 3) Upload to Kaggle
- The generated file is: `forca-hack/outputs/submission_social_tfidf.csv`


