<div align="center">

# Learning Localized Visual Representations from Radiology Reports for Chest X-ray Vision–Language Models

[![Paper](https://img.shields.io/badge/Paper-arXiv-blue.svg)](#-citation)   
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://github.com/mk-runner/CheXLoc)   
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/MK-runner/CheXLoc)   
[![Code](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/mk-runner/CheXLoc)

</div>

---

## 📢 News
- **2025-03-01** &nbsp; Release [**generated-radiology-reports**](https://huggingface.co/MK-runner/CheXLoc) — **labels** = reference reports, **report** = generated reports  


---

## 🔍 Overview

CheXLoc learns localized visual representations for chest X-ray vision–language models using paired radiographs and radiology reports, **without pixel-wise spatial annotations**.

The framework combines:

1. **Global image–report alignment**, which preserves holistic cross-modal correspondence.
2. **Entity-guided local patch–token alignment**, which encourages correspondence between image patches and clinically informative report tokens.
3. **Entity-guided token importance estimation**, which distinguishes the relative importance of clinical entities such as anatomy, observations, negation, and temporal information.

All token-level supervision is derived deterministically from radiology reports. No manually annotated spatial labels are used during pretraining.

---


## ⚙️ Requirements

We recommend using a dedicated Conda environment.

```bash
conda create -n chexloc python=3.9
conda activate chexloc

pip install -r requirements.txt
```

The main dependencies include:

```text
transformers==4.43.3
radgraph==0.1.18
torch==2.3.1+cu118
```

Please refer to `requirements.txt` for the complete dependency list.

---

# 📦 Checkpoints and Generated Results

The following pretrained models and evaluation checkpoints are required for reproducing the experiments.

| Component  | Checkpoint                                 | Purpose                      |
| ---------- | ------------------------------------------ | ---------------------------- |
| CheXLoc    | `https://huggingface.co/MK-runner/CheXLoc`                      | CheXLoc pretrained model     |
| RADDINO   | `microsoft/rad-dino`                       | Chest X-ray image encoder    |
| CXR-BERT   | `microsoft/BiomedVLP-CXR-BERT-specialized` | Clinical text encoder        |
| DistilGPT2 | `distilbert/distilgpt2`                    | Report generation decoder    |
| CheXbert   | `https://huggingface.co/StanfordAIMI/RRG_scorers/blob/main/chexbert.pth`                             | Report-generation evaluation |
| RadGraph   | `https://huggingface.co/StanfordAIMI/RRG_scorers/blob/main/modern-radgraph-xl.tar.gz`                                 | Report-generation evaluation |
| BERT       | `google-bert/bert-base-uncased`                        | BERTScore calculation        |

Place the downloaded checkpoints under:

```text
checkpoints/
├── chexloc/
├── raddino/
├── cxr_bert/
├── distilgpt2/
├── chexbert/
└── radgraph/
```

Then configure the corresponding paths in the provided scripts.

---

# 📂 Datasets

## Medical Images and Reports

### MIMIC-CXR

MIMIC-CXR training set is used for CheXLoc pretraining and the primary report-generation evaluation.

The official MIMIC-CXR training split contains paired chest radiographs and radiology reports.

Please obtain the dataset from:

* [PhysioNet](https://physionet.org/content/mimic-cxr/)

PhysioNet credentialing and data-use requirements apply.

---

### CheXpert Plus

CheXpert Plus is used for external report-generation evaluation.

The dataset contains paired chest radiographs and radiology reports collected from Stanford Health Care.

Please obtain the dataset from the [official source](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1) and organize the images and reports according to the provided evaluation metadata.

---

### ReXGradient

ReXGradient is a multi-institutional chest radiograph dataset used to evaluate report generation under external distribution shifts.

The evaluation follows the official train/validation/test splits.

Please obtain the dataset from:
* [HuggingFace](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K)

---

### IU X-ray

IU X-ray is used for zero-shot report-generation evaluation.

The images can be obtained from:

* [OpenI / IU X-ray](https://openi.nlm.nih.gov/)

---

## Downstream Datasets

CheXLoc representations are further evaluated on downstream chest X-ray disease classification and lesion segmentation tasks.

Please refer to the corresponding dataset providers and the dataset-specific preprocessing scripts under:

```text
downstream/datasets.py
```

---
# 📝 Pretraining on the MIMIC-CXR Training set
```bash
cd script

bash mimic-pretraining.sh
```


# 📝 Report Generation

CheXLoc can be evaluated for chest X-ray report generation on multiple datasets.

The evaluation includes:

* MIMIC-CXR
* CheXpert Plus
* ReXGradient
* IU X-ray

The primary report-generation metrics include:

* BLEU
* SembScore
* RadGraph F1
* 1/RadCliQ-V1
* RATEScore
* GREEN

---

## 1. Generate Reports


```bash
cd script

bash mimic-report-generation.sh
```

Generated reports are saved under:

```text
results/
├── MIMIC-CXR/
    └──generated reports
├── CheXpert-Plus/
    └──generated reports
├── ReXGradient/
    └──generated reports
└── IU-Xray/
    └──generated reports
```

Each generated report should be paired with its corresponding reference report and study/image identifier.

---

## 2. Evaluate Generated Reports

Run:

```python
args = {
    'chexbert_path': "/home/miao/data/dataset/checkpoints/chexbert.pth",
    'bert_path': "/home/miao/data/dataset/checkpoints/google-bert/bert-base-uncased",
    'radgraph_path': "/home/miao/data/dataset/checkpoints/radgraph",
    'ckpt_zoo_dir': '/home/miao/data/dataset/checkpoints'  # The directory must contain `chexbert_path`, `radgraph_path`, and `bert_path`. See `📦 Checkpoints and Generated Results` for details.

}
path = 'https://huggingface.co/MK-runner/CheXLoc/blob/main/checkpoints/mimic-cxr/report%20generation/2025_12_07_15_best/generated_reports/11-12-2025_13-57-49.csv'
df = pd.read_csv(path)
refs = df['reference_report'].values.reshape(-1).tolist()
hyps = df['generated_report'].values.reshape(-1).tolist()
scores = compute_ce_scores(hyps=hyps, refs=refs, local_ckpt_zoo=args['ckpt_zoo_dir'])
green_score = compute_green_scores(hyps=hyps, refs=refs, local_ckpt_zoo=args['ckpt_zoo_dir'], file_name=f'temp')
```

The evaluation script computes the metrics used in the paper.

A typical output is:

```text
BLEU-2       : ...
BLEU-4       : ...
SembScore     : ...
RadGraph  : ...
1/RadCliQ-V1 : ...
RATEScore    : ...
GREEN        : ...
```

For external datasets:

```bash
cd script

bash chexpert-plus-report-generation.sh

bash rexgradient-report-generation.sh
```

---

## 3. Report-Generation Evaluation Protocol

For fair comparison with existing methods, the evaluation uses the corresponding official test subsets and evaluation protocols.

For the efficiency comparison, models are evaluated using the same hardware and inference protocol. The reported inference time includes preprocessing, model inference, and report generation.

---

# 🩺 Disease Classification

CheXLoc pretrained visual representations are evaluated on downstream chest X-ray disease classification.

The classification experiments investigate whether the visual representations learned from report supervision transfer to downstream disease classification, particularly when only limited downstream supervision is available.

---

## 1. Prepare the Dataset

See the Data availability section in the manuscript.

---

## 2. Zero-shot disease classification

```bash
python main_zero_shot_classification.py
```


---

## 3. Supervised disease classification

```bash
python main_classification_segmentation.py
```


---

All compared methods use the same downstream data split and training protocol.

---

# 🎯 Lesion Segmentation

```bash
python main_classification_segmentation.py
```


---

# 🖼️ Image Dependence

Image dependence evaluates whether a report-generation model actually relies on visual information from the input radiograph.

We use a **blank-radiograph perturbation** protocol.

For each test image, the original radiograph is replaced by a blank image while keeping the model configuration unchanged.

The difference in report-generation performance measures the model's dependence on the visual input.

---
## Generated Reports Using Original and Blank Images

```bash
% original image
python main_perturbations.py --phrase inference --data_name rexrank-mimic --test_ckpt_path https://huggingface.co/MK-runner/CheXLoc/blob/main/checkpoints/mimic-cxr/report%20generation/2025_12_07_15_best/checkpoints/epoch%3D14-step%3D56295_temp.ckpt

% blank image
python main_perturbations.py --phrase inference --data_name rexrank-mimic-blank --test_ckpt_path https://huggingface.co/MK-runner/CheXLoc/blob/main/checkpoints/mimic-cxr/report%20generation/2025_12_07_15_best/checkpoints/epoch%3D14-step%3D56295_temp.ckpt
```

## Evaluation
Modify `main_visual_dependence.py` to specify the local paths to the reports generated from the original and blank images.

```bash
python main_visual_dependence.py
```


The image-dependence score is computed as the difference in:

```text
Δ(1/RadCliQ-V1)
```

between the original and blank-image conditions.

A larger decrease after replacing the radiograph indicates greater dependence of the generated report on the visual input.

The blank-image experiment is intended as a perturbation-based evaluation rather than a training condition.

---

# 📍 Localization Awareness

Localization awareness evaluates whether model predictions are sensitive to image regions corresponding to specific abnormalities.

The evaluation uses lesion-region perturbation.

For a disease or lesion category \(c\), a corresponding spatial mask \(M_c\) is used to identify the relevant image region.

We compare the model response under:

```text
Original image:
x

Lesion-only image:
x ⊙ M_c

Lesion-removed image:
x ⊙ (1 - M_c)
```

The exact perturbation protocol follows the implementation provided in:

```bash
% obtain generated reports from lesion-only and lesion-removed images

%% lesion-only
python main_perturbations.py --phrase inference --data_name local-aware --local_aware lesion-only --test_ckpt_path https://huggingface.co/MK-runner/CheXLoc/blob/main/checkpoints/mimic-cxr/report%20generation/2025_12_07_15_best/checkpoints/epoch%3D14-step%3D56295_temp.ckpt

%% lesion-removed
python main_perturbations.py --phrase inference --data_name local-aware --local_aware lesion-removed --test_ckpt_path https://huggingface.co/MK-runner/CheXLoc/blob/main/checkpoints/mimic-cxr/report%20generation/2025_12_07_15_best/checkpoints/epoch%3D14-step%3D56295_temp.ckpt
```
## Run Localization Evaluation

Compute sufficiency, necessity, and LAGE scores:
```bash
python main_classification_segmentation.py
```

---




# 📜 Citation

If you use CheXLoc, its pretrained models, generated reports, or evaluation code, please cite:

```bibtex
@article{CheXLoc,
  author  = {Liu, Kang and ...},
  title   = {Learning localized visual representations from radiology reports for chest X-ray vision--language models},
  journal = {arXiv},
  year    = {2026}
}
```

---


# 🙏 Acknowledgements

We thank the authors and maintainers of the datasets, models, and evaluation tools used in this work, including MIMIC-CXR, CheXpert Plus, ReXGradient, IU X-ray, RAD-DINO, CXR-BERT, CheXbert, RadGraph, RaTEScore, RadCliQ, and GREEN.

We also thank the open-source medical vision–language community and the authors of the statistical analysis code used in this work. The statistical analysis code is available [here](https://gist.github.com/rajpurkar/2cc9f61c8f4b0b56a52d5959a47ea7f8).

