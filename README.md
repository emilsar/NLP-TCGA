# NLP-TCGA

Coaching evaluation and reference notes for the **AI Campus @ Cedars-Sinai — Project 7: NLP in Cancer Pathology Reports**.

This repo collects what's needed to decide whether to take on the project as a community-college coaching effort, and documents the corpus, labels, and two existing reference implementations.

---

## Course notebooks — cancer-type classification

Open any notebook directly in Google Colab. The first cell downloads this repo (notebooks **and** data, ~100 MB) so everything just runs — no setup, no accounts, no local install.

| # | Notebook | What it covers | Colab |
|---|---|---|---|
| 1 | `1-Compile_Dataset` | Joining report text to cancer-type labels by patient barcode | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/NLP-TCGA/blob/main/notebooks/cancer_type/1-Compile_Dataset.ipynb) |
| 2 | `2-Train_Val_Test_Split` | Train/validation/test splitting; checking label balance | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/NLP-TCGA/blob/main/notebooks/cancer_type/2-Train_Val_Test_Split.ipynb) |
| 3 | `3-Bag_of_Words` | Tokenization, stopwords, turning text into a count matrix | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/NLP-TCGA/blob/main/notebooks/cancer_type/3-Bag_of_Words.ipynb) |
| 4a | `4-BoW_LR` | Logistic regression; per-class performance; top words per cancer type | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/NLP-TCGA/blob/main/notebooks/cancer_type/4-BoW_LR.ipynb) |
| 4b | `4-BoW_RF` | Random forest on the same features, for comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/NLP-TCGA/blob/main/notebooks/cancer_type/4-BoW_RF.ipynb) |
| 5 | `5-BoW_ML` | scikit-learn pipelines and hyperparameter search | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilsar/NLP-TCGA/blob/main/notebooks/cancer_type/5-BoW_ML.ipynb) |

Notebooks are adapted from [guilopgar/AI-Campus-Project-7-NLP](https://github.com/guilopgar/AI-Campus-Project-7-NLP); the only change is an added setup cell at the top of each.

### Data in this repo

```
data/
├── corpus/TCGA_Reports.csv                    9,523 reports (barcode + text)
└── cancer_type/
    ├── tcga_patient_to_cancer_type.csv        labels, from TCGA clinical metadata
    ├── tcga-tumor-types.csv                   code → full cancer name
    ├── tcga_reports_cancer_type.csv           output of notebook 1
    └── {train,val,test}_tcga_reports_...csv   output of notebook 2
```

All three layers are checked in, so any notebook runs standalone without running the ones before it.

**One column is the feature** (`text`, the whole pathology report) and **one column is the target** (`cancer_type`, 32 classes).

---

## The project

**Source page:** https://cedars.nationalcampus.ai/project/32/

**Goal.** Develop NLP techniques to extract diagnostic information from cancer pathology reports — ultimately to support patient diagnosis, treatment selection, and trial matching.

**Tasks (per the project description).**
- Cancer-type classification from report text.
- TNM-stage classification (T, N, M).
- Optional: extraction of related fields and downstream prognostic use.

**Method progression on the menu.** TF-IDF / bag-of-words → classical ML (SVM, LR, RF) → BERT and clinical variants → decoder-only LLMs and prompt engineering.

**Libraries called out:** NLTK, spaCy, scikit-learn, Hugging Face Transformers.

---

## Source data

Hosted by the Tatonetti Lab on AWS S3:
http://tatonettilab-resources.s3-website-us-west-1.amazonaws.com/?p=tcga-path-reports/

| File | Compressed | Uncompressed | Contents |
|---|---:|---:|---|
| `aws_response.tar.gz` | 739 MB | 2.2 GB | 24,213 Python pickle files — one per page — containing raw AWS Textract `AnalyzeDocument` responses for **9,546 patients / reports**. |
| `imgs_for_aws.zip` | 22.5 GiB | 36.9 GB | 25,478 JPGs — one per page (3301×5100 ~300 dpi) for **9,848 patients**. Heavily redacted (PHI masked with black bars in the source PDF). |

### Pickle structure
Filenames: `aws_response/TCGA-<barcode>.<report-uuid>_Page_<N>_response.p`

Each `.p` is a Python `dict`:
```text
DocumentMetadata, AnalyzeDocumentModelVersion, ResponseMetadata,
Blocks: [
  {BlockType: 'PAGE',  Geometry, Relationships},
  {BlockType: 'LINE',  Text, Confidence, Geometry: {BoundingBox, Polygon}},
  {BlockType: 'WORD',  Text, Confidence, Geometry},
  ...
]
```

To reconstruct one report's text:
```python
import pickle, glob, re
pages = sorted(
    glob.glob(f'aws_response/TCGA-XX-XXXX.<uuid>_Page_*_response.p'),
    key=lambda p: int(re.search(r'_Page_(\d+)_', p).group(1))
)
text = []
for path in pages:
    resp = pickle.load(open(path, 'rb'))
    text.extend(b['Text'] for b in resp['Blocks'] if b['BlockType'] == 'LINE')
report_text = '\n'.join(text)
```

### Coverage gap
The image zip has 302 reports / 1,259 pages with no Textract pickle — likely OCR failures. If you re-OCR, you can recover ~3% more reports.

---

## Labels are not in the S3 files

The pickles only carry text + geometry. **Labels come from TCGA clinical metadata** (GDC Data Portal / Xena Browser), joined to text via patient barcode (`TCGA-F2-6879`). Available label types:

| Label | Type | Use |
|---|---|---|
| Cancer type (`project_id`: TCGA-LUAD, TCGA-BRCA, …, 33 codes) | multi-class | classification |
| Histologic subtype | multi-class | classification |
| **TNM stage** (pT, pN, pM, overall I–IV) | ordinal / multi-class | classification |
| Tumor grade | ordinal | classification |
| Vital status (alive/dead) | binary | classification |
| `days_to_death` / `days_to_last_follow_up` | continuous, right-censored | **survival regression** |
| Recurrence | binary + time | survival |
| Treatments (radiation y/n, drugs) | multi-label | classification |

Cancer-type and TNM labels are essentially re-extracting what's already written in the report. **Survival** is the genuinely prognostic target, and right-censoring matters.

---

## Reference implementation 1 — guilopgar (teaching)

**Repo:** https://github.com/guilopgar/AI-Campus-Project-7-NLP

Designed as a curriculum walkthrough.

### Tasks
- Cancer-type classification (33 classes).
- TNM staging — three independent classifiers for T, N, M.

### Methods
| Task | Methods |
|---|---|
| Cancer-type | Bag-of-Words → Logistic Regression, Random Forest, generic ML sweep |
| TNM | Bag-of-Words + LR baseline → Clinical-BigBird (BERT, 4096-token context) fine-tuned per axis → zero-shot LLM prompting with a JSON system prompt |

### Data shipped (already cleaned)
```
data/
├── corpus/
│   └── TCGA_Reports.csv               ← one row per patient: barcode + full text
├── cancer_type/
│   ├── tcga_patient_to_cancer_type.csv
│   ├── tcga_reports_cancer_type.csv
│   └── {train,val,test}_tcga_reports_cancer_type.csv
└── tnm_stage/
    ├── TCGA_{T14,N03,M01}_patients.csv
    ├── tcga_reports_tnm_stage.csv
    └── {train,val,test}_tcga_reports_tnm_stage.csv
```

**OCR and wrangling are pre-solved** — students start from `(text, label)` CSVs.

### Code layout
```
code/
├── cancer_type/     1-Compile_Dataset → 2-Split → 3-BoW → 4-BoW_LR/RF → 5-BoW_ML
├── tnm_stage/       1-Compile_Dataset → 2-Split → 3-BoW_LR → 4-BERT_T/N/M → 5-BERT_Eval → 6-LLM_TNM → 7-LLM_Eval
├── utils.py
└── llm_utils.py
```

### What's missing relative to the project description
No SVM (despite being on the menu), no survival/prognosis, no use of the page images, no external validation.

---

## Reference implementation 2 — tatonetti-lab (research)

**Repo:** https://github.com/tatonetti-lab/tnm-stage-classifier
**Preprint:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10327265/

The published paper that the teaching repo is built around. Method is named **BBTEN** (BigBird TNM Extraction from Notes).

### Scope
- TNM only (no cancer-type).
- Uses the **same** `TCGA_{T14,N03,M01}_patients.csv` label files as the guilopgar repo.

### Methods
- BigBird BERT fine-tuned on TCGA, per-axis.
- **Llama-3-8B-Instruct fine-tuned with QLoRA** (4-bit quantization, LoRA adapters via PEFT, bitsandbytes).

### External validation
~8,000 pathology reports from **Columbia University Medical Center (CUIMC)**. Reported AU-ROC 0.815–0.942 out-of-distribution — the headline result.

### Released artifacts
HuggingFace models:
- https://huggingface.co/jkefeli/CancerStage_Classifier_T
- https://huggingface.co/jkefeli/CancerStage_Classifier_N
- https://huggingface.co/jkefeli/CancerStage_Classifier_M

### Code layout
```
T14_TumorSize/
├── TCGA_Train/      train_allsubtypes_T14.py + eval_metrics_multi_updated.py + T14_example.sh
├── TCGA_Test/
└── CUIMC_Test/      testbestmodel_CUIMC_T14.py + eval_metrics_multi_testset_cuimc.py
N03_RegionalLymphNodeInvolvement/   (same shape)
M01_DistantMetastasis/              (same shape)
Llama-3-ft/         llama3-finetune.py (QLoRA training script)
TCGA_Metadata/      label CSVs + cancer-type binary pickle
TCGA_Pathology_Reports/  TCGA_Reports.csv.zip (the same corpus as guilopgar)
Demo/               minimal end-to-end demo using the held-out T14 test set
```

`.py` scripts and `.sh` launchers — assumes a multi-GPU environment.

---

## guilopgar vs tatonetti-lab — at a glance

| | guilopgar (teaching) | tatonetti-lab (research) |
|---|---|---|
| Purpose | Curriculum | Paper |
| Tasks | Cancer-type **+** TNM | TNM only |
| Methods | BoW + LR/RF; Clinical-BigBird; zero-shot LLM | BigBird fine-tuned + **Llama-3-8B QLoRA** |
| Datasets | TCGA only | TCGA + **CUIMC external validation** |
| Output | Jupyter notebooks | `.py` scripts, HuggingFace models, paper |
| Reproducibility target | Laptop / single GPU | Multi-GPU, bitsandbytes, PEFT |
| Headline metric | (none) | AU-ROC 0.815–0.942 OOD |

**Ambition:** tatonetti-lab is more ambitious (external validation, LLM fine-tuning, paper, public models). guilopgar is broader in *coverage* (adds cancer-type + the BoW pedagogical progression) but shallower per method.

**Challenge:** tatonetti-lab is harder to *execute* (QLoRA, bitsandbytes, hospital data access). guilopgar is harder to *teach* (walking students from TF-IDF to transformers to prompt engineering in one course).

**Relationship:** they're a coursebook and a senior thesis pointing at each other. Same TCGA labels.

---

## Pedagogical framing for community-college coaching

- **guilopgar = ground floor.** Pre-cleaned data; students go straight to modeling. Good for students who'd disengage during 30–50% data wrangling.
- **tatonetti-lab = stretch / capstone.** Replicate BigBird numbers → try Llama-3 QLoRA recipe → external validation if hospital data accessible.
- **Survival prediction is the natural extension.** Neither reference repo does it. Right-censored data; clean binary 5-yr survival via filter-then-classify, or full Cox/Kaplan-Meier.
- **Reframe the deliverable as a tool**, not a metric. "Build something that takes a pathology PDF and emits a structured diagnosis a trial-matcher could consume" demos better than "predict TNM stage."
- **Skip the 24 GB image zip** unless going multimodal or re-OCRing the 302 reports the Textract run missed.

### Watchouts
- Pathology jargon is steep; budget an orientation session.
- Right-censored survival labels need an explicit teaching moment.
- The "isn't this just data wrangling?" worry is real for the *raw* corpus but defanged by guilopgar's pre-cleaned CSVs.
