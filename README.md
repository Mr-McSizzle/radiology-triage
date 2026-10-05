# NeuroScan Edge

**Experimental AI decision-support pipeline for chest X-ray analysis, explainability, and triage workflows.**

NeuroScan Edge combines convolutional image models, test-time augmentation, Grad-CAM visualization, rule-based clinical reasoning, input validation, and a FastAPI application layer.

> Research prototype only. This repository is not a medical device and is not validated for clinical diagnosis or patient-care decisions.

## System components

- **Image inference** using DenseNet-121 and ResNet-50 based pipelines.
- **Test-time augmentation** to compare / stabilize predictions across transformed inputs.
- **Grad-CAM** generation for visual inspection of model-attended regions.
- **Clinical reasoning rules** for combining findings and prioritization logic.
- **Input validation** for obviously invalid or unsuitable image uploads.
- **FastAPI application layer** with queue / command-center style workflows.
- **Evaluation scripts and saved result artifacts** for inspecting model behavior.

## Architecture

```text
X-ray input
    ↓
input validation
    ↓
model inference / TTA
    ↓
finding scores
    ├── Grad-CAM explanation
    └── reasoning / prioritization rules
             ↓
       API + triage UI
```

## Evaluation transparency

The repository includes `evaluation_report.txt` and `evaluation_results.csv`. The checked-in evaluation report currently records:

| Metric | Result |
|---|---:|
| Images evaluated | 7,146 |
| Top-1 match rate | 24.67% |
| Top-5 match rate | 40.12% |
| Thresholded match rate | 39.17% |
| Precision | 0.220 |
| Recall | 0.742 |
| F1 | 0.340 |

These results are included deliberately: the project is an engineering and research prototype, **not evidence of clinical-grade performance**.

## Running the API

Create an environment, install the dependencies listed by the repository, then launch the FastAPI application with Uvicorn. The project README previously used:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Research limitations

- No claim of clinical safety, efficacy, or regulatory validation.
- Evaluation metrics are well below what would be required for autonomous diagnostic use.
- Rule-based reasoning cannot substitute for prospective clinical validation.
- Model calibration, dataset shift, subgroup performance, and external-site generalization require dedicated study.
- Grad-CAM is an interpretability aid, not proof of causal reasoning.

## Why keep this project public

The interesting part of NeuroScan Edge is not a claim that AI has solved radiology. It is the **systems problem**: model orchestration, uncertainty-aware workflow design, explainability, validation, and human-facing triage infrastructure around imperfect models.