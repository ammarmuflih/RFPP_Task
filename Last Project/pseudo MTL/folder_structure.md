fsdd-multitask/
├── data/
│   ├── raw/                    # .wav files
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── src/
│   ├── data/
│   │   ├── loader.py           # load .wav + metadata
│   │   └── splitter.py         # speaker-aware split
│   ├── features/
│   │   ├── mfcc.py             # MFCC + delta extraction
│   │   ├── spectral.py         # ZCR, centroid, rolloff
│   │   ├── aggregator.py       # mean/std/max/min pooling
│   │   └── pipeline.py         # full feature pipeline
│   ├── models/
│   │   ├── mlp.py              # MLP implementation
│   │   ├── baselines.py        # SVM, RF, XGBoost
│   │   └── trainer.py          # training + eval loop
│   ├── evaluation/
│   │   ├── metrics.py          # accuracy, F1, confusion matrix
│   │   └── visualizer.py       # plot hasil
│   └── utils/
│       ├── config.py
│       └── logger.py
├── notebooks/
│   ├── 01_eda.ipynb            # eksplorasi dataset
│   ├── 02_feature_analysis.ipynb
│   └── 03_results.ipynb
├── configs/
│   ├── digit_model.yaml
│   └── speaker_model.yaml
├── tests/
├── train.py
├── evaluate.py
└── README.md

## DATA PIPELINE
# Augmentasi audio — WAJIB untuk generalisasi
augmentations = [
    AddGaussianNoise(p=0.3),
    TimeStretch(rate_range=(0.8, 1.2), p=0.3),
    PitchShift(semitones_range=(-2, 2), p=0.3),
    TimeMasking(max_ms=100, p=0.2),      # SpecAugment
    FrequencyMasking(max_hz=500, p=0.2), # SpecAugment
]

# Split strategy — PENTING
# Jangan random split! Split by speaker untuk
# menghindari data leakage
# Train: speaker 1-4, Val: speaker 5, Test: speaker 6

## MULTI-TASK LOSS
# Weighted loss — kunci MTL yang baik
loss = α * loss_digit + β * loss_speaker

# Best practice: mulai dengan α=β=0.5
# lalu tune dengan uncertainty weighting (Kendall et al.)
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Log variance sebagai learnable parameter
        self.log_var_digit = nn.Parameter(torch.zeros(1))
        self.log_var_speaker = nn.Parameter(torch.zeros(1))

    def forward(self, loss_digit, loss_speaker):
        loss = (torch.exp(-self.log_var_digit) * loss_digit
                + self.log_var_digit
                + torch.exp(-self.log_var_speaker) * loss_speaker
                + self.log_var_speaker)
        return loss

## Experiment Tracking
Gunakan Weights & Biases (WandB) — ini standar industri dan bagus untuk porto:

import wandb
wandb.init(project="fsdd-multitask", config=cfg)
wandb.log({"digit_acc": acc_digit, "speaker_acc": acc_speaker})

fsdd-multitask/
├── data/
│   ├── raw/                    # .wav files
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── src/
│   ├── data/
│   │   ├── loader.py           # load .wav + metadata
│   │   └── splitter.py         # speaker-aware split
│   ├── features/
│   │   ├── mfcc.py             # MFCC + delta extraction
│   │   ├── spectral.py         # ZCR, centroid, rolloff
│   │   ├── aggregator.py       # mean/std/max/min pooling
│   │   └── pipeline.py         # full feature pipeline
│   ├── models/
│   │   ├── mlp.py              # MLP implementation
│   │   ├── baselines.py        # SVM, RF, XGBoost
│   │   └── trainer.py          # training + eval loop
│   ├── evaluation/
│   │   ├── metrics.py          # accuracy, F1, confusion matrix
│   │   └── visualizer.py       # plot hasil
│   └── utils/
│       ├── config.py
│       └── logger.py
├── notebooks/
│   ├── 01_eda.ipynb            # eksplorasi dataset
│   ├── 02_feature_analysis.ipynb
│   └── 03_results.ipynb
├── configs/
│   ├── digit_model.yaml
│   └── speaker_model.yaml
├── tests/
├── train.py
├── evaluate.py
└── README.md

## Fusion
Spektogram -> PCA 50 + MFCC