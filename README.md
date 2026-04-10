# DA6401 Assignment 2 — Visual Perception Pipeline

Built a complete multi-task visual perception system on the Oxford-IIIT Pet dataset covering breed classification, bounding box localisation, and trimap segmentation.

## Results

| Task | Metric | Score |
|------|--------|-------|
| Classification | Macro F1 | 1.00 |
| Localisation | Acc @ IoU ≥ 0.5 | 90% |
| Localisation | Acc @ IoU ≥ 0.75 | 70% |
| Segmentation | Macro Dice | 0.88 |

## Links

- **WandB Report:** https://wandb.ai/da25m017-indian-institute-of-technology-madras/da6401-assignment2/reports/Assignment-2--VmlldzoxNjQwOTUzOQ
- **GitHub Repo:** https://github.com/Mohmad-Yaqoob/da6401_assignment_2

## Project Structure

```
.
├── checkpoints/
├── data/
│   └── pets_dataset.py   # Oxford-IIIT Pet loader (cls/loc/seg/all modes)
├── losses/
│   └── iou_loss.py       # Custom IoU loss for bbox regression
├── models/
│   ├── layers.py         # CustomDropout (inverted, bernoulli mask)
│   ├── vgg11.py          # VGG11Encoder backbone
│   ├── classification.py # VGG11Classifier
│   ├── localization.py   # VGG11Localizer
│   ├── segmentation.py   # VGG11UNet
│   └── multitask.py      # MultiTaskPerceptionModel (3 encoders)
├── multitask.py          # Root-level re-export for autograder
├── train.py              # Training script for all 3 tasks
└── requirements.txt
```

## Training

```bash
# Task 1 — Classification
python train.py --task classification --data_dir data/oxford_pet --epochs 60 --lr 5e-4

# Task 2 — Localisation
python train.py --task localization --data_dir data/oxford_pet --epochs 40 --lr 5e-4

# Task 3 — Segmentation
python train.py --task segmentation --data_dir data/oxford_pet --epochs 40 --lr 5e-4
```

## Architecture Highlights

- **VGG11Encoder** — 5-block VGG11 backbone with BatchNorm after every conv, Kaiming init
- **CustomDropout** — hand-rolled bernoulli mask with inverted scaling, no nn.Dropout used
- **ClassificationHead** — FC 4096→4096→37 with BatchNorm1d and CustomDropout
- **RegressionHead** — FC 25088→1024→4 with sigmoid×224 for bounded pixel output
- **VGG11UNet** — symmetric decoder with ConvTranspose2d upsampling and skip connections
- **MultiTaskPerceptionModel** — three separate encoders (one per task) to avoid feature distribution mismatch between heads trained independently