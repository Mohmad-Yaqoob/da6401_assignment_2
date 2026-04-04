# DA6401 Assignment 2 — Visual Perception Pipeline

## Links
- **W&B Report**: https://wandb.ai/da25m017-indian-institute-of-technology-madras/da6401-assignment2/reports/Assignment-2--VmlldzoxNjQwOTUzOQ?accessToken=lsiv1cra6pf4t3li1onberkxr2s6qg5ikwriupyrq3n7o77eqtgzcl6a0fxqiiuj
- **GitHub Repo**: https://github.com/Mohmad-Yaqoob/da6401_assignment_2

## Project Structure
```
.
├── checkpoints/
│   └── checkpoints.md
├── data/
│   └── pets_dataset.py
├── losses/
│   ├── __init__.py
│   └── iou_loss.py
├── models/
│   ├── __init__.py
│   ├── classification.py
│   ├── layers.py
│   ├── localization.py
│   ├── multitask.py
│   ├── segmentation.py
│   └── vgg11.py
├── inference.py
├── requirements.txt
├── train.py
└── README.md
```

## Tasks
- Task 1: VGG11 classification with custom BatchNorm and Dropout
- Task 2: Bounding box localization with MSE + IoU loss
- Task 3: U-Net segmentation with VGG11 encoder
- Task 4: Unified multi-task pipeline

## Training
```bash
python train.py --task cls --epochs 30 --lr 1e-4 --dropout_p 0.5
python train.py --task loc --epochs 30 --lr 5e-4
python train.py --task seg --epochs 30 --lr 1e-3 --strategy full
python train.py --task multi --epochs 30 --lr 5e-4
```
