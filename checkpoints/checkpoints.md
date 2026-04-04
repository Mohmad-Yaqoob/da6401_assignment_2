Checkpoint Submission Instructions

Submit the following three checkpoint files inside this folder:
1. Task-1: classifier.pth
2. Task-2: localizer.pth
3. Task-3: unet.pth

These filenames are mandatory for evaluation.

Each checkpoint is saved as:
{
    "state_dict": model.state_dict(),
    "epoch": epoch,
    "best_metric": best_metric,
}