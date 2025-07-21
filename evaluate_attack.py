from utils.utils import get_model, AverageMeter
from udacity_dataset import (
    UdacityImageDataset,
    UdacityImageTestDataset,
    UdacityImageAttackDataset,
    AnomalImageDataset
)
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dave  = get_model("dave2v1").to(device).eval()
    epoch = get_model("epoch").to(device).eval()
    dave.load_state_dict(torch.load("/raid/007-Xiangyu-Experiments/selforacle/EnhanceTransferability/result/checkpoint_best_dave2v1_normalized.pth", map_location=device))
    epoch.load_state_dict(torch.load("/raid/007-Xiangyu-Experiments/selforacle/EnhanceTransferability/result/checkpoint_best_epoch_normalized.pth", map_location=device))
    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    clean_ds = UdacityImageTestDataset(
        base_dir="/raid/007-Xiangyu-Experiments/selforacle/evaluation_data",
        data=["CHAUFFEUR-Track1-Normal"],
        mode="clean",
        transform=transform,
        fmodel=None,
    )
    adv_ds = AnomalImageDataset("/raid/007-Xiangyu-Experiments/selforacle/evaluation/sp_attack/epoch", transform=transform)

    assert len(clean_ds) == len(adv_ds), "Datasets must be the same length"

    clean_loader = DataLoader(clean_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    adv_loader   = DataLoader(adv_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # 3.  Helper
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_angles(model, loader):
        preds = []
        for imgs, *_ in tqdm(loader):
            preds.append(model(imgs.to(device)).flatten().cpu())
        return torch.cat(preds)

    # ------------------------------------------------------------------
    # 4.  Run & report
    # ------------------------------------------------------------------
    results = {}
    for name, model in [("Dave-2 v1", dave), ("Epoch", epoch)]:
        y_clean = predict_angles(model, clean_loader)
        y_adv   = predict_angles(model, adv_loader)

        diff = abs(y_adv - y_clean )                               # signed shift
        results[name] = {
            "mean_shift" : diff.mean().item(),
            "std_shift"  : diff.std(unbiased=False).item(),    # population std
            # Optional extras:
            # "MAE"        : diff.abs().mean().item(),
            # "RMSE"       : diff.pow(2).mean().sqrt().item(),
        }

    # Nicely formatted print-out
    for k,v in results.items():
        print(f"[{k:10}]  mean Δ: {v['mean_shift']:+.4f}  |  σ: {v['std_shift']:.4f}")