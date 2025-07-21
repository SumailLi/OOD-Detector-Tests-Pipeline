from utils.utils import get_model, AverageMeter
import torch
from torch.utils.data import TensorDataset, DataLoader
from udacity_dataset import UdacityImageDataset, UdacityImageTestDataset, UdacityImageAttackDataset
from torch import nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import numpy as np
class UdacityDataset(Dataset):
    def __init__(self, root_folders, csv_name="driving_log.csv", transform=None):
        """
        Args:
            root_folders (list of str): List of paths like 'training_data/track1/normal'
            csv_name (str): CSV file name in each folder
            transform: torchvision transforms to apply to images
        """
        self.samples = []
        self.transform = transform 

        for folder in root_folders:
            csv_path = os.path.join(folder, csv_name)
            img_folder = os.path.join(folder, "IMG")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                filename = row[13][5:]
                img_name = os.path.basename(filename.strip())
                img_path = os.path.join(img_folder, img_name)
                steering_angle = float(row[8])
                self.samples.append((img_path, steering_angle))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, angle = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        np_image = np.array(image)
        if self.transform:
            image = self.transform(image)
        angle = torch.tensor(angle, dtype=torch.float32)
        return image, angle, np_image
    

    
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def save_attack_images(images, batch_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(images.size(0)):
        img_path = os.path.join(save_dir, f"adv_{batch_idx:03d}_{i:02d}.png")
        save_image(images[i], img_path)


def pgd_attack_regression(model, x, y_true, epsilon=0.1, alpha=0.01, steps=40):
    x = x.clone().detach().cuda()
    y_true = y_true.clone().detach().cuda()

    adv_x = x.clone().detach().requires_grad_(True)

    for _ in range(steps):
        output = model(adv_x)
        loss = F.mse_loss(output, y_true)

        model.zero_grad()
        loss.backward()

        # Gradient ascent on the input
        grad_sign = adv_x.grad.sign()
        adv_x = adv_x + alpha * grad_sign

        # Project perturbation to the epsilon L-infinity ball
        perturbation = torch.clamp(adv_x - x, min=-epsilon, max=epsilon)
        adv_x = torch.clamp(x + perturbation, 0, 1).detach().requires_grad_(True)

    return adv_x.detach()

def  spsa_attack(
    model: nn.Module,
    images: torch.Tensor,        # (B,3,H,W) in [0,1]
    y_true: torch.Tensor,        # (B,1) or (B,)  ground-truth angles
    *,
    num_steps: int = 300,
    sigma: float = 0.001,
    step_size: float = 0.01,
    print_every: int = 50
) -> torch.Tensor:
    """
    Returns a batch of adversarial images with the same shape as `images`.
    The attack maximises the absolute steering-angle error.
    """
    device  = images.device
    B       = images.size(0)
    x_adv   = images.clone()
    y_true  = y_true.view(B).float().to(device)

    model.eval()
    for step in range(1, num_steps + 1):
        # SPSA two-point gradient estimate
        noise   = torch.randn_like(x_adv)
        pos_img = (x_adv + sigma * noise).clamp(0, 1)
        neg_img = (x_adv - sigma * noise).clamp(0, 1)

        with torch.no_grad():
            pred_pos = model(pos_img).view(B)   # (B,)
            pred_neg = model(neg_img).view(B)

        loss_pos = (pred_pos - y_true).abs()    # (B,)
        loss_neg = (pred_neg - y_true).abs()

        grad_est = ((loss_pos - loss_neg) / (2 * sigma)).view(B, 1, 1, 1) * noise

        # Ascent step in pixel space (increase error)
        x_adv = (x_adv + step_size * grad_est.sign()).clamp(0, 1)

        if step % print_every == 0 or step == num_steps:
            with torch.no_grad():
                preds = model(x_adv).view(B)
            mean_err = (preds - y_true).abs().mean().item()
            print(f"[SPSA] step {step:3d}/{num_steps}, mean error = {mean_err:.3f}")

    return x_adv

def salt_pepper_noise(x: torch.Tensor,
                      prob: float = 0.02,
                      salt_val: float =  1.0,   # use  1.0 for [-1,1];  1.0 for [0,1]
                      pepper_val: float = -1.0  # use -1.0 for [-1,1];  0.0 for [0,1]
                     ) -> torch.Tensor:
    """
    x:        input batch  [B, C, H, W]  (already on the right device)
    prob:     overall corruption probability (salt+pepper). 0.02 ⇒ 2 % pixels flipped
    returns:  noisy clone of x
    """
    rnd = torch.rand_like(x)                         # U[0,1] for every pixel
    noisy = x.clone()
    noisy[rnd <  prob * 0.5] = pepper_val            # first half → pepper
    noisy[rnd > 1 - prob * 0.5] = salt_val           # second half → salt
    return noisy

if __name__ == "__main__":
    name = "epoch"
    attack = "pgd"
    #dave2v1
    davev1 = get_model(name)
    state_dict = torch.load(f"../result/checkpoint_best_{name}_normalized.pth")
    davev1.load_state_dict(state_dict)
    davev1.cuda()
    epsilon = 8/255
    model =davev1
    save_dir = f"../evaluation/{attack}_attack/{name}"
    path = "../"
    # Load dataset
    data_folders = [
    "evaluation_data/CHAUFFEUR-Track1-Normal",
    ]
    transform = transforms.Compose([
    transforms.Resize((160, 320)),  # Resize to 160x160 (H, W)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # org_transform = transforms.Compose([
    # transforms.Resize((160, 320)),  # Resize to 160x160 (H, W)
    # transforms.ToTensor(),
    # ])
    
    full_path = [os.path.join(path, data) for idx, data in enumerate(data_folders)]
    loss_fn = nn.MSELoss()
    dataset = UdacityDataset(root_folders=full_path, csv_name="driving_log.csv", transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    if attack == "fgsm":
        for batch_idx, ((data, target, org_img)) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data, target = data.cuda(), target.cuda()
            data.requires_grad = True
            model.zero_grad()
            output = model(data)
            loss = loss_fn(output, target.float())
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
            save_attack_images(perturbed_data.cpu(), batch_idx, save_dir)
    elif attack == "pgd":
        for batch_idx, (data, target, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data, target = data.cuda(), target.cuda()
            target = target.view(-1, 1)  # if your output is [B, 1]

            # Run PGD attack
            perturbed_data = pgd_attack_regression(model, data, target, epsilon, alpha=0.01, steps=40)

            # Save adversarial examples
            save_attack_images(perturbed_data.cpu(), batch_idx, save_dir)

    elif attack == "spsa":
        for batch_idx, (data, target, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data, target = data.cuda(), target.cuda()
            target = target.view(-1, 1)  # if your output is [B, 1]

            # Run PGD attack
            perturbed_data = spsa_attack(model, data, target, num_steps=300, sigma=0.001, step_size=0.01)

            # Save adversarial examples
            save_attack_images(perturbed_data.cpu(), batch_idx, save_dir)
    elif attack == "sp":                        # <-- new keyword
        for batch_idx, (data, target,_) in tqdm(enumerate(dataloader),
                                            total=len(dataloader)):
            data, target = data.cuda(), target.cuda()
            target = target.view(-1, 1)                 # keep your regression shape

            # Apply salt-and-pepper noise
            perturbed_data = salt_pepper_noise(data,
                                            prob=0.02,          # tweak as needed
                                            salt_val=1.0,
                                            pepper_val=-1.0)

            # Save adversarial / corrupted examples
            save_attack_images(perturbed_data.cpu(), batch_idx, save_dir)
            
