import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import time
import warnings
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

class Config:
    DATA_ROOT = "/TASMNet/FFHQ_stage1"

    IMAGE_SIZE = 128
    N_CHANNELS = 3

    BATCH_SIZE = 32
    LR = 0.001
    NUM_EPOCHS = 200
    VAL_SPLIT = 0.1

    MODEL_NAME = "checkpoint"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = "./checkpoints_FFHQ"

    os.makedirs(SAVE_DIR, exist_ok=True)


config = Config()


class SemiTransparentGlassesDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        半透明眼镜数据集
        参数:
            root_dir (str): 数据集根目录
            mode (str): 'train', 'val' 或 'test'
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        self.image_paths = []
        self.mask_paths = []
        self.lens_paths = []
        self.soft_mask_paths = []

        self._load_data_paths()

    def _load_data_paths(self):
        glasses_dir = os.path.join(self.root_dir, "glasses_image")
        mask_dir = os.path.join(self.root_dir, "glasses_mask")
        lens_mask_dir = os.path.join(self.root_dir, "lens_mask")
        soft_mask_dir = os.path.join(self.root_dir, "lens_soft_mask")

        all_files = sorted([f for f in os.listdir(glasses_dir) if f.endswith('.png') or f.endswith('.jpg')])

        train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=config.VAL_SPLIT, random_state=42)

        if self.mode == 'train':
            files = train_files
        elif self.mode == 'val':
            files = val_files
        else:  # test
            files = test_files

        for file in files:
            self.image_paths.append(os.path.join(glasses_dir, file))
            self.mask_paths.append(os.path.join(mask_dir, file))
            self.lens_paths.append(os.path.join(lens_mask_dir, file))
            soft_mask_path = os.path.join(soft_mask_dir, file)

            if os.path.exists(soft_mask_path):
                self.soft_mask_paths.append(soft_mask_path)
            else:
                # 对于测试集，软掩码可能不存在，用一个占位符
                self.soft_mask_paths.append(None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            with Image.open(self.image_paths[idx]) as img:
                glasses_img = img.convert('RGB')
            with Image.open(self.mask_paths[idx]) as mask:
                glasses_mask = mask.convert('L')
            with Image.open(self.lens_paths[idx]) as lens:
                lens_mask = lens.convert('L')

            if self.soft_mask_paths[idx] is not None:
                with Image.open(self.soft_mask_paths[idx]) as soft:
                    soft_mask = soft.convert('L')
                soft_mask = np.array(soft_mask, dtype=np.float32) / 255.0
            else:
                soft_mask = np.zeros(np.array(glasses_mask).shape, dtype=np.float32)

            glasses_img = np.array(glasses_img, dtype=np.float32) / 255.0
            glasses_mask = np.array(glasses_mask, dtype=np.float32) / 255.0
            lens_mask = np.array(lens_mask, dtype=np.float32) / 255.0

            input_img = np.zeros((glasses_img.shape[0], glasses_img.shape[1], 5), dtype=np.float32)
            input_img[:, :, :3] = glasses_img
            input_img[:, :, 3] = glasses_mask
            input_img[:, :, 4] = lens_mask

            if self.transform:
                augmented = self.transform(image=input_img, mask=soft_mask)
                input_img = augmented['image']
                soft_mask = augmented['mask']

                if soft_mask.ndim == 2:
                    soft_mask = soft_mask.unsqueeze(0)

            return input_img, soft_mask

        except Exception as e:
            print(f"Error loading data {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))


try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2


    train_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.5, 0.5], max_pixel_value=1.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.5, 0.5], max_pixel_value=1.0),
        ToTensorV2(),
    ])
except ImportError:
    print("Albumentations not installed. Using basic transforms.")
    train_transform = None
    val_transform = None


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        mid_channels = max(1, in_channels // reduction_ratio)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.attention(x)
        return x * att


class SoftmaskPredictor(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super().__init__()

        # 编码器
        self.enc1 = self._block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = self._block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = self._block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = self._block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(512, 1024)

        # 解码器
        self.upconv4 = self._upconv(1024, 512)
        self.dec4 = self._block(1024, 512)

        self.upconv3 = self._upconv(512, 256)
        self.dec3 = self._block(512, 256)

        self.upconv2 = self._upconv(256, 128)
        self.dec2 = self._block(256, 128)

        self.upconv1 = self._upconv(128, 64)
        self.dec1 = self._block(128, 64)

        # 输出层
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)

        self.attention_se_block = AttentionBlock(256)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        enc3 = self.attention_se_block(enc3)  # 应用注意力
        pool3 = self.pool3(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)

        bottleneck = self.bottleneck(pool4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # 输出层
        out = self.conv_out(dec1)
        out = torch.sigmoid(out)  # 将输出限制在0-1范围

        return out


class SoftmaskLoss(nn.Module):
    def __init__(self, bg_weight=0.2):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.bg_weight = bg_weight

    def forward(self, pred, target, lens_mask):
        loss_in_lens = self.l1_loss(pred * lens_mask, target * lens_mask)

        background_mask = 1 - lens_mask
        loss_in_background = self.l1_loss(pred * background_mask, torch.zeros_like(pred))

        total_loss = loss_in_lens + self.bg_weight * loss_in_background

        return total_loss


def train_model():
    train_dataset = SemiTransparentGlassesDataset(root_dir=config.DATA_ROOT, mode='train', transform=train_transform)
    val_dataset = SemiTransparentGlassesDataset(root_dir=config.DATA_ROOT, mode='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SoftmaskPredictor(in_channels=5, out_channels=1).to(config.DEVICE)

    criterion = SoftmaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS} [Train]")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            lens_mask = inputs[:, 4, :, :].unsqueeze(1)
            loss = criterion(outputs, targets, lens_mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, targets in progress_bar_val:
                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
                outputs = model(inputs)
                lens_mask = inputs[:, 4, :, :].unsqueeze(1)
                loss = criterion(outputs, targets, lens_mask)
                val_loss += loss.item()
                progress_bar_val.set_postfix({'val_loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, os.path.join(config.SAVE_DIR, f"{config.MODEL_NAME}_best.pth"))
            print(f"Saved new best model with val_loss: {best_val_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, os.path.join(config.SAVE_DIR, f"{config.MODEL_NAME}_epoch_{epoch + 1}.pth"))

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.2f}s")

    torch.save({'model_state_dict': model.state_dict()},
               os.path.join(config.SAVE_DIR, f"{config.MODEL_NAME}_final.pth"))
    print("Training complete!")


def evaluate_model(model_path, visualize=True):
    test_dataset = SemiTransparentGlassesDataset(root_dir=config.DATA_ROOT, mode='test', transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=min(8, config.BATCH_SIZE), shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SoftmaskPredictor(in_channels=5, out_channels=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    if visualize:
        vis_dir = os.path.join(config.SAVE_DIR, "visualizations_simplified")
        os.makedirs(vis_dir, exist_ok=True)

    test_loss_l1 = 0.0
    criterion = nn.L1Loss()

    def denormalize_image(img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
        denormed = img_tensor[:, :3, :, :] * std + mean
        return denormed.clamp(0, 1)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            lens_mask = inputs[:, 4, :, :].unsqueeze(1)

            loss = criterion(outputs * lens_mask, targets * lens_mask)
            test_loss_l1 += loss.item()

            if visualize:
                glasses_img_denorm = denormalize_image(inputs)
                glasses_img = glasses_img_denorm.cpu().numpy().transpose(0, 2, 3, 1)

                lens_mask_np = lens_mask.cpu().numpy().transpose(0, 2, 3, 1)[:, :, :, 0]
                pred_softmask = outputs.cpu().numpy()[:, 0, :, :]
                gt_softmask = targets.cpu().numpy()[:, 0, :, :]

                for j in range(glasses_img.shape[0]):
                    fig, ax = plt.subplots(1, 5, figsize=(25, 5))

                    ax[0].imshow(glasses_img[j])
                    ax[0].set_title('Input Image');
                    ax[0].axis('off')

                    ax[1].imshow(lens_mask_np[j], cmap='gray');
                    ax[1].set_title('Lens Mask');
                    ax[1].axis('off')

                    masked_pred = np.where(lens_mask_np[j] > 0.5, pred_softmask[j], 0)
                    im_pred = ax[2].imshow(masked_pred, cmap='plasma', vmin=0, vmax=1);
                    ax[2].set_title('Predicted');
                    ax[2].axis('off')
                    fig.colorbar(im_pred, ax=ax[2])

                    masked_gt = np.where(lens_mask_np[j] > 0.5, gt_softmask[j], 0)
                    im_gt = ax[3].imshow(masked_gt, cmap='plasma', vmin=0, vmax=1);
                    ax[3].set_title('Ground Truth');
                    ax[3].axis('off')
                    fig.colorbar(im_gt, ax=ax[3])

                    error = np.abs(masked_pred - masked_gt)
                    mae_val = error[lens_mask_np[j] > 0.5].mean() if np.any(lens_mask_np[j] > 0.5) else 0
                    im_error = ax[4].imshow(error, cmap='hot', vmin=0, vmax=0.5);
                    ax[4].set_title(f'Error (MAE: {mae_val:.3f})');
                    ax[4].axis('off')
                    fig.colorbar(im_error, ax=ax[4])

                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f'test_{i}_{j}.png'), bbox_inches='tight', dpi=200)
                    plt.close(fig)

    avg_test_loss = test_loss_l1 / len(test_loader)
    print(f"Average Test L1 Loss (in lens region): {avg_test_loss:.4f}")


def predict_softmask(model, glasses_img, glasses_mask, lens_mask):
    if isinstance(glasses_img, Image.Image): glasses_img = np.array(glasses_img, dtype=np.float32) / 255.0
    if isinstance(glasses_mask, Image.Image): glasses_mask = np.array(glasses_mask.convert('L'),
                                                                      dtype=np.float32) / 255.0
    if isinstance(lens_mask, Image.Image): lens_mask = np.array(lens_mask.convert('L'), dtype=np.float32) / 255.0

    input_data = np.zeros((glasses_img.shape[0], glasses_img.shape[1], 5), dtype=np.float32)
    input_data[:, :, :3] = glasses_img
    input_data[:, :, 3] = glasses_mask
    input_data[:, :, 4] = lens_mask

    transformed = val_transform(image=input_data)
    input_tensor = transformed['image'].unsqueeze(0).to(config.DEVICE)

    model.eval()
    with torch.no_grad():
        softmask_pred = model(input_tensor).squeeze().cpu().numpy()

    return softmask_pred


class TestGlassesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.mask_paths, self.lens_paths = [], [], []
        self._load_data_paths()

    def _load_data_paths(self):
        glasses_dir = os.path.join(self.root_dir, "glasses_image")
        mask_dir = os.path.join(self.root_dir, "glasses_mask")
        lens_mask_dir = os.path.join(self.root_dir, "lens_mask")
        all_files = sorted([f for f in os.listdir(glasses_dir) if f.endswith(('.png', '.jpg'))])
        for file in all_files:
            if os.path.exists(os.path.join(mask_dir, file)) and os.path.exists(os.path.join(lens_mask_dir, file)):
                self.image_paths.append(os.path.join(glasses_dir, file))
                self.mask_paths.append(os.path.join(mask_dir, file))
                self.lens_paths.append(os.path.join(lens_mask_dir, file))
            else:
                print(f"Warning: Missing mask for {file}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            with Image.open(self.image_paths[idx]) as img:
                glasses_img = img.convert('RGB')
            with Image.open(self.mask_paths[idx]) as mask:
                glasses_mask = mask.convert('L')
            with Image.open(self.lens_paths[idx]) as lens:
                lens_mask = lens.convert('L')

            original_size = glasses_img.size[::-1]  # H, W
            glasses_img_np = np.array(glasses_img, dtype=np.float32) / 255.0
            glasses_mask_np = np.array(glasses_mask, dtype=np.float32) / 255.0
            lens_mask_np = np.array(lens_mask, dtype=np.float32) / 255.0

            input_img = np.zeros((glasses_img_np.shape[0], glasses_img_np.shape[1], 5), dtype=np.float32)
            input_img[:, :, :3] = glasses_img_np
            input_img[:, :, 3] = glasses_mask_np
            input_img[:, :, 4] = lens_mask_np

            if self.transform:
                input_img = self.transform(image=input_img)['image']

            filename = os.path.basename(self.image_paths[idx])
            return input_img, original_size, filename
        except Exception as e:
            print(f"Error loading test data {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SoftmaskPredictor(in_channels=5, out_channels=1)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading model: {e}");
        raise
    model.to(device)
    model.eval()
    print(f"Successfully loaded model from {model_path}")
    return model


def test_model_on_new_dataset(model_path, test_root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_path)
    test_dataset = TestGlassesDataset(root_dir=test_root_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Found {len(test_dataset)} images to process.")

    softmask_dir = os.path.join(output_dir, "predicted_softmasks")
    overlay_dir = os.path.join(output_dir, "overlay_visualizations")
    os.makedirs(softmask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    for inputs, original_size, filename in tqdm(test_loader, desc="Processing images"):
        inputs = inputs.to(config.DEVICE)
        with torch.no_grad():
            softmask_pred = model(inputs).squeeze().cpu().numpy()

        orig_height, orig_width = original_size[0].item(), original_size[1].item()
        if softmask_pred.shape != (orig_height, orig_width):
            softmask_pred = cv2.resize(softmask_pred, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

        file_name = filename[0]
        base_name = os.path.splitext(file_name)[0]

        # 保存软掩模
        mask_image = (np.clip(softmask_pred, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(mask_image).save(os.path.join(softmask_dir, f"{base_name}_softmask.png"))

        # 创建叠加可视化
        img_path = os.path.join(test_root_dir, "glasses_image", file_name)
        with Image.open(img_path) as img:
            orig_img_array = np.array(img.convert('RGB'))

        heatmap = (plt.cm.plasma(softmask_pred)[:, :, :3] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(orig_img_array, 0.5, heatmap, 0.5, 0)
        Image.fromarray(overlay).save(os.path.join(overlay_dir, f"{base_name}_overlay.png"))

    print(f"Results saved to {output_dir}")


# 主程序入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TASMNet")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Running mode: train or test")
    parser.add_argument("--model", help="Path to the model for testing")
    parser.add_argument("--test_data", help="Path to the test dataset directory")
    parser.add_argument("--output_dir", default="./test_results_simplified", help="Output directory for test results")

    args = parser.parse_args()

    if args.mode == "train":
        train_model()
        best_model_path = os.path.join(config.SAVE_DIR, f"{config.MODEL_NAME}_best.pth")
        if os.path.exists(best_model_path):
            print("\nEvaluating the best model on the test set...")
            evaluate_model(best_model_path, visualize=True)

    elif args.mode == "test":
        if not args.model or not args.test_data:
            print("ERROR: Test mode requires --model and --test_data arguments.")
            exit(1)

        print(f"Testing model: {args.model}")
        print(f"On dataset: {args.test_data}")
        print(f"Saving results to: {args.output_dir}")

        test_model_on_new_dataset(
            model_path=args.model,
            test_root_dir=args.test_data,
            output_dir=args.output_dir
        )

# python TASMNet.py --mode test --model ./checkpoints_FFHQ/checkpoint_best.pth --test_data /FFHQ_stage2 --output_dir ./FFHQ_stage2