"""
    By KaiDuo Liu, Sep 2024
    Contact: dlammm2066@outlook.com
    description: Ghost imaging (GI) is capable of reconstructing images
    under low-light conditions by single-pixel measurements. However,
    improving image resolution often requires extensive single-pixel sampling,
    limiting practical applications. Here we propose a super-resolution
    algorithm of Ghost Imaging using CNN with Grouped orthogonalization
    algorithm Constraint (GICGC), which aims to reconstruct natural images
    at super-resolution with strong local regularities and self-similarity.
    Here is simple sample code
"""
import os
import time
import datetime
import torchvision.transforms as transforms
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import utils
import cv2


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(SingleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            SingleConv(in_channels, in_channels),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upup = nn.BatchNorm2d(in_channels // 2)
            self.upupup = nn.LeakyReLU(inplace=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.upup = nn.BatchNorm2d(in_channels // 2)
            self.upupup = nn.LeakyReLU(inplace=True)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = self.upup(x1)
        x1 = self.upupup(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.Sigmoid()
        )


class GICGC_model_Unet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(GICGC_model_Unet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits


class CustomImageDataset(Dataset):

    transform_haha = transforms.Compose([
        transforms.ToTensor()
    ])

    def __init__(self, image_data, transform_haha=None):
        self.image_data = image_data
        self.transform_haha = transform_haha

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        image = self.image_data[index]
        if self.transform_haha:
            image = self.transform_haha(image)
        return image


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def create_model_haha(num_classes):
    model = GICGC_model_Unet(in_channels=1, num_classes=num_classes, bilinear=True, base_c=32)
    return model


def add_white_noise(signal, snr_db):
    np.random.seed(1)
    signal_power = np.sqrt(np.mean(signal**2))
    noise_power = signal_power / (10**(snr_db/10))
    noise_std = np.sqrt(noise_power)
    noise = np.random.randn(*signal.shape) * noise_std
    noisy_signal = signal + noise
    return noisy_signal


def train_one_epoch_haha(model, optimizer, data_loader, device, epoch, frame_num, frame_3d, bucket,
                         lr_scheduler, print_freq=1, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    target = torch.tensor(bucket, dtype=torch.float64)
    target = target.to(device)
    target.requires_grad_(False)
    optimizer.zero_grad()
    for image in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device)
        image = image.to(torch.float32)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            output = output[0, 0]
            bucket_purpose = []
            for i in range(frame_num):
                frame_2d = frame_3d[i, :, :]
                frame_2d = torch.from_numpy(frame_2d)
                frame_2d = frame_2d.to(device)
                bucket_single_purpose = torch.sum(torch.mul(output, frame_2d))
                bucket_purpose.append(bucket_single_purpose)
            bucket_purpose = torch.stack(bucket_purpose, dim=0)
            bucket_purpose = (bucket_purpose - torch.mean(bucket_purpose)) / torch.std(bucket_purpose)
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(bucket_purpose, target)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)
        outputImage = output.detach().numpy()
        x_max = max(map(max, outputImage))
        x_min = min(map(min, outputImage))
        image_new = np.zeros((args.Height, args.Width))
        for i in range(args.Height):
            for j in range(args.Width):
                outputImage[i][j] = np.round((255 * (outputImage[i][j] - x_min) / (x_max - x_min)))
        # 5. The output of each iteration
        file_name = f"./Output_{epoch + 1}.png"
        cv2.imwrite(file_name, outputImage)
    return metric_logger.meters["loss"].global_avg, lr


def main(args):
    # 1.loading illumination patterns
    frame_3d = np.zeros((args.frame_num, args.Height, args.Width))
    for i in range(args.frame_num):
        frame_3d_data = f"./speckle_{i + 1}.png"
        frame_3d_data = Image.open(frame_3d_data)
        frame_3d_data = np.array(frame_3d_data)
        frame_3d[i, :, :] = frame_3d_data

    # 2. loading one-dimensional bucket signals, 1st of two means
    # bucket = []
    # for i in range(args.frame_num):
    #     bucket_data = f"./bucket_{i + 1}.png"
    #     bucket_data = Image.open(bucket_data)
    #     bucket_data = np.asarray(bucket_data)
    #     bucket_data = np.sum(bucket_data)
    #     bucket.append(bucket_data)
    # bucket = np.array(bucket)

    # 2. loading one-dimensional bucket signals, 2nd of two means
    bucket = []
    image = r"./Ghost.png"
    x = Image.open(image)  # 打开图片
    gray_x = x.convert('L')  # 将图片转为灰度
    image = np.array(gray_x)  # 将灰度图片转为矩阵
    for i in range(args.frame_num):
        frame_2d = frame_3d[i, :, :]
        bucket_single = np.sum(np.multiply(image, frame_2d))
        bucket.append(bucket_single)
    bucket = np.array(bucket)  # 将列表转换为 NumPy 数组

    # 3. add noise or not/ bucket processing
    # bucket = add_white_noise(bucket, 22)

    # 4. using Grouped orthonormalization algorithm
    frame_3d_schmidt = np.zeros((args.frame_num, args.Height, args.Width))
    frame_3d_schmidt_alpha = np.zeros((args.frame_num, args.Height * args.Width))
    frame_3d_schmidt_Beta = np.zeros((args.frame_num, args.Height * args.Width))
    bucket_schmidt = np.zeros(args.frame_num)
    gs = 100
    for k in range(args.frame_num // gs + 1):
        for i in range(k * gs, min((k + 1) * gs, args.frame_num)):
            frame_3d_schmidt_alpha[i, :] = frame_3d[i, :, :].reshape(1, args.Height * args.Width)
            frame_3d_schmidt_Beta[i, :] = frame_3d[i, :, :].reshape(1, args.Height * args.Width)
            bucket_schmidt[i] = bucket[i]
            for j in range(k * gs, i):
                numerator = np.dot(frame_3d_schmidt_alpha[i, :], frame_3d_schmidt_Beta[j, :])
                denominator = np.dot(frame_3d_schmidt_Beta[j, :], frame_3d_schmidt_Beta[j, :])
                frame_3d_schmidt_Beta[i, :] -= (numerator / denominator) * frame_3d_schmidt_Beta[j, :]
                bucket_schmidt[i] -= (numerator / denominator) * bucket_schmidt[j]
            bucket_schmidt[i] /= np.linalg.norm(np.squeeze(frame_3d_schmidt_Beta[i, :]))
            frame_3d_schmidt_alpha[i, :] /= np.linalg.norm(np.squeeze(frame_3d_schmidt_alpha[i, :]))
            frame_3d_schmidt_Beta[i, :] /= np.linalg.norm(np.squeeze(frame_3d_schmidt_Beta[i, :]))
    for i in range(args.frame_num):
        frame_3d_schmidt[i, :, :] = frame_3d_schmidt_Beta[i, :].reshape(args.Height, args.Width)

    bucket_schmidt = (bucket_schmidt - np.mean(bucket_schmidt)) / np.std(bucket_schmidt)
    bucket = (bucket - np.min(bucket)) / (np.max(bucket) - np.min(bucket))
    bucket_mean = np.mean(bucket)
    frame_3d_mean = np.mean(frame_3d, 0)
    obj = np.zeros((args.Height, args.Width))
    for i in range(args.frame_num):
        obj += (bucket[i] - bucket_mean) * (np.squeeze(frame_3d[i, :, :]) - frame_3d_mean)
    raw = obj / args.frame_num
    generated_images = [raw]
    transform_haha = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = CustomImageDataset(generated_images, transform_haha=transform_haha)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = args.num_classes
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model = create_model_haha(num_classes=num_classes)
    model.to(device)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch_haha(model, optimizer, train_loader, device, epoch, args.frame_num,
                                             frame_3d_schmidt,
                                             bucket_schmidt, lr_scheduler=lr_scheduler, print_freq=args.print_freq,
                                             scaler=scaler)
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.10f}\n"
            f.write(train_info + "\n\n")
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch GICGC_model_Unet training")
    parser.add_argument("--data-path", default="./", help="DRIVE root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=20, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--frame-num", default="1000", type=int, help="frame_num")
    parser.add_argument("--Height", default="128", type=int, help="Height")
    parser.add_argument("--Width", default="128", type=int, help="Width")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    main(args)
