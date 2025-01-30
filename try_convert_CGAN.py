#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# ------------------------------------------------
# 1. Same Generator architecture as training
# ------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_residual_blocks=6):
        super(Generator, self).__init__()
        model = [
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_channels = ngf
        out_channels = ngf * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels *= 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_channels)]

        # Upsampling
        out_channels = in_channels // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels //= 2

        # Output layer
        model += [
            nn.Conv2d(in_channels, output_nc, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# ------------------------------------------------
# 2. Main inference function
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Load CycleGAN Generator and process a single image.")
    parser.add_argument("--image", default = "first_image_received.jpg", help="Path to the input image")
    parser.add_argument("--model", default = "CycleGAN/CarF_netG_AB_epoch_9.pth", help="Path to the CarF_netG_AB_epoch_9.pth file")
    parser.add_argument("--output", default="output.png", help="Where to save the generated image")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use (e.g. cuda:0 or cpu)")
    args = parser.parse_args()

    # 1. Initialize device
    device = torch.device(args.device)

    # 2. Instantiate the Generator, load weights
    netG_AB = Generator(input_nc=3, output_nc=3, ngf=64, n_residual_blocks=6).to(device)
    netG_AB.load_state_dict(torch.load(args.model, map_location=device))
    netG_AB.eval()

    # 3. Define the same transforms you used in training
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])  # transforms from [0,1] -> [-1,1]
    ])

    # 4. Load and transform the input image
    input_img = Image.open(args.image).convert("RGB")
    input_tensor = transform(input_img).unsqueeze(0).to(device)  # shape: (1,3,H,W)

    print("size: ", input_tensor.shape)

    # 5. Generate with the loaded model (inference)
    with torch.no_grad():
        fake_car = netG_AB(input_tensor)  # shape: (1,3,H,W), range ~ [-1,1]

    # 6. "Denormalize" from [-1,1] back to [0,1]
    fake_car = (fake_car * 0.5) + 0.5  # shape: (1,3,H,W), now in [0,1]

    # 7. Convert tensor to PIL image
    #    1) remove batch dimension
    #    2) clamp to [0,1] just in case
    #    3) turn into PIL
    fake_car = fake_car.squeeze(0).cpu().clamp_(0,1)
    out_pil = transforms.ToPILImage()(fake_car)

    # 8. Save the image
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_pil.save(args.output)
    print(f"Saved transformed image to {args.output}")

if __name__ == "__main__":
    main()
