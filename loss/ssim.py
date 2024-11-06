# ssim_loss.py
import torch
import torch.nn.functional as F

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3 
        self.window = self._create_window(window_size)

    def _create_window(self, window_size, sigma=1.5):
        coords = torch.arange(window_size).float() - window_size // 2
        coords = coords.repeat(window_size, 1)
        coords_x = coords
        coords_y = coords.t()
        gaussian = torch.exp(-(coords_x**2 + coords_y**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        return gaussian.unsqueeze(0).unsqueeze(0)

    def forward(self, img1, img2):
        window = self.window.expand(self.channel, 1, self.window_size, self.window_size).to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        return 1 - ssim_map.mean(1).mean(1).mean(1)