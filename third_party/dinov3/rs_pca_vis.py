#!/usr/bin/env python3
"""
遥感图像DINOv3 PCA彩虹色可视化脚本

基于DINOv3 SAT-493M预训练模型，为遥感图像生成类似pca.ipynb中的彩虹色可视化效果。
支持前景检测和指定点识别功能。

作者: AI Assistant
日期: 2025-09-19
"""

import argparse
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from scipy import signal, ndimage
import cv2

# 添加项目路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from dinov3.hub.backbones import dinov3_vitl16, dinov3_vit7b16
except ImportError as e:
    print(f"导入DINOv3模块失败: {e}")
    print("请确保DINOv3项目已正确安装或设置PYTHONPATH")
    sys.exit(1)


class RemoteSensingPCAVisualizer:
    """遥感图像DINOv3 PCA可视化器"""

    def __init__(self, model_name="vitl16", device="cuda"):
        """
        初始化可视化器

        Args:
            model_name: 模型名称 ('vitl16' 或 'vit7b16')
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.patch_size = 16

        # 遥感图像的归一化参数 (SAT-493M)
        self.SATELLITE_MEAN = (0.430, 0.411, 0.296)
        self.SATELLITE_STD = (0.213, 0.156, 0.143)

        # 模型层数配置
        self.MODEL_TO_NUM_LAYERS = {
            "vits16": 12,
            "vits16plus": 12,
            "vitb16": 12,
            "vitl16": 24,
            "vith16plus": 32,
            "vit7b16": 40,
        }

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载DINOv3模型"""
        print(f"正在加载DINOv3 {self.model_name} 模型...")

        # 构建权重路径
        weights_dir = project_root / "pretrained_weights" 
        if self.model_name == "vitl16":
            weights_file = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        elif self.model_name == "vit7b16":
            weights_file = "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
        else:
            raise ValueError(f"不支持的模型名称: {self.model_name}")

        weights_path = weights_dir / weights_file

        if not weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        # 加载模型
        if self.model_name == "vitl16":
            self.model = dinov3_vitl16(weights=str(weights_path))
        elif self.model_name == "vit7b16":
            self.model = dinov3_vit7b16(weights=str(weights_path))

        self.model.to(self.device)
        self.model.eval()

        print(f"模型加载完成，使用设备: {self.device}")

    def load_image(self, image_path):
        """
        加载并预处理图像

        Args:
            image_path: 图像路径

        Returns:
            预处理后的图像tensor和原始图像
        """
        # 加载图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        # 调整图像尺寸为patch size的倍数
        w, h = image.size
        h_patches = int(768 / self.patch_size)  # 使用固定patch数量
        w_patches = int((w * 768) / (h * self.patch_size))

        new_h = h_patches * self.patch_size
        new_w = w_patches * self.patch_size

        # 调整图像尺寸
        image_resized = TF.resize(image, (new_h, new_w))

        # 转换为tensor并归一化
        image_tensor = TF.to_tensor(image_resized)
        image_tensor = TF.normalize(image_tensor, mean=self.SATELLITE_MEAN, std=self.SATELLITE_STD)

        return image_tensor, image_resized, image

    def extract_features(self, image_tensor):
        """
        提取图像特征

        Args:
            image_tensor: 预处理后的图像tensor

        Returns:
            提取的patch特征
        """
        n_layers = self.MODEL_TO_NUM_LAYERS[self.model_name]

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.float32):
                feats = self.model.get_intermediate_layers(
                    image_tensor.unsqueeze(0).to(self.device),
                    n=range(n_layers),
                    reshape=True,
                    norm=True
                )

                # 获取最后一层的特征
                x = feats[-1].squeeze().detach().cpu()
                dim = x.shape[0]

                # 重塑为patch特征: [H*W, D]
                x = x.view(dim, -1).permute(1, 0)

        return x

    def detect_foreground(self, image, method="auto", threshold=0.5, points=None):
        """
        检测前景区域

        Args:
            image: PIL图像
            method: 检测方法 ('auto', 'manual', 'points', 'none')
            threshold: 自动检测阈值
            points: 手动指定的点坐标 [(x1,y1), (x2,y2), ...]

        Returns:
            前景mask
        """
        if method == "none":
            # 不进行前景检测，返回全1的mask（整个图像）
            return torch.ones(image.size[1], image.size[0])

        elif method == "manual":
            # 手动指定前景区域（这里简化为基于颜色的检测）
            image_array = np.array(image)

            # 转换为HSV空间，更好地分离前景和背景
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

            # 基于亮度和饱和度检测前景（可以根据具体遥感数据调整）
            brightness = hsv[:, :, 2]
            saturation = hsv[:, :, 1]

            # 自适应阈值
            brightness_thresh = np.percentile(brightness, 70)
            saturation_thresh = np.percentile(saturation, 30)

            foreground_mask = (brightness > brightness_thresh) & (saturation > saturation_thresh)

        elif method == "points" and points is not None:
            # 基于指定点生成前景区域
            h, w = image.size[1], image.size[0]
            foreground_mask = np.zeros((h, w), dtype=bool)

            # 为每个指定点创建一个区域
            for point in points:
                x, y = point
                # 创建以点为中心的小区域
                y_min = max(0, y - 50)
                y_max = min(h, y + 50)
                x_min = max(0, x - 50)
                x_max = min(w, x + 50)
                foreground_mask[y_min:y_max, x_min:x_max] = True

        else:  # method == "auto"
            # 自动检测（简化的版本）
            image_array = np.array(image)

            # 使用边缘检测
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            # 膨胀边缘以创建前景区域
            kernel = np.ones((5, 5), np.uint8)
            foreground_mask = cv2.dilate(edges, kernel, iterations=2).astype(bool)

        # 应用中值滤波平滑mask
        foreground_mask = signal.medfilt2d(foreground_mask.astype(float), kernel_size=5) > threshold

        return torch.from_numpy(foreground_mask.astype(float))

    def create_pca_visualization(self, features, foreground_mask, image_size):
        """
        创建PCA彩虹色可视化

        Args:
            features: patch特征 [H*W, D]，其中 H*W 是patch数量
            foreground_mask: 前景mask [H, W]，图像级别
            image_size: 图像尺寸 (H, W)

        Returns:
            PCA可视化图像
        """
        h_patches, w_patches = image_size[0] // self.patch_size, image_size[1] // self.patch_size

        # 将图像级别的mask转换为patch级别的mask
        # 使用平均池化将图像mask转换为patch mask
        from torch.nn.functional import avg_pool2d
        patch_mask = avg_pool2d(
            foreground_mask.unsqueeze(0).unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).squeeze()

        # 展平patch mask用于选择patch特征
        foreground_selection = patch_mask.view(-1) > 0.5

        # 提取前景patch特征
        fg_patches = features[foreground_selection]

        if len(fg_patches) == 0:
            print("警告: 未检测到前景区域，使用所有patch")
            fg_patches = features

        # PCA降维
        print(f"正在对 {len(fg_patches)} 个patch进行PCA...")
        pca = PCA(n_components=3, whiten=True)
        pca.fit(fg_patches.numpy())

        # 对所有patch应用PCA
        projected_image = torch.from_numpy(
            pca.transform(features.numpy())
        ).view(h_patches, w_patches, 3)

        # 使用sigmoid生成鲜艳颜色
        projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)

        # 应用patch级别的前景mask
        # 如果patch_mask都是1（表示不进行前景检测），则不应用mask
        if not torch.all(patch_mask == 1.0):
            projected_image *= (patch_mask.unsqueeze(0) > 0.5)

        return projected_image

    def visualize(self, image_path, output_path=None, foreground_method="auto",
                  foreground_threshold=0.5, points=None, show_plot=True):
        """
        执行完整的可视化流程

        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径 (可选)
            foreground_method: 前景检测方法
            foreground_threshold: 前景检测阈值
            points: 指定点坐标
            show_plot: 是否显示matplotlib图表

        Returns:
            保存的可视化结果路径
        """
        print(f"正在处理图像: {image_path}")

        # 1. 加载和预处理图像
        image_tensor, image_resized, original_image = self.load_image(image_path)
        print(f"图像尺寸: 原始={original_image.size}, 调整后={image_resized.size}")

        # 2. 提取特征
        features = self.extract_features(image_tensor)
        print(f"提取的特征形状: {features.shape}")

        # 3. 检测前景
        foreground_mask = self.detect_foreground(
            image_resized,
            method=foreground_method,
            threshold=foreground_threshold,
            points=points
        )
        print(f"前景区域占比: {foreground_mask.mean().item():.2%}")

        # 4. 创建PCA可视化
        pca_visualization = self.create_pca_visualization(
            features,
            foreground_mask,
            image_resized.size[::-1]  # (H, W)
        )

        # 5. 保存或显示结果
        if output_path is None:
            output_path = Path(image_path).stem + "_pca_visualization.png"

        if show_plot:
            plt.figure(figsize=(12, 5), dpi=150)

            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title("原始图像", fontsize=12)
            plt.axis('off')

            # 前景mask
            plt.subplot(1, 3, 2)
            plt.imshow(foreground_mask.numpy(), cmap='gray')
            plt.title("前景检测", fontsize=12)
            plt.axis('off')

            # PCA可视化
            plt.subplot(1, 3, 3)
            plt.imshow(pca_visualization.permute(1, 2, 0).numpy())
            plt.title("PCA彩虹色可视化", fontsize=12)
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.show()
        else:
            # 只保存PCA可视化结果
            pca_image = pca_visualization.permute(1, 2, 0).numpy()
            pca_image = (pca_image * 255).astype(np.uint8)
            Image.fromarray(pca_image).save(output_path)

        print(f"可视化结果已保存到: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="遥感图像DINOv3 PCA彩虹色可视化")
    parser.add_argument("--image_path", default="0.jpg", help="输入图像路径")
    parser.add_argument("-o", "--output", help="输出图像路径")
    parser.add_argument("-m", "--model", choices=["vitl16", "vit7b16"],
                       default="vitl16", help="使用的DINOv3模型")
    parser.add_argument("-d", "--device", choices=["cuda", "cpu"],
                       default="cuda", help="计算设备")
    parser.add_argument("--enable-foreground", action="store_true",
                       help="启用前景检测，默认关闭（可视化整个图像）")
    parser.add_argument("-f", "--foreground_method", choices=["auto", "manual", "points", "none"],
                       default="auto", help="前景检测方法（当启用--enable-foreground时使用）")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                       help="前景检测阈值")
    parser.add_argument("-p", "--points", nargs='+', type=int,
                       default=[524, 686], help="指定点坐标 (x1 y1 x2 y2 ...)，默认为 [524, 686]")
    parser.add_argument("--no-plot", action="store_true",
                       help="不显示matplotlib图表，只保存结果")

    args = parser.parse_args()

    # 处理指定点坐标
    points = None
    if args.points:
        if len(args.points) % 2 != 0:
            print("错误: 点坐标必须是成对的 (x y)")
            return
        points = [(args.points[i], args.points[i+1])
                 for i in range(0, len(args.points), 2)]

    # 根据是否启用前景检测设置参数
    if not args.enable_foreground:
        # 不启用前景检测时，使用整个图像
        args.foreground_method = "none"
    else:
        # 启用前景检测时，如果有指定点则使用points模式
        if points and len(points) == 1 and points[0] == (524, 686):
            args.foreground_method = "points"

    try:
        # 创建可视化器
        visualizer = RemoteSensingPCAVisualizer(
            model_name=args.model,
            device=args.device
        )

        # 执行可视化
        output_path = visualizer.visualize(
            image_path=args.image_path,
            output_path=args.output,
            foreground_method=args.foreground_method,
            foreground_threshold=args.threshold,
            points=points,
            show_plot=not args.no_plot
        )

        print(f"\n✅ 可视化完成!")
        print(f"📁 输出文件: {output_path}")

    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())