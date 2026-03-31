#!/usr/bin/env python3
"""
遥感图像PCA可视化使用示例

展示如何使用remote_sensing_pca_visualization.py脚本的各种功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from remote_sensing_pca_visualization import RemoteSensingPCAVisualizer


def example_basic_usage():
    """基本用法示例"""
    print("=== 基本用法示例 ===")

    # 创建可视化器（使用默认设置）
    visualizer = RemoteSensingPCAVisualizer()

    # 假设你有一个遥感图像文件
    # image_path = "path/to/your/satellite_image.jpg"

    # 基本可视化
    # result = visualizer.visualize(image_path)
    print("基本用法：visualizer.visualize('image.jpg')")


def example_advanced_usage():
    """高级用法示例"""
    print("\n=== 高级用法示例 ===")

    # 使用ViT-7B模型
    visualizer_large = RemoteSensingPCAVisualizer(model_name="vit7b16")

    # 指定前景点
    points = [(100, 200), (300, 400), (500, 600)]

    # 高级可视化
    # result = visualizer_large.visualize(
    #     image_path="satellite_image.jpg",
    #     output_path="output_advanced.png",
    #     foreground_method="points",
    #     points=points,
    #     show_plot=True
    # )

    print("高级用法示例代码：")
    print("""
    visualizer = RemoteSensingPCAVisualizer(model_name="vit7b16")
    points = [(100, 200), (300, 400)]
    result = visualizer.visualize(
        image_path="satellite_image.jpg",
        output_path="output.png",
        foreground_method="points",
        points=points
    )
    """)


def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")

    visualizer = RemoteSensingPCAVisualizer()

    # 假设有一批遥感图像
    # image_dir = "path/to/satellite_images/"
    # output_dir = "path/to/output/"

    # for image_file in os.listdir(image_dir):
    #     if image_file.endswith(('.jpg', '.png', '.tif')):
    #         input_path = os.path.join(image_dir, image_file)
    #         output_path = os.path.join(output_dir, f"vis_{image_file}")
    #
    #         try:
    #             visualizer.visualize(input_path, output_path, show_plot=False)
    #             print(f"处理完成: {image_file}")
    #         except Exception as e:
    #             print(f"处理失败 {image_file}: {e}")

    print("批量处理示例代码：")
    print("""
    for image_file in os.listdir('satellite_images/'):
        if image_file.endswith(('.jpg', '.png')):
            input_path = f'satellite_images/{image_file}'
            output_path = f'output/vis_{image_file}'
            visualizer.visualize(input_path, output_path, show_plot=False)
    """)


def example_custom_foreground_detection():
    """自定义前景检测示例"""
    print("\n=== 自定义前景检测示例 ===")

    class CustomRemoteSensingVisualizer(RemoteSensingPCAVisualizer):
        def detect_foreground(self, image, method="custom", threshold=0.5, **kwargs):
            """自定义前景检测逻辑"""
            import numpy as np
            from scipy import ndimage

            # 将PIL图像转换为numpy数组
            image_array = np.array(image)

            # 示例：基于NDVI（归一化植被指数）的检测
            # 这里只是示例，实际需要根据你的遥感数据调整
            if len(image_array.shape) == 3:
                # 假设图像是RGB格式
                r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

                # 简化的植被检测（实际应使用真实的波段数据）
                vegetation_mask = (g > r) & (g > b) & (g > threshold * 255)

                # 形态学操作清理mask
                vegetation_mask = ndimage.binary_opening(vegetation_mask, iterations=2)
                vegetation_mask = ndimage.binary_closing(vegetation_mask, iterations=2)

                return torch.from_numpy(vegetation_mask.astype(float))

            # 默认返回全前景
            return torch.ones(image.size[1], image.size[0])

    # 使用自定义可视化器
    # custom_visualizer = CustomRemoteSensingVisualizer()
    # result = custom_visualizer.visualize("satellite_image.jpg")

    print("自定义前景检测示例：")
    print("""
    class CustomVisualizer(RemoteSensingPCAVisualizer):
        def detect_foreground(self, image, method="custom", **kwargs):
            # 实现你的自定义检测逻辑
            # 例如：基于NDVI的植被检测
            pass
    """)


def show_command_line_examples():
    """命令行使用示例"""
    print("\n=== 命令行使用示例 ===")

    examples = [
        "# 基本使用",
        "python remote_sensing_pca_visualization.py satellite_image.jpg",

        "# 指定输出路径",
        "python remote_sensing_pca_visualization.py image.jpg -o result.png",

        "# 使用ViT-7B模型",
        "python remote_sensing_pca_visualization.py image.jpg -m vit7b16",

        "# 指定前景点",
        "python remote_sensing_pca_visualization.py image.jpg -p 100 200 300 400",

        "# 使用CPU",
        "python remote_sensing_pca_visualization.py image.jpg -d cpu",

        "# 只保存结果",
        "python remote_sensing_pca_visualization.py image.jpg --no-plot",
    ]

    for example in examples:
        print(f"  {example}")


def main():
    """主函数"""
    print("遥感图像DINOv3 PCA可视化工具 - 使用示例")
    print("=" * 50)

    # 显示各种使用示例
    example_basic_usage()
    example_advanced_usage()
    example_batch_processing()
    example_custom_foreground_detection()
    show_command_line_examples()

    print("\n" + "=" * 50)
    print("💡 提示：")
    print("1. 确保权重文件存在于 pretrained_weights/DINOv3 ViT SAT-493M/")
    print("2. 调整前景检测参数以获得更好的结果")
    print("3. 使用ViT-7B获得更高的准确性（需要更多计算资源）")
    print("4. 批量处理时建议使用 --no-plot 选项")


if __name__ == "__main__":
    main()
