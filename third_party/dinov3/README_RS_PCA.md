# 遥感图像DINOv3 PCA彩虹色可视化工具

基于DINOv3 SAT-493M预训练模型，为遥感图像生成类似pca.ipynb中的彩虹色可视化效果。

## 功能特性

- ✅ 使用专门为遥感数据预训练的DINOv3模型 (SAT-493M)
- ✅ 支持ViT-L和ViT-7B两种模型规格
- ✅ 自动前景检测和手动指定点功能
- ✅ 生成彩虹色PCA可视化效果
- ✅ 支持多种输出格式和显示选项
- ✅ 完整的命令行接口

## 安装依赖

```bash
# 安装必要的Python包
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn scipy opencv-python pillow tqdm

# 如果使用CUDA，请确保安装对应的PyTorch版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 快速开始

### 基本用法

```bash
# 使用默认设置处理图像
python remote_sensing_pca_visualization.py your_image.jpg

# 指定输出路径
python remote_sensing_pca_visualization.py your_image.jpg -o output.png
```

### 高级用法

```bash
# 使用ViT-7B模型（更大更准确，但更慢）
python remote_sensing_pca_visualization.py your_image.jpg -m vit7b16

# 使用CPU（如果没有GPU）
python remote_sensing_pca_visualization.py your_image.jpg -d cpu

# 指定特定的前景点
python remote_sensing_pca_visualization.py your_image.jpg -p 100 200 300 400

# 使用手动前景检测方法
python remote_sensing_pca_visualization.py your_image.jpg -f manual -t 0.7

# 只保存结果，不显示图表
python remote_sensing_pca_visualization.py your_image.jpg --no-plot
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `image_path` | 输入图像路径 | 必需 |
| `-o, --output` | 输出图像路径 | 自动生成 |
| `-m, --model` | 模型类型 (vitl16/vit7b16) | vitl16 |
| `-d, --device` | 计算设备 (cuda/cpu) | cuda |
| `-f, --foreground_method` | 前景检测方法 (auto/manual) | auto |
| `-t, --threshold` | 前景检测阈值 | 0.5 |
| `-p, --points` | 指定点坐标 (x1 y1 x2 y2 ...) | 无 |
| `--no-plot` | 只保存结果，不显示图表 | False |

## 前景检测方法

### 自动检测 (auto)
使用边缘检测和形态学操作自动识别前景区域。

### 手动检测 (manual)
基于HSV色彩空间的亮度和饱和度进行前景检测。

### 指定点检测 (points)
通过指定坐标点创建感兴趣区域：
```bash
python remote_sensing_pca_visualization.py image.jpg -p 100 200 300 400
```

## 输出结果

脚本会生成包含三个子图的图像：
1. **原始图像** - 输入的遥感图像
2. **前景检测** - 检测到的前景区域（灰度图）
3. **PCA彩虹色可视化** - 基于DINOv3特征的彩虹色可视化

## 技术细节

### 模型配置
- **ViT-L**: 24层，参数量3亿，适合大多数应用
- **ViT-7B**: 40层，参数量67亿，更高的准确性但需要更多计算资源

### 图像预处理
- 调整尺寸为16×16 patch的倍数
- 使用遥感专用的归一化参数：
  - Mean: (0.430, 0.411, 0.296)
  - Std: (0.213, 0.156, 0.143)

### PCA可视化
- 对前景patch特征进行3D PCA降维
- 使用sigmoid函数生成鲜艳的彩虹色
- 支持whitening确保各分量方差一致

## 示例输出

```
正在加载DINOv3 vitl16 模型...
模型加载完成，使用设备: cuda
正在处理图像: satellite_image.jpg
图像尺寸: 原始=(1920, 1080), 调整后=(768, 1366)
提取的特征形状: torch.Size([3072, 1024])
前景区域占比: 45.2%
正在对 1389 个patch进行PCA...
可视化结果已保存到: satellite_image_pca_visualization.png

✅ 可视化完成!
📁 输出文件: satellite_image_pca_visualization.png
```

## 故障排除

### 常见问题

1. **CUDA相关错误**
   ```bash
   # 使用CPU模式
   python remote_sensing_pca_visualization.py image.jpg -d cpu
   ```

2. **内存不足**
   ```bash
   # 使用较小的模型
   python remote_sensing_pca_visualization.py image.jpg -m vitl16
   ```

3. **权重文件不存在**
   - 确保`pretrained_weights/DINOv3 ViT SAT-493M/`目录存在且包含权重文件
   - 或者手动指定权重路径

4. **导入错误**
   ```bash
   # 设置Python路径
   export PYTHONPATH=/path/to/dinov3-main:$PYTHONPATH
   ```

### 性能优化

- 使用ViT-L模型获得更好的性能平衡
- 对于大图像，考虑使用`--no-plot`选项跳过matplotlib显示
- GPU内存充足时可以使用ViT-7B获得更好的效果

## 扩展功能

### 作为Python模块使用

```python
from remote_sensing_pca_visualization import RemoteSensingPCAVisualizer

# 创建可视化器
visualizer = RemoteSensingPCAVisualizer(model_name="vitl16", device="cuda")

# 处理图像
result_path = visualizer.visualize(
    image_path="your_image.jpg",
    output_path="output.png",
    foreground_method="auto",
    show_plot=True
)
```

### 自定义前景检测

```python
# 继承并重写detect_foreground方法
class CustomVisualizer(RemoteSensingPCAVisualizer):
    def detect_foreground(self, image, method="custom", **kwargs):
        # 实现自定义的前景检测逻辑
        pass
```

## 引用

基于DINOv3论文：
```
@article{simeoni2025dinov3,
  title={DINOv3},
  author={Siméoni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Michaël and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timothée and Moutakanni, Théo and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and Jégou, Hervé and Labatut, Patrick and Bojanowski, Piotr},
  year={2025}
}
```

## 许可证

遵循DINOv3项目的许可证条款。
