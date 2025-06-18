# YOLO 裂缝检测项目

本项目基于 Ultralytics YOLO 框架，集成了专门用于裂缝检测的自定义图像增强算法。

## 项目结构

```
crackv2/
├── ultralytics/                    # YOLO框架源码 (从GitHub克隆)
│   └── ultralytics/
│       └── data/
│           └── augment.py          # 已集成CustomAugment类
├── cfg/
│   └── default.yaml                # 训练配置文件
├── models/                         # 模型配置和权重目录
├── data/                          # 数据集配置目录
├── runs/                          # 训练结果输出目录
├── main.py                        # 主训练脚本
├── custom_augment.py              # 原始自定义增强算法
└── README.md                      # 项目说明文档
```

## 功能特性

### 1. 自定义图像增强算法
集成了专门用于裂缝检测的高级图像增强技术：

- **对比度增强**: 使用CLAHE (限制对比度自适应直方图均衡化)
- **边缘增强**: 基于Canny边缘检测的特征增强
- **噪声添加**: 随机高斯噪声增强数据多样性
- **亮度调节**: 自适应亮度调整
- **高级直方图均衡化**: 使用双曲正切映射的专业算法

### 2. YOLO框架集成
- 成功将自定义增强算法集成到 Ultralytics YOLO 数据预处理流水线
- 在 `v8_transforms` 函数中添加了 `CustomAugment` 类
- 支持通过配置文件控制增强参数

### 3. 配置化训练
- 完整的YAML配置文件 (`cfg/default.yaml`)
- 支持自定义增强参数配置
- 包含完整的训练、验证和导出设置

## 安装和使用

### 1. 环境要求
```bash
pip install ultralytics
pip install scipy  # 用于高级图像处理算法
```

### 2. 数据集准备
创建数据集配置文件 `data/crack_dataset.yaml`:
```yaml
path: ../datasets/crack_dataset  # 数据集根目录
train: images/train              # 训练图像
val: images/val                  # 验证图像
test: images/test               # 测试图像

nc: 1                           # 类别数量
names: ['crack']                # 类别名称
```

### 3. 训练模型

#### 基本训练命令
```bash
# 基本训练（使用默认自定义增强设置）
python main.py --data data/crack_dataset.yaml --model yolo11n.yaml

# 使用预训练权重
python main.py --data data/crack_dataset.yaml --model yolo11n.yaml --weights yolo11n.pt
```

#### 控制自定义增强
```bash
# 禁用自定义增强
python main.py --data data/crack_dataset.yaml --model yolo11n.yaml --disable-custom-augment

# 设置高强度自定义增强
python main.py --data data/crack_dataset.yaml --model yolo11n.yaml \
    --custom-augment 0.8 --custom-intensity 0.7 --custom-sigma 8

# 设置轻度自定义增强
python main.py --data data/crack_dataset.yaml --model yolo11n.yaml \
    --custom-augment 0.3 --custom-intensity 0.2 --custom-sigma 3

# 只修改增强概率，其他参数使用默认值
python main.py --data data/crack_dataset.yaml --model yolo11n.yaml --custom-augment 0.6
```

### 4. 配置参数
自定义增强参数已完全集成到ultralytics框架中。在 `ultralytics/ultralytics/cfg/default.yaml` 中可以调整以下参数：
```yaml
# 自定义增强参数 (已集成到框架默认配置中)
custom_augment: 0.5      # 自定义增强应用概率 (0.0=禁用, 1.0=总是应用)
custom_intensity: 0.4    # 自定义增强强度 (0.0-1.0)
custom_sigma: 5          # 自定义增强平滑系数 (1-10)
```

### 5. 控制自定义增强

#### 方法1: 直接修改框架配置
编辑 `ultralytics/ultralytics/cfg/default.yaml` 文件中的参数。

#### 方法2: 程序化控制
```python
from ultralytics import YOLO
import yaml

# 加载并修改配置
with open('ultralytics/ultralytics/cfg/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 禁用自定义增强
config['custom_augment'] = 0.0

# 或启用高强度增强
config['custom_augment'] = 0.8
config['custom_intensity'] = 0.7
config['custom_sigma'] = 8

# 训练
model = YOLO('yolo11n.yaml')
model.train(data='your_data.yaml', **config)
```

#### 方法3: 命令行参数控制（推荐）
main.py 现在支持直接通过命令行参数控制自定义增强：
```bash
# 禁用自定义增强
python main.py --data dataset.yaml --model yolo11n.yaml --disable-custom-augment

# 自定义增强参数
python main.py --data dataset.yaml --model yolo11n.yaml \
    --custom-augment 0.8 --custom-intensity 0.6 --custom-sigma 7
```

#### 方法4: 测试脚本
运行 `python test_main_augment_control.py` 查看详细使用示例。

### 6. 命令行参数说明

main.py 支持以下命令行参数：

#### 必需参数
- `--data`: 数据集配置文件路径
- `--model`: 模型架构配置文件路径

#### 可选参数
- `--hyp`: 训练超参数配置文件路径（默认使用内置配置）
- `--weights`: 预训练权重路径（默认: yolo11n.pt）

#### 自定义增强控制参数
- `--disable-custom-augment`: 禁用自定义增强算法
- `--custom-augment FLOAT`: 增强概率 (0.0-1.0)
- `--custom-intensity FLOAT`: 增强强度 (0.0-1.0)  
- `--custom-sigma INT`: 平滑系数 (1-10)

#### 参数优先级
命令行参数 > 配置文件设置

#### 示例命令
```bash
# 查看所有参数
python main.py --help

# 完全禁用自定义增强
python main.py --data dataset.yaml --model yolo11n.yaml --disable-custom-augment

# 精确控制增强参数
python main.py --data dataset.yaml --model yolo11n.yaml \
    --custom-augment 0.7 --custom-intensity 0.5 --custom-sigma 6
```

## 技术亮点

### 1. 高级直方图均衡化算法
实现了基于双曲正切映射的高级直方图均衡化：
- 动态Canny阈值计算
- 自适应形态学处理
- 直方图峰值分析
- 智能强度计算
- 边缘融合技术

### 2. 智能增强策略
- 随机选择增强方法避免过拟合
- 异常处理确保训练稳定性
- 与YOLO标签系统完美兼容
- 保持边界框和分割掩码的一致性

### 3. 工程化实现
- 模块化设计，易于扩展
- 完整的错误处理和日志记录
- 配置化参数管理
- 向后兼容性保证

## 训练脚本功能

`main.py` 提供了完整的训练流程：
- 自动路径验证和目录创建
- 配置文件加载和验证
- 模型初始化和设置
- 示例数据集配置生成
- 训练过程监控和日志

## 自定义增强算法详解

`CustomAugment` 类提供了5种增强方法：

1. **enhance_contrast()**: CLAHE对比度增强
2. **enhance_edges()**: Canny边缘特征增强
3. **add_noise()**: 高斯噪声添加
4. **adjust_brightness()**: HSV亮度调节
5. **_tanh_hist_equalization()**: 高级直方图均衡化

每次调用时随机选择一种方法，确保数据增强的多样性。

## 注意事项

1. 确保安装了所有依赖包，特别是 `scipy`
2. 数据集路径需要正确配置
3. 建议使用GPU进行训练以获得更好的性能
4. 可以根据具体数据集调整增强参数

## 后续扩展

- 可以添加更多专门的裂缝检测增强算法
- 支持多尺度训练策略
- 集成更多评估指标
- 添加模型导出和部署功能

## 联系方式

如有问题或建议，请创建Issue或提交Pull Request。 