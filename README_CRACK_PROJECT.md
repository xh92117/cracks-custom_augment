# YOLO 裂缝检测项目 🔍

本项目基于 Ultralytics YOLO 框架，集成了专门用于裂缝检测的自定义图像增强算法。

## 🎯 项目特色

### ✅ 自定义增强算法集成
- **专门针对裂缝检测优化**的图像增强技术
- **完全集成到 Ultralytics 框架**中，无需额外配置文件
- **灵活的命令行控制**，支持实时调整增强参数

### 🔧 核心功能
1. **对比度增强**: 使用CLAHE (限制对比度自适应直方图均衡化)
2. **边缘增强**: 基于Canny边缘检测的特征增强
3. **噪声添加**: 随机高斯噪声增强数据多样性
4. **亮度调节**: 自适应亮度调整
5. **高级直方图均衡化**: 使用双曲正切映射的专业算法

## 📦 项目结构

```
crackv2/
├── ultralytics/                    # Ultralytics YOLO框架 (已集成自定义增强)
│   ├── cfg/default.yaml           # 框架配置文件 (包含自定义增强参数)
│   ├── data/augment.py            # 数据增强模块 (包含CustomAugment类)
│   └── ...                        # 其他框架文件
├── main.py                        # 主训练脚本 (支持自定义增强控制)
└── README_CRACK_PROJECT.md       # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备
```bash
pip install ultralytics
pip install scipy  # 用于高级图像处理算法
```

### 2. 数据集配置
创建数据集配置文件 `crack_dataset.yaml`:
```yaml
path: ../datasets/crack_dataset  # 数据集根目录
train: images/train              # 训练图像
val: images/val                  # 验证图像
test: images/test               # 测试图像

nc: 1                           # 类别数量
names: ['crack']                # 类别名称
```

### 3. 训练命令

#### 基本训练
```bash
# 使用默认自定义增强设置
python main.py --data crack_dataset.yaml --model yolo11n.yaml

# 使用预训练权重
python main.py --data crack_dataset.yaml --model yolo11n.yaml --weights yolo11n.pt
```

#### 自定义增强控制
```bash
# 禁用自定义增强
python main.py --data crack_dataset.yaml --model yolo11n.yaml --disable-custom-augment

# 高强度自定义增强
python main.py --data crack_dataset.yaml --model yolo11n.yaml \
    --custom-augment 0.8 --custom-intensity 0.7 --custom-sigma 8

# 轻度自定义增强
python main.py --data crack_dataset.yaml --model yolo11n.yaml \
    --custom-augment 0.3 --custom-intensity 0.2 --custom-sigma 3

# 只修改增强概率
python main.py --data crack_dataset.yaml --model yolo11n.yaml --custom-augment 0.6
```

## ⚙️ 参数说明

### 命令行参数

#### 必需参数
- `--data`: 数据集配置文件路径
- `--model`: 模型架构配置文件路径

#### 可选参数
- `--hyp`: 训练超参数配置文件路径（默认: `ultralytics/cfg/default.yaml`）
- `--weights`: 预训练权重路径（默认: `yolo11n.pt`）

#### 自定义增强控制参数
- `--disable-custom-augment`: 禁用自定义增强算法
- `--custom-augment FLOAT`: 增强概率 (0.0-1.0)
- `--custom-intensity FLOAT`: 增强强度 (0.0-1.0)  
- `--custom-sigma INT`: 平滑系数 (1-10)

### 参数详解

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| `custom_augment` | 0.0-1.0 | 0.5 | 自定义增强应用概率，0.0=完全禁用，1.0=总是应用 |
| `custom_intensity` | 0.0-1.0 | 0.4 | 增强强度，控制边缘增强等效果的强烈程度 |
| `custom_sigma` | 1-10 | 5 | 平滑系数，用于高级直方图均衡化算法 |

## 🎛️ 控制方式

### 1. 命令行控制（推荐）
直接通过 main.py 的命令行参数控制：
```bash
python main.py --data dataset.yaml --model yolo11n.yaml --custom-augment 0.8
```

### 2. 配置文件控制
编辑 `ultralytics/cfg/default.yaml` 文件：
```yaml
# 自定义增强参数
custom_augment: 0.5      # 增强概率
custom_intensity: 0.4    # 增强强度
custom_sigma: 5          # 平滑系数
```

### 3. 程序化控制
```python
from ultralytics import YOLO
import yaml

# 加载并修改配置
with open('ultralytics/cfg/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 禁用自定义增强
config['custom_augment'] = 0.0

# 或启用高强度增强
config['custom_augment'] = 0.8
config['custom_intensity'] = 0.7
config['custom_sigma'] = 8

# 训练
model = YOLO('yolo11n.yaml')
model.train(data='crack_dataset.yaml', **config)
```

## 🔬 技术亮点

### 高级直方图均衡化算法
实现了基于双曲正切映射的高级直方图均衡化：
- 动态Canny阈值计算
- 自适应形态学处理
- 直方图峰值分析
- 智能强度计算
- 边缘融合技术

### 智能增强策略
- **随机选择增强方法**避免过拟合
- **异常处理机制**确保训练稳定性
- **与YOLO标签系统完美兼容**
- **保持边界框和分割掩码的一致性**

### 工程化实现
- **模块化设计**，易于扩展
- **完整的错误处理和日志记录**
- **配置化参数管理**
- **向后兼容性保证**

## 📊 使用示例

### 不同场景的配置建议

#### 高质量数据集
```bash
# 轻度增强，保持数据原始特征
python main.py --data dataset.yaml --model yolo11n.yaml --custom-augment 0.3 --custom-intensity 0.2
```

#### 小数据集
```bash
# 高强度增强，增加数据多样性
python main.py --data dataset.yaml --model yolo11n.yaml --custom-augment 0.8 --custom-intensity 0.6
```

#### 对比实验
```bash
# 禁用自定义增强作为基准
python main.py --data dataset.yaml --model yolo11n.yaml --disable-custom-augment

# 启用自定义增强对比效果
python main.py --data dataset.yaml --model yolo11n.yaml --custom-augment 0.5
```

## 🚀 优势特性

- ✅ **一键部署**: 无需复杂配置，开箱即用
- ✅ **灵活控制**: 多种方式控制增强参数
- ✅ **专业算法**: 针对裂缝检测优化的增强技术
- ✅ **完整集成**: 深度集成到Ultralytics框架中
- ✅ **参数验证**: 自动参数验证和错误处理
- ✅ **详细日志**: 清晰的训练过程信息显示

## 📝 注意事项

1. 确保安装了所有依赖包，特别是 `scipy`
2. 数据集路径需要正确配置
3. 建议使用GPU进行训练以获得更好的性能
4. 可以根据具体数据集调整增强参数
5. 命令行参数优先级高于配置文件设置

## 🔗 相关链接

- [Ultralytics 官方文档](https://docs.ultralytics.com/)
- [YOLO 模型介绍](https://www.ultralytics.com/yolo)
- [项目 GitHub 仓库](https://github.com/xh92117/cracks-custom_augment)

## 📄 许可证

本项目基于 AGPL-3.0 许可证。详见 [LICENSE](LICENSE) 文件。

---

**如有问题或建议，欢迎提交 Issue 或 Pull Request！** 🎉 