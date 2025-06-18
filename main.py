import argparse
import os
import yaml
from ultralytics import YOLO

# 禁用 wandb，避免交互式提示
os.environ["WANDB_DISABLED"] = "true"

def main():
    """
    YOLO 训练启动器。
    通过命令行参数来配置并启动训练任务，支持控制自定义增强算法。
    
    核心参数:
    - --data: 数据集配置文件
    - --model: 模型架构配置文件
    - --hyp: 训练超参数配置文件
    - --weights: 预训练权重文件
    
    自定义增强控制:
    - --disable-custom-augment: 禁用自定义增强
    - --custom-augment: 设置增强概率 (0.0-1.0)
    - --custom-intensity: 设置增强强度 (0.0-1.0)
    - --custom-sigma: 设置平滑系数 (1-10)
    """
    parser = argparse.ArgumentParser(description='极简 YOLO 训练启动器')
    
    # --- 定义四个核心配置参数 ---
    
    # 1. 数据集配置文件
    parser.add_argument('--data', type=str, required=True, 
                        help='数据集配置文件路径')
                        
    # 2. 模型架构配置文件
    parser.add_argument('--model', type=str, required=True, 
                        help='模型架构配置文件路径')
                        
    # 3. 训练超参数配置文件
    parser.add_argument('--hyp', type=str, default='ultralytics/cfg/default.yaml', 
                        help='训练超参数配置文件路径 (现在使用内置配置，支持自定义增强)')
                        
    # 4. 预训练权重文件
    parser.add_argument('--weights', type=str, default='yolo11n.pt', 
                        help='预训练权重路径')
                        
    # 5. 自定义增强控制
    parser.add_argument('--disable-custom-augment', action='store_true',
                        help='禁用自定义增强算法（设置custom_augment=0.0）')
    parser.add_argument('--custom-augment', type=float, default=None,
                        help='自定义增强概率 (0.0-1.0)，会覆盖配置文件设置')
    parser.add_argument('--custom-intensity', type=float, default=None,
                        help='自定义增强强度 (0.0-1.0)，会覆盖配置文件设置')
    parser.add_argument('--custom-sigma', type=int, default=None,
                        help='自定义增强平滑系数 (1-10)，会覆盖配置文件设置')

    args = parser.parse_args()

    # --- 执行逻辑 ---

    # 1. 加载训练超参数
    try:
        with open(args.hyp, 'r') as f:
            train_kwargs = yaml.safe_load(f)
        print(f"信息：已从 {args.hyp} 加载训练超参数。")
    except Exception as e:
        print(f"错误：无法加载超参数文件 {args.hyp}。原因: {e}")
        return

    # 2. 将核心配置添加到训练参数中
    train_kwargs['data'] = args.data
    
    # 2.5. 处理自定义增强参数
    if args.disable_custom_augment:
        train_kwargs['custom_augment'] = 0.0
        print("信息：已禁用自定义增强算法 (custom_augment=0.0)")
    
    # 应用命令行中指定的自定义增强参数（如果有的话）
    if args.custom_augment is not None:
        if not (0.0 <= args.custom_augment <= 1.0):
            print(f"警告：custom_augment 值 {args.custom_augment} 超出范围 [0.0, 1.0]，将使用配置文件默认值")
        else:
            train_kwargs['custom_augment'] = args.custom_augment
            print(f"信息：设置自定义增强概率为 {args.custom_augment}")
    
    if args.custom_intensity is not None:
        if not (0.0 <= args.custom_intensity <= 1.0):
            print(f"警告：custom_intensity 值 {args.custom_intensity} 超出范围 [0.0, 1.0]，将使用配置文件默认值")
        else:
            train_kwargs['custom_intensity'] = args.custom_intensity
            print(f"信息：设置自定义增强强度为 {args.custom_intensity}")
    
    if args.custom_sigma is not None:
        if not (1 <= args.custom_sigma <= 10):
            print(f"警告：custom_sigma 值 {args.custom_sigma} 超出范围 [1, 10]，将使用配置文件默认值")
        else:
            train_kwargs['custom_sigma'] = args.custom_sigma
            print(f"信息：设置自定义增强平滑系数为 {args.custom_sigma}")
    
    # 显示当前自定义增强配置
    current_custom_augment = train_kwargs.get('custom_augment', '未指定(使用配置文件默认值)')
    current_custom_intensity = train_kwargs.get('custom_intensity', '未指定(使用配置文件默认值)')
    current_custom_sigma = train_kwargs.get('custom_sigma', '未指定(使用配置文件默认值)')
    print(f"\n--- 当前自定义增强配置 ---")
    print(f"增强概率: {current_custom_augment}")
    print(f"增强强度: {current_custom_intensity}")
    print(f"平滑系数: {current_custom_sigma}")
    
    # 3. 初始化模型架构
    model = YOLO(args.model)
    print(f"信息：已从 {args.model} 加载模型架构。")

    # 4. 设置预训练权重 (如果指定了)
    if args.weights.lower() != 'none':
        train_kwargs['pretrained'] = args.weights
        print(f"信息：将使用预训练权重: {args.weights}")
    else:
        print("信息：将从头开始训练模型。")

    # 5. 开始训练
    try:
        print("\n--- 开始训练 ---")
        # 使用 **kwargs 将所有参数一次性传递给 train 方法
        model.train(**train_kwargs)
        print("\n--- 训练完成 ---")
    except Exception as e:
        print(f"\n训练过程中出现严重错误: {e}")

if __name__ == '__main__':
    # 确保已安装 pyyaml
    try:
        import yaml
    except ImportError:
        print("错误：PyYAML 未安装。请运行 'pip install pyyaml'。")
    else:
        main()