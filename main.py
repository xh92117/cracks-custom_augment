#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import sys
import yaml
import argparse
from pathlib import Path

# 添加ultralytics模块到Python路径
sys.path.append(str(Path(__file__).parent / 'ultralytics'))

from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr


class CrackDetectionTrainer:
    """
    裂缝检测训练器类
    整合自定义数据增强和YOLO训练流程
    """
    
    def __init__(self, 
                 config_path="cfg/default.yaml",
                 model_config=None,
                 pretrained_model=None,
                 dataset_path=None):
        """
        初始化训练器
        
        Args:
            config_path (str): 配置文件路径
            model_config (str): 模型配置文件路径 
            pretrained_model (str): 预训练模型路径
            dataset_path (str): 数据集路径
        """
        self.config_path = config_path
        self.model_config = model_config or "models/yolo11n.yaml"
        self.pretrained_model = pretrained_model or "models/yolo11n.pt"
        self.dataset_path = dataset_path or "data/crack_dataset.yaml"
        
        # 验证路径
        self._validate_paths()
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化模型
        self.model = None
        
    def _validate_paths(self):
        """验证必要的路径是否存在"""
        LOGGER.info(f"验证配置路径...")
        
        # 检查配置文件
        if not os.path.exists(self.config_path):
            LOGGER.warning(f"配置文件不存在: {self.config_path}")
            
        # 创建必要的目录
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("runs", exist_ok=True)
        
        LOGGER.info("路径验证完成")
        
    def _load_config(self):
        """加载训练配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            LOGGER.info(f"成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            LOGGER.warning(f"加载配置文件失败: {e}")
            # 返回默认配置
            return self._get_default_config()
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'workers': 8,
            'device': '',
            'project': 'runs/detect',
            'name': 'crack_detection',
            'custom_augment': 0.5,
            'patience': 50,
            'save': True,
            'plots': True,
            'val': True
        }
    
    def setup_model(self):
        """设置YOLO模型"""
        try:
            # 检查是否有预训练模型
            if os.path.exists(self.pretrained_model):
                LOGGER.info(f"加载预训练模型: {self.pretrained_model}")
                self.model = YOLO(self.pretrained_model)
            elif os.path.exists(self.model_config):
                LOGGER.info(f"从配置创建新模型: {self.model_config}")
                self.model = YOLO(self.model_config)
            else:
                # 使用默认的YOLOv8n模型
                LOGGER.info("使用默认YOLOv8n模型")
                self.model = YOLO('yolo11n.pt')
                
        except Exception as e:
            LOGGER.error(f"模型设置失败: {e}")
            LOGGER.info("尝试下载默认模型...")
            self.model = YOLO('yolo11n.pt')
    
    def create_sample_dataset_config(self):
        """创建示例数据集配置文件"""
        dataset_config = {
            'path': '../datasets/crack_dataset',  # 数据集根目录
            'train': 'images/train',  # 训练图像相对路径
            'val': 'images/val',      # 验证图像相对路径
            'test': 'images/test',    # 测试图像相对路径
            
            'nc': 1,  # 类别数量
            'names': ['crack'],  # 类别名称
            
            # 可选的数据集描述
            'roboflow': {
                'workspace': 'your-workspace',
                'project': 'crack-detection',
                'version': 1
            }
        }
        
        # 保存示例配置
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        LOGGER.info(f"创建示例数据集配置: {self.dataset_path}")
        return dataset_config
    
    def train(self):
        """开始训练"""
        if self.model is None:
            LOGGER.error("模型未初始化，请先调用 setup_model()")
            return
        
        # 检查数据集配置
        if not os.path.exists(self.dataset_path):
            LOGGER.warning(f"数据集配置不存在: {self.dataset_path}")
            self.create_sample_dataset_config()
            LOGGER.warning("请配置您的数据集路径后重新运行训练")
            return
        
        try:
            LOGGER.info(colorstr('bright_blue', 'bold', '开始YOLO裂缝检测训练...'))
            
            # 训练参数
            train_args = {
                'data': self.dataset_path,
                'epochs': self.config.get('epochs', 100),
                'batch': self.config.get('batch', 16),
                'imgsz': self.config.get('imgsz', 640),
                'workers': self.config.get('workers', 8),
                'device': self.config.get('device', ''),
                'project': self.config.get('project', 'runs/detect'),
                'name': self.config.get('name', 'crack_detection'),
                'patience': self.config.get('patience', 50),
                'save': self.config.get('save', True),
                'plots': self.config.get('plots', True),
                'val': self.config.get('val', True),
                'amp': self.config.get('amp', True),
                'verbose': self.config.get('verbose', True),
                'seed': self.config.get('seed', 0),
                'deterministic': self.config.get('deterministic', True),
                'resume': self.config.get('resume', False),
                'optimizer': self.config.get('optimizer', 'auto'),
                'custom_augment': self.config.get('custom_augment', 0.5),  # 自定义增强参数
            }
            
            # 开始训练
            results = self.model.train(**train_args)
            
            LOGGER.info(colorstr('bright_green', 'bold', '训练完成!'))
            return results
            
        except Exception as e:
            LOGGER.error(f"训练过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO裂缝检测训练脚本')
    parser.add_argument('--config', type=str, default='cfg/default.yaml',
                        help='训练配置文件路径')
    parser.add_argument('--model', type=str, default=None,
                        help='模型配置文件路径 (例如: models/yolo11n.yaml)')
    parser.add_argument('--weights', type=str, default=None,
                        help='预训练权重路径 (例如: models/yolo11n.pt)')
    parser.add_argument('--data', type=str, default=None,
                        help='数据集配置文件路径 (例如: data/crack_dataset.yaml)')
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print('='*60)
    print('🚀 YOLO 裂缝检测训练系统')
    print('='*60)
    print(f"📁 配置文件: {args.config}")
    print(f"🤖 模型配置: {args.model or '默认'}")
    print(f"⚖️  预训练权重: {args.weights or '默认'}")
    print(f"📊 数据集配置: {args.data or '默认'}")
    print('='*60 + '\n')
    
    # 创建训练器
    trainer = CrackDetectionTrainer(
        config_path=args.config,
        model_config=args.model,
        pretrained_model=args.weights,
        dataset_path=args.data
    )
    
    # 设置模型
    trainer.setup_model()
    
    # 开始训练
    trainer.train()
    
    print('✅ 任务完成!')


if __name__ == '__main__':
    main() 