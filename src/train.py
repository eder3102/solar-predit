"""
模型训练主程序
"""
import logging
import torch
import numpy as np
from pathlib import Path
import argparse
import os

# 根据环境变量选择配置
if os.getenv('SOLAR_ENV') == 'prod':
    from src.config.prod_config import TRAIN_CONFIG, DATA_CONFIG
else:
    from src.config.dev_config import TRAIN_CONFIG, DATA_CONFIG

from src.utils.data_processor import DataProcessor
from src.utils.dataset import DatasetFactory
from src.models.trainer import ModelTrainer
from src.utils.model_utils import get_model_info

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模型训练程序')
    parser.add_argument('--train_data', type=str, default=DATA_CONFIG['train_data_path'],
                      help='训练数据路径')
    parser.add_argument('--val_data', type=str, default=DATA_CONFIG['validation_data_path'],
                      help='验证数据路径')
    parser.add_argument('--model_dir', type=str, default=TRAIN_CONFIG['model_save_path'],
                      help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'],
                      help='批次大小')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'],
                      help='训练轮数')
    parser.add_argument('--show_model_info', action='store_true',
                      help='是否显示模型信息')
    return parser.parse_args()

def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_args()
        
        # 初始化数据处理器
        data_processor = DataProcessor()
        
        # 加载训练数据
        logger.info("Loading training data...")
        train_df = data_processor.load_data(args.train_data)
        train_df = data_processor.prepare_data(train_df, is_training=True)
        train_features, train_targets = data_processor.split_features_target(train_df)
        
        # 加载验证数据
        logger.info("Loading validation data...")
        val_df = data_processor.load_data(args.val_data)
        val_df = data_processor.prepare_data(val_df, is_training=False)
        val_features, val_targets = data_processor.split_features_target(val_df)
        
        # 创建数据加载器
        logger.info("Creating data loaders...")
        # 创建序列数据加载器用于深度学习模型
        train_seq_loader, _ = DatasetFactory.create_dataloaders(
            train_features, train_targets,
            batch_size=args.batch_size,
            validation_split=0.0  # 使用单独的验证集
        )
        
        val_seq_loader = DatasetFactory.create_test_dataloader(
            val_features, val_targets,
            batch_size=args.batch_size
        )
        
        # 创建单步数据加载器用于XGBoost
        train_step_loader = DatasetFactory.create_single_step_loader(
            train_features, train_targets,
            batch_size=args.batch_size
        )
        
        val_step_loader = DatasetFactory.create_single_step_loader(
            val_features, val_targets,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        # 初始化训练器
        logger.info("Initializing trainer...")
        trainer = ModelTrainer()
        
        # 显示模型信息
        if args.show_model_info:
            logger.info("Model information:")
            model_info = get_model_info()
            for key, value in model_info.items():
                logger.info(f"{key}: {value}")
        
        # 开始训练
        logger.info("Starting training...")
        trainer.train(
            train_loader=train_seq_loader,
            val_loader=val_seq_loader,
            save_dir=args.model_dir
        )
        
        # 显示训练后的模型信息
        if args.show_model_info:
            logger.info("Model information after training:")
            model_info = get_model_info(Path(args.model_dir))
            for key, value in model_info.items():
                logger.info(f"{key}: {value}")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 