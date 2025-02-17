"""
数据集类，用于数据加载和批处理
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import logging
import os

# 根据环境变量选择配置
if os.getenv('SOLAR_ENV') == 'prod':
    from src.config.prod_config import TRAIN_CONFIG
else:
    from src.config.dev_config import TRAIN_CONFIG

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 sequence_length: int = 2):  # 减少序列长度
        """
        初始化数据集
        
        Args:
            features: 特征数组
            targets: 目标变量数组
            sequence_length: 序列长度
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
        # 记录特征维度
        self.feature_dim = features.shape[1]
        logger.info(f"TimeSeriesDataset initialized with feature_dim: {self.feature_dim}")
        
    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.features) - self.sequence_length + 1
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            特征序列和目标值的元组
        """
        feature_seq = self.features[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length - 1]
        return feature_seq, target
        
class SingleStepDataset(Dataset):
    """单步预测数据集类"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        初始化数据集
        
        Args:
            features: 特征数组
            targets: 目标变量数组
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            特征和目标值的元组
        """
        return self.features[idx], self.targets[idx]
        
class DatasetFactory:
    """数据集工厂类"""
    
    @staticmethod
    def create_dataloaders(
        features: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        sequence_length: int = 6,  # 减少序列长度以适应小数据集
        shuffle: bool = True,
        validation_split: float = 0.2,
        num_workers: int = 0  # 开发环境使用单线程
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        创建数据加载器
        
        Args:
            features: 特征数组
            targets: 目标变量数组
            batch_size: 批次大小
            sequence_length: 序列长度
            shuffle: 是否打乱数据
            validation_split: 验证集比例
            num_workers: 数据加载线程数
            
        Returns:
            训练集和验证集的DataLoader元组
        """
        try:
            # 创建完整数据集
            dataset = TimeSeriesDataset(features, targets, sequence_length)
            
            # 计算验证集大小
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            
            # 分割数据集
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ) if val_size > 0 else None
            
            logger.info(f"Created dataloaders - Train size: {train_size}, "
                       f"Validation size: {val_size}")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Error creating dataloaders: {str(e)}")
            raise
            
    @staticmethod
    def create_test_dataloader(
        features: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        sequence_length: int = 6,  # 减少序列长度以适应小数据集
        num_workers: int = 0  # 开发环境使用单线程
    ) -> DataLoader:
        """
        创建测试数据加载器
        
        Args:
            features: 特征数组
            targets: 目标变量数组
            batch_size: 批次大小
            sequence_length: 序列长度
            num_workers: 数据加载线程数
            
        Returns:
            测试集DataLoader
        """
        try:
            dataset = TimeSeriesDataset(features, targets, sequence_length)
            
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            
            logger.info(f"Created test dataloader - Size: {len(dataset)}")
            
            return test_loader
            
        except Exception as e:
            logger.error(f"Error creating test dataloader: {str(e)}")
            raise
            
    @staticmethod
    def create_single_step_loader(
        features: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0  # 开发环境使用单线程
    ) -> DataLoader:
        """
        创建单步预测数据加载器
        
        Args:
            features: 特征数组
            targets: 目标变量数组
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载线程数
            
        Returns:
            数据加载器
        """
        try:
            dataset = SingleStepDataset(features, targets)
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
            
            logger.info(f"Created single step dataloader - Size: {len(dataset)}")
            
            return loader
            
        except Exception as e:
            logger.error(f"Error creating single step dataloader: {str(e)}")
            raise 