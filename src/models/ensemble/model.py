"""
模型融合实现
"""
import torch
import numpy as np
from typing import Dict, List, Union
import logging
from pathlib import Path
import os

# 根据环境变量选择配置
if os.getenv('SOLAR_ENV') == 'prod':
    from src.config.prod_config import ENSEMBLE_CONFIG
else:
    from src.config.dev_config import ENSEMBLE_CONFIG

from src.models.filternet.model import FilterNet
from src.models.bilstm.model import BiLSTM
from src.models.xgboost.model import XGBoostModel
from src.utils.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class EnsembleModel:
    """模型融合类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化模型融合
        
        Args:
            config: 融合配置字典
        """
        # 使用默认配置或传入配置
        self.config = config or ENSEMBLE_CONFIG
        
        # 初始化子模型
        self.models = {
            'filternet': FilterNet(),
            'bilstm': BiLSTM(),
            'xgboost': XGBoostModel()
        }
        
        # 模型权重
        self.weights = self.config['weights']
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator()
        
        logger.info(f"Initialized Ensemble with config: {self.config}")
        
    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        融合预测
        
        Args:
            features: 输入特征
            
        Returns:
            融合预测结果
        """
        try:
            predictions = {}
            
            # FilterNet预测
            if isinstance(features, np.ndarray):
                features_torch = torch.FloatTensor(features)
            else:
                features_torch = features
                
            with torch.no_grad():
                self.models['filternet'].eval()
                predictions['filternet'] = self.models['filternet'](
                    features_torch).cpu().numpy()
                    
            # BiLSTM预测
            with torch.no_grad():
                self.models['bilstm'].eval()
                predictions['bilstm'] = self.models['bilstm'](
                    features_torch).cpu().numpy()
                    
            # XGBoost预测
            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
            else:
                features_np = features
                
            predictions['xgboost'] = self.models['xgboost'].predict(
                features_np).reshape(-1, 1)
                
            # 加权融合
            weighted_sum = np.zeros_like(predictions['filternet'])
            for model_name, pred in predictions.items():
                weighted_sum += pred * self.weights[model_name]
                
            return weighted_sum
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            raise
            
    def update_weights(self, val_predictions: Dict[str, np.ndarray],
                      val_targets: np.ndarray):
        """
        更新模型权重
        
        Args:
            val_predictions: 各模型在验证集上的预测结果
            val_targets: 验证集目标值
        """
        try:
            # 计算各模型的MAE
            scores = {}
            for model_name, preds in val_predictions.items():
                metrics = self.metrics_calculator.calculate_metrics(
                    val_targets, preds, metrics=['mae'])
                scores[model_name] = metrics['mae']
                
            # 计算新权重(MAE越小权重越大)
            total_error = sum(1/score for score in scores.values())
            new_weights = {
                model: (1/score)/total_error 
                for model, score in scores.items()
            }
            
            # 更新权重
            self.weights = new_weights
            logger.info(f"Updated ensemble weights: {self.weights}")
            
        except Exception as e:
            logger.error(f"Error updating ensemble weights: {str(e)}")
            raise
            
    def save_models(self, save_dir: Union[str, Path]):
        """
        保存所有模型
        
        Args:
            save_dir: 保存目录
        """
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存深度学习模型
            torch.save(self.models['filternet'].state_dict(),
                      save_dir / 'filternet.pth')
            torch.save(self.models['bilstm'].state_dict(),
                      save_dir / 'bilstm.pth')
                      
            # 保存XGBoost模型
            self.models['xgboost'].save_model(save_dir / 'xgboost.pkl')
            
            # 保存权重
            np.save(save_dir / 'weights.npy', self.weights)
            
            logger.info(f"Saved all models to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
            
    def load_models(self, model_dir: Union[str, Path]):
        """
        加载所有模型
        
        Args:
            model_dir: 模型目录
        """
        try:
            model_dir = Path(model_dir)
            
            # 加载深度学习模型
            self.models['filternet'].load_state_dict(
                torch.load(model_dir / 'filternet.pth'))
            self.models['bilstm'].load_state_dict(
                torch.load(model_dir / 'bilstm.pth'))
                
            # 加载XGBoost模型
            self.models['xgboost'].load_model(model_dir / 'xgboost.pkl')
            
            # 加载权重
            self.weights = np.load(model_dir / 'weights.npy',
                                 allow_pickle=True).item()
                                 
            logger.info(f"Loaded all models from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
            
    def get_model_sizes(self) -> Dict[str, int]:
        """
        获取所有模型大小
        
        Returns:
            各模型大小的字典
        """
        try:
            sizes = {
                'filternet': self.models['filternet'].get_model_size(),
                'bilstm': self.models['bilstm'].get_model_size(),
                'xgboost': self.models['xgboost'].get_model_size()
            }
            return sizes
        except Exception as e:
            logger.error(f"Error getting model sizes: {str(e)}")
            raise 