"""
XGBoost模型实现
"""
import xgboost as xgb
import numpy as np
from typing import Dict, Union, Optional
import logging
import joblib
from pathlib import Path
import os

# 根据环境变量选择配置
if os.getenv('SOLAR_ENV') == 'prod':
    from src.config.prod_config import MODEL_CONFIG
else:
    from src.config.dev_config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class XGBoostModel:
    """XGBoost模型类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化XGBoost模型
        
        Args:
            config: 模型配置字典
        """
        # 使用默认配置或传入配置
        self.config = config or MODEL_CONFIG['xgboost']
        
        # 初始化模型
        self.model = xgb.XGBRegressor(
            max_depth=self.config['max_depth'],
            learning_rate=self.config['learning_rate'],
            n_estimators=self.config['n_estimators'],
            objective=self.config['objective'],
            n_jobs=-1,  # 使用所有CPU核心
            random_state=42
        )
        
        logger.info(f"Initialized XGBoost with config: {self.config}")
        
    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[list] = None) -> Dict[str, list]:
        """
        训练模型
        
        Args:
            X: 特征数组
            y: 目标变量数组
            eval_set: 评估数据集
            
        Returns:
            训练历史
        """
        try:
            # 训练模型
            self.model.fit(
                X, y,
                eval_set=[(X, y)] + (eval_set if eval_set else []),
                eval_metric='rmse',
                early_stopping_rounds=10,
                verbose=False  # 设置为False以减少输出
            )
            
            # 获取训练历史
            results = {
                'train_score': self.model.evals_result_['validation_0']['rmse'][-1]
            }
            
            # 如果有验证集，添加验证分数
            if eval_set:
                results['val_score'] = self.model.evals_result_['validation_1']['rmse'][-1]
            
            logger.info(f"XGBoost training completed. Final RMSE - Train: {results['train_score']:.6f}" + 
                       (f", Validation: {results['val_score']:.6f}" if 'val_score' in results else ""))
            return results
            
        except Exception as e:
            logger.error(f"Error training XGBoost: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        模型预测
        
        Args:
            X: 特征数组
            
        Returns:
            预测结果数组
        """
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {str(e)}")
            raise
            
    def save_model(self, path: Union[str, Path]):
        """
        保存模型
        
        Args:
            path: 模型保存路径
        """
        try:
            joblib.dump(self.model, path)
            logger.info(f"Saved XGBoost model to {path}")
        except Exception as e:
            logger.error(f"Error saving XGBoost model: {str(e)}")
            raise
            
    def load_model(self, path: Union[str, Path]):
        """
        加载模型
        
        Args:
            path: 模型加载路径
        """
        try:
            self.model = joblib.load(path)
            logger.info(f"Loaded XGBoost model from {path}")
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {str(e)}")
            raise
            
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        try:
            importance = self.model.feature_importances_
            scores = {f"feature_{i}": score 
                     for i, score in enumerate(importance)}
            return dict(sorted(scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True))
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise
            
    def get_model_size(self) -> int:
        """
        获取模型大小(字节)
        
        Returns:
            模型大小
        """
        try:
            return self.model.get_booster().save_raw().nbytes
        except Exception as e:
            logger.error(f"Error getting model size: {str(e)}")
            raise 