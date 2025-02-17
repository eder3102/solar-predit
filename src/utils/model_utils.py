"""
模型工具类
"""
import torch
import logging
from typing import Dict
from pathlib import Path
import os

# 根据环境变量选择配置
if os.getenv('SOLAR_ENV') == 'prod':
    from src.config.prod_config import MODEL_CONFIG
else:
    from src.config.dev_config import MODEL_CONFIG

from src.models.ensemble.model import EnsembleModel

logger = logging.getLogger(__name__)

def get_model_info(model_dir: Path = None) -> Dict:
    """
    获取模型信息
    
    Args:
        model_dir: 模型目录，如果提供则加载已有模型
        
    Returns:
        模型信息字典
    """
    try:
        # 初始化模型
        model = EnsembleModel()
        if model_dir is not None:
            model.load_models(model_dir)
            
        # 获取各模型配置
        model_configs = {
            'filternet': model.models['filternet'].config,
            'bilstm': model.models['bilstm'].config,
            'xgboost': model.models['xgboost'].config
        }
        
        # 获取模型参数量
        model_params = {
            'filternet': model.models['filternet'].get_model_size(),
            'bilstm': model.models['bilstm'].get_model_size(),
            'xgboost': 'Not available before training'  # XGBoost在训练前无法获取大小
        }
        
        # 汇总信息
        info = {
            'model_configs': model_configs,
            'model_parameters': model_params,
            'ensemble_weights': model.weights
        }
        
        # 如果模型已训练，添加模型大小信息
        if model_dir is not None:
            try:
                model_sizes = model.get_model_sizes()
                total_size_mb = sum(size for size in model_sizes.values()) / (1024 * 1024)
                info['model_sizes'] = {
                    k: f"{v/1024/1024:.2f}MB" for k, v in model_sizes.items()
                }
                info['total_size'] = f"{total_size_mb:.2f}MB"
            except Exception as e:
                logger.warning(f"Could not get model sizes: {str(e)}")
                info['model_sizes'] = 'Not available'
                info['total_size'] = 'Not available'
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise 