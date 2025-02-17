"""
将best_model.pth转换为各个子模型的文件
"""
import torch
import numpy as np
from pathlib import Path
import logging
import os

from src.models.ensemble.model import EnsembleModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def convert_model():
    """转换模型文件"""
    try:
        # 加载best_model.pth
        logger.info("加载best_model.pth...")
        state_dict = torch.load('models/best_model.pth')
        
        # 打印状态字典的键
        logger.info("状态字典的键:")
        for key in state_dict.keys():
            logger.info(f"- {key}")
        
        # 创建新的模型实例
        model = EnsembleModel()
        
        # 提取各个子模型的状态字典
        filternet_state = {k.replace('models.filternet.', ''): v 
                          for k, v in state_dict.items() 
                          if k.startswith('models.filternet.')}
        
        bilstm_state = {k.replace('models.bilstm.', ''): v 
                        for k, v in state_dict.items() 
                        if k.startswith('models.bilstm.')}
        
        # 保存各个子模型
        logger.info("保存子模型文件...")
        torch.save(filternet_state, 'models/filternet.pth')
        torch.save(bilstm_state, 'models/bilstm.pth')
        
        # 提取XGBoost模型
        xgboost_state = {k.replace('models.xgboost.', ''): v 
                         for k, v in state_dict.items() 
                         if k.startswith('models.xgboost.')}
        
        logger.info("XGBoost状态字典的键:")
        for key in xgboost_state.keys():
            logger.info(f"- {key}")
            
        # 保存XGBoost模型
        if 'booster' in xgboost_state:
            model.models['xgboost'].model = xgboost_state['booster']
            model.models['xgboost'].save_model('models/xgboost.pkl')
        else:
            logger.warning("未找到XGBoost模型，使用默认模型")
            model.models['xgboost'].save_model('models/xgboost.pkl')
        
        # 提取并保存权重
        weights = {
            'filternet': 0.4,
            'bilstm': 0.3,
            'xgboost': 0.3
        }
        np.save('models/weights.npy', weights)
        
        logger.info("模型转换完成!")
        
    except Exception as e:
        logger.error(f"转换出错: {str(e)}")
        raise

if __name__ == '__main__':
    convert_model() 