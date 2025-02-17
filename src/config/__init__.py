"""
配置模块初始化文件
根据环境变量选择加载开发或生产环境配置
"""
import os
from typing import Dict, Any

# 默认使用开发环境配置
ENV = os.getenv('SOLAR_ENV', 'dev')

if ENV == 'prod':
    from .prod_config import *
else:
    from .dev_config import *

def get_config() -> Dict[str, Any]:
    """
    获取当前环境的配置
    
    Returns:
        当前环境的配置字典
    """
    if ENV == 'prod':
        from .prod_config import (MODEL_CONFIG, TRAIN_CONFIG, API_CONFIG,
                                SYSTEM_CONFIG, MONITOR_CONFIG, LOG_CONFIG,
                                FEATURE_CONFIG, DATA_CONFIG, ENSEMBLE_CONFIG,
                                TEST_CONFIG)
    else:
        from .dev_config import (MODEL_CONFIG, TRAIN_CONFIG, API_CONFIG,
                               SYSTEM_CONFIG, MONITOR_CONFIG, LOG_CONFIG,
                               FEATURE_CONFIG, DATA_CONFIG, ENSEMBLE_CONFIG,
                               TEST_CONFIG)
    
    return {
        'model_config': MODEL_CONFIG,
        'train_config': TRAIN_CONFIG,
        'api_config': API_CONFIG,
        'system_config': SYSTEM_CONFIG,
        'monitor_config': MONITOR_CONFIG,
        'log_config': LOG_CONFIG,
        'feature_config': FEATURE_CONFIG,
        'data_config': DATA_CONFIG,
        'ensemble_config': ENSEMBLE_CONFIG,
        'test_config': TEST_CONFIG
    } 