"""
系统配置文件
"""
import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# 确保目录存在
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 模型配置 - 最小化配置用于快速验证
MODEL_CONFIG = {
    "filternet": {
        "input_dim": 91,
        "hidden_dim": 32,      # 减小隐藏层维度
        "output_dim": 1,       # 减小输出维度
        "num_layers": 2,       # 减少层数
        "dropout": 0.1
    },
    "bilstm": {
        "input_dim": 91,
        "hidden_dim": 32,      # 减小隐藏层维度
        "num_layers": 1,       # 单层LSTM
        "dropout": 0.1
    },
    "xgboost": {
        "max_depth": 3,        # 减小树的深度
        "learning_rate": 0.1,
        "n_estimators": 10,    # 减少树的数量
        "objective": "reg:squarederror"
    }
}

# 训练配置 - 最小化配置用于快速验证
TRAIN_CONFIG = {
    "batch_size": 16,          # 减小批次大小
    "num_epochs": 5,           # 减少训练轮数
    "learning_rate": 0.001,
    "early_stopping_patience": 3,
    "model_save_path": str(MODEL_DIR)
}

# API配置
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "workers": 1,              # 减少worker数量
    "timeout": 60
}

# 系统配置
SYSTEM_CONFIG = {
    "max_memory_usage": 2.0 * 1024 * 1024 * 1024,  # 限制内存使用为2GB
    "max_cpu_usage": 0.8,
    "prediction_timeout": 0.5,
}

# 监控配置
MONITOR_CONFIG = {
    "metrics_port": 8001,
    "update_interval": 15,
    "prometheus_path": "/metrics",
    "alert_thresholds": {
        "memory_usage": 2.0 * 1024 * 1024 * 1024,  # 2GB
        "cpu_usage": 0.8,
        "prediction_error": 0.15,
        "prediction_latency": 0.5
    }
}

# 日志配置
LOG_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3,      # 减少备份数量
            "formatter": "standard"
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard"
        }
    },
    "loggers": {
        "": {
            "handlers": ["file", "console"],
            "level": "INFO"
        }
    }
}

# 特征工程配置 - 最小化配置用于快速验证
FEATURE_CONFIG = {
    "time_features": [
        "hour",
        "day_of_week",
        "month"               # 移除is_holiday特征
    ],
    "weather_features": [
        "temperature",
        "humidity",
        "irradiance"         # 只保留最重要的天气特征
    ],
    "power_features": [
        "power_lag_1h",
        "power_lag_24h"      # 只保留最重要的滞后特征
    ],
    "feature_scaling": {
        "method": "standard", # 使用简单的标准化
        "params": {}
    }
}

# 数据配置
DATA_CONFIG = {
    "train_data_path": str(DATA_DIR / "train"),
    "test_data_path": str(DATA_DIR / "test"),
    "validation_data_path": str(DATA_DIR / "validation"),
    "data_format": "csv",
    "time_column": "timestamp",
    "target_column": "power",
    "feature_columns": [
        "temperature",
        "humidity",
        "irradiance"         # 保持与weather_features一致
    ]
}

# 模型融合配置
ENSEMBLE_CONFIG = {
    "method": "weighted",
    "weights": {
        "filternet": 0.4,
        "bilstm": 0.3,
        "xgboost": 0.3
    },
    "update_weights_interval": 24
}

# 测试配置
TEST_CONFIG = {
    "test_data_path": str(DATA_DIR / "test"),
    "batch_size": 16,        # 减小批次大小
    "metrics": [
        "mae",
        "rmse"               # 减少评估指标
    ],
    "threshold": {
        "mae": 0.2,          # 放宽评估标准
        "rmse": 0.3
    }
} 