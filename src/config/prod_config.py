"""
生产环境配置文件 - 用于正式环境
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

# 模型配置
MODEL_CONFIG = {
    "filternet": {
        "input_dim": 91,
        "hidden_dim": 64,
        "output_dim": 96,
        "num_layers": 4,
        "dropout": 0.1
    },
    "bilstm": {
        "input_dim": 91,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2
    },
    "xgboost": {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "objective": "reg:squarederror"
    }
}

# 训练配置
TRAIN_CONFIG = {
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "model_save_path": str(MODEL_DIR)
}

# API配置
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "workers": 2,
    "timeout": 120
}

# 系统配置
SYSTEM_CONFIG = {
    "max_memory_usage": 3.5 * 1024 * 1024 * 1024,  # 3.5GB
    "max_cpu_usage": 0.8,  # 80%
    "prediction_timeout": 0.5,  # 500ms
}

# 监控配置
MONITOR_CONFIG = {
    "metrics_port": 8001,
    "update_interval": 15,
    "prometheus_path": "/metrics",
    "alert_thresholds": {
        "memory_usage": 3.5 * 1024 * 1024 * 1024,  # 3.5GB
        "cpu_usage": 0.8,  # 80%
        "prediction_error": 0.15,  # MAE
        "prediction_latency": 0.5  # 500ms
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
            "backupCount": 5,
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

# 特征工程配置
FEATURE_CONFIG = {
    "time_features": [
        "hour",
        "day_of_week",
        "month",
        "is_holiday"
    ],
    "weather_features": [
        "temperature",
        "humidity",
        "irradiance",
        "wind_speed",
        "cloud_cover"
    ],
    "power_features": [
        "power_lag_1h",
        "power_lag_24h",
        "power_lag_48h"
    ],
    "feature_scaling": {
        "method": "robust",  # 或 "standard", "minmax"
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
        "irradiance",
        "wind_speed",
        "cloud_cover"
    ]
}

# 模型融合配置
ENSEMBLE_CONFIG = {
    "method": "weighted",  # 或 "stacking", "boosting"
    "weights": {
        "filternet": 0.4,
        "bilstm": 0.3,
        "xgboost": 0.3
    },
    "update_weights_interval": 24  # 小时
}

# 测试配置
TEST_CONFIG = {
    "test_data_path": str(DATA_DIR / "test"),
    "batch_size": 32,
    "metrics": [
        "mae",
        "rmse",
        "r2"
    ],
    "threshold": {
        "mae": 0.15,
        "rmse": 0.2,
        "r2": 0.9
    }
} 