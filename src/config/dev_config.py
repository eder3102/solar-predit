"""
开发环境配置文件 - 用于快速验证
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
        "input_dim": 29,       # 更新为实际特征维度：18个基础特征 + 8个周期特征 + 3个交互特征
        "hidden_dim": 128,     # 增加隐藏层维度
        "output_dim": 1,
        "num_layers": 3,       # 增加层数
        "dropout": 0.2
    },
    "bilstm": {
        "input_dim": 29,       # 与FilterNet相同
        "hidden_dim": 128,     # 增加隐藏层维度
        "num_layers": 2,
        "dropout": 0.2
    },
    "xgboost": {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,   # 增加树的数量
        "objective": "reg:squarederror",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist"  # 使用直方图方法以节省内存
    }
}

# 训练配置
TRAIN_CONFIG = {
    # 基础配置
    "batch_size": 64,
    "learning_rate": {
        "stage1": {
            "initial": 0.0001,  # 降低初始学习率
            "min": 1e-6,
            "factor": 0.5,
            "patience": 5
        },
        "stage2": {
            "initial": 0.00005,  # 更小的学习率用于融合训练
            "min": 1e-6,
            "factor": 0.5,
            "patience": 5
        },
        "stage3": {
            "initial": 0.00001,  # 最小的学习率用于微调
            "min": 1e-7,
            "factor": 0.5,
            "patience": 5
        }
    },
    
    # 训练阶段配置
    "stage1_epochs": 50,      # 单模型训练轮数
    "stage2_epochs": 30,      # 固定权重训练轮数
    "stage3_start_epoch": 80, # 动态权重训练开始轮数
    "max_epochs": 150,        # 最大训练轮数
    
    # 早停配置
    "early_stopping_patience": {
        "stage1": 10,
        "stage2": 8,
        "stage3": 15
    },
    
    # 模型权重配置
    "initial_weights": {      # 第二阶段的固定权重
        "filternet": 0.4,
        "bilstm": 0.4,
        "xgboost": 0.2
    },
    
    # 正则化配置
    "l1_weight": 0.001,      # 降低L1正则化权重
    
    # 权重更新配置
    "weight_update": {
        "smoothing_factor": 0.8,  # 权重平滑系数
        "min_weight": 0.1,        # 最小权重限制
    },
    
    # 其他配置
    "model_save_path": str(MODEL_DIR),
    "gradient_clip": 1.0,
}

# API配置
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,             # 开发环境开启debug
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
            "level": "DEBUG"       # 开发环境使用DEBUG级别
        }
    }
}

# 特征工程配置
FEATURE_CONFIG = {
    "time_features": [
        "hour",
        "day_of_week",
        "month",
        "day_of_year"
    ],
    "weather_features": [
        "temperature",
        "total_irradiance",
        "wind_speed",
        "direct_irradiance",
        "diffuse_irradiance",
        "relative_humidity",
        "pressure",
        "cloud_type"
    ],
    "solar_features": [
        "solar_elevation",
        "dni_ratio",
        "dew_point",
        "air_mass",
        "clearness_index",
        "cell_temperature"
    ],
    "derived_features": {
        "periodic": [
            "hour_sin", "hour_cos",
            "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos",
            "day_of_year_sin", "day_of_year_cos"
        ],
        "interactions": [
            "irradiance_temp",
            "clearness_elevation",
            "wind_chill"
        ]
    },
    "feature_scaling": {
        "method": "robust",
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
    "sequence_length": 48,  # 24小时（30分钟间隔）
    "feature_columns": [
        # 基础气象特征
        "temperature", "total_irradiance", "wind_speed",
        "direct_irradiance", "diffuse_irradiance",
        "relative_humidity", "pressure", "cloud_type",
        # 太阳能特征
        "solar_elevation", "dni_ratio", "dew_point",
        "air_mass", "clearness_index", "cell_temperature",
        # 时间特征
        "hour", "day_of_week", "month", "day_of_year"
    ]
}

# 模型融合配置
ENSEMBLE_CONFIG = {
    "method": "weighted",
    "weights": {
        "filternet": 0.4,    # 调整权重
        "bilstm": 0.3,
        "xgboost": 0.3
    },
    "update_weights_interval": 48  # 24小时（30分钟间隔）
}

# 测试配置
TEST_CONFIG = {
    "test_data_path": str(DATA_DIR / "test"),
    "batch_size": 128,        # 增加测试批次大小
    "metrics": [
        "mae",
        "rmse",
        "r2",
        "mape"
    ],
    "threshold": {
        "mae": 0.15,          # 保持评估标准
        "rmse": 0.2,
        "r2": 0.85,
        "mape": 15.0
    }
} 