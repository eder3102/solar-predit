"""
数据预处理和特征工程工具类
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from typing import List, Dict, Union, Tuple, Optional
import logging
from pathlib import Path
import os
import joblib

# 根据环境变量选择配置
if os.getenv('SOLAR_ENV') == 'prod':
    from src.config.prod_config import FEATURE_CONFIG, DATA_CONFIG
else:
    from src.config.dev_config import FEATURE_CONFIG, DATA_CONFIG

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据预处理和特征工程类"""
    
    def __init__(self):
        """初始化数据处理器"""
        self.scalers = {}
        
    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            加载的数据DataFrame
        """
        try:
            data_path = Path(data_path)
            if data_path.is_dir():
                data_path = data_path / 'data.csv'
            
            df = pd.read_csv(data_path)
            df[DATA_CONFIG['time_column']] = pd.to_datetime(df[DATA_CONFIG['time_column']])
            df.set_index(DATA_CONFIG['time_column'], inplace=True)
            logger.info(f"Successfully loaded data from {data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
            raise
            
    def generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成时间特征
        
        Args:
            df: 输入数据DataFrame
            
        Returns:
            添加时间特征后的DataFrame
        """
        df_copy = df.copy()
        
        # 提取时间特征
        for feature in FEATURE_CONFIG['time_features']:
            if feature == 'hour':
                df_copy['hour'] = df_copy.index.hour
            elif feature == 'day_of_week':
                df_copy['day_of_week'] = df_copy.index.dayofweek
            elif feature == 'month':
                df_copy['month'] = df_copy.index.month
        
        return df_copy
        
    def generate_power_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成发电量相关特征
        
        Args:
            df: 输入数据DataFrame
            
        Returns:
            添加发电量特征后的DataFrame
        """
        df_copy = df.copy()
        target = DATA_CONFIG['target_column']
        
        # 添加滞后特征
        for feature in FEATURE_CONFIG['power_features']:
            if feature == 'power_lag_1h':
                df_copy[feature] = df_copy[target].shift(1)
            elif feature == 'power_lag_24h':
                df_copy[feature] = df_copy[target].shift(24)
        
        return df_copy
        
    def scale_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        特征缩放
        
        Args:
            df: 输入数据DataFrame
            is_training: 是否为训练阶段
            
        Returns:
            特征缩放后的DataFrame
        """
        df_copy = df.copy()
        
        if is_training:
            for column in DATA_CONFIG['feature_columns']:
                scaler = StandardScaler()
                df_copy[column] = scaler.fit_transform(df_copy[[column]])
                self.scalers[column] = scaler
        else:
            for column in DATA_CONFIG['feature_columns']:
                if column in self.scalers:
                    df_copy[column] = self.scalers[column].transform(df_copy[[column]])
                    
        return df_copy
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 输入数据DataFrame
            
        Returns:
            处理缺失值后的DataFrame
        """
        df_copy = df.copy()
        
        # 对数值型列使用前向填充
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
        df_copy[numeric_columns] = df_copy[numeric_columns].fillna(method='ffill')
        
        # 对剩余的缺失值使用后向填充
        df_copy = df_copy.fillna(method='bfill')
        
        return df_copy
        
    def save_scalers(self, save_dir: Union[str, Path]):
        """保存特征缩放器"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scalers, save_dir / 'scalers.pkl')
        logger.info(f"Saved scalers to {save_dir}")

    def load_scalers(self, model_dir: Union[str, Path]):
        """加载特征缩放器"""
        model_dir = Path(model_dir)
        
        self.scalers = joblib.load(model_dir / 'scalers.pkl')
        logger.info(f"Loaded scalers from {model_dir}")

    def prepare_data(self, df: pd.DataFrame, is_training: bool = True, model_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        数据准备主函数
        
        Args:
            df: 输入数据DataFrame
            is_training: 是否为训练阶段
            model_dir: 模型目录，用于加载scaler
            
        Returns:
            处理后的DataFrame
        """
        try:
            # 处理缺失值
            df = self.handle_missing_values(df)
            
            # 添加周期性特征编码
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            
            # 特征列表
            feature_columns = [
                'temperature', 'total_irradiance', 'wind_speed', 'direct_irradiance',
                'diffuse_irradiance', 'relative_humidity', 'pressure', 'cloud_type',
                'solar_elevation', 'dni_ratio', 'dew_point', 'air_mass',
                'clearness_index', 'cell_temperature',
                'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos'
            ]
            
            # 确保所有特征列都存在
            existing_columns = []
            for col in feature_columns:
                if col in df.columns:
                    existing_columns.append(col)
                else:
                    logger.warning(f"列 {col} 不存在于数据中")
            
            # 添加目标变量
            if DATA_CONFIG['target_column'] in df.columns:
                existing_columns.append(DATA_CONFIG['target_column'])
            
            # 使用存在的列
            df = df[existing_columns]
            
            # 特征缩放
            if is_training:
                df = self.scale_features(df, is_training=True)
            else:
                if model_dir is not None:
                    self.load_scalers(model_dir)
                df = self.scale_features(df, is_training=False)
            
            # 删除包含NaN的行
            df = df.dropna()
            
            logger.info("Data preparation completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
            
    def split_features_target(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        分离特征和目标变量
        
        Args:
            df: 输入数据DataFrame
            
        Returns:
            特征数组和目标变量数组的元组
        """
        # 使用所有非目标列作为特征
        feature_columns = [col for col in df.columns if col != DATA_CONFIG['target_column']]
        features = df[feature_columns].values
        target = df[DATA_CONFIG['target_column']].values
        return features, target 