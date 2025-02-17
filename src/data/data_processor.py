import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 根据环境变量选择配置
if os.getenv('SOLAR_ENV') == 'prod':
    from src.config.prod_config import FEATURE_CONFIG, DATA_CONFIG
else:
    from src.config.dev_config import FEATURE_CONFIG, DATA_CONFIG

class DataProcessor:
    def __init__(self):
        # 从配置文件获取特征列表
        self.base_feature_columns = DATA_CONFIG['feature_columns']
        self.target_column = DATA_CONFIG['target_column']
        self.scaler = StandardScaler()

    def _handle_missing_values(self, df):
        """处理缺失值"""
        df = df.copy()
        
        # 1. 对时间相关特征使用前向填充
        time_features = ['hour', 'day_of_week', 'month', 'day_of_year']
        df[time_features] = df[time_features].fillna(method='ffill')
        
        # 2. 对天气特征使用更智能的填充策略
        # 温度：使用同一时间点的历史平均值
        if 'temperature' in df.columns and df['temperature'].isnull().any():
            df['temperature'] = df.groupby(['hour', 'month'])['temperature'].transform(
                lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 0)
            )
        
        # 辐照度相关特征：基于太阳高度角和云量估算
        irradiance_features = ['total_irradiance', 'direct_irradiance', 'diffuse_irradiance']
        for feature in irradiance_features:
            if feature in df.columns and df[feature].isnull().any():
                # 使用太阳高度角和时间特征的相关性
                df[feature] = df.groupby(['hour', 'month'])[feature].transform(
                    lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 0)
                )
                # 夜间（太阳高度角<=0）的辐照度设为0
                if 'solar_elevation' in df.columns:
                    night_mask = df['solar_elevation'] <= 0
                    df.loc[night_mask, feature] = 0
        
        # 3. 对其他特征使用更复杂的填充策略
        # 相对湿度：使用温度和露点温度的关系
        if 'relative_humidity' in df.columns and df['relative_humidity'].isnull().any():
            df['relative_humidity'] = df.groupby(['hour', 'month'])['relative_humidity'].transform(
                lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 50)  # 使用50%作为默认值
            )
            # 确保在0-100范围内
            df['relative_humidity'] = df['relative_humidity'].clip(0, 100)
        
        # 风速：使用时间和季节的相关性
        if 'wind_speed' in df.columns and df['wind_speed'].isnull().any():
            df['wind_speed'] = df.groupby(['hour', 'month'])['wind_speed'].transform(
                lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 2)  # 使用2m/s作为默认值
            )
            # 确保非负
            df['wind_speed'] = df['wind_speed'].clip(lower=0)
        
        # 云量：使用时间和季节的相关性
        if 'cloud_type' in df.columns and df['cloud_type'].isnull().any():
            df['cloud_type'] = df.groupby(['hour', 'month'])['cloud_type'].transform(
                lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 5)  # 使用5作为默认值
            )
            df['cloud_type'] = df['cloud_type'].round().clip(0, 9)
        
        # DNI比例：基于直射辐照和总辐照计算
        if 'dni_ratio' in df.columns and df['dni_ratio'].isnull().any():
            if 'direct_irradiance' in df.columns and 'total_irradiance' in df.columns:
                df['dni_ratio'] = df['direct_irradiance'] / (df['total_irradiance'] + 1e-6)
                df['dni_ratio'] = df['dni_ratio'].clip(0, 1)
            else:
                df['dni_ratio'] = df.groupby(['hour', 'month'])['dni_ratio'].transform(
                    lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 0.5)  # 使用0.5作为默认值
                )
        
        # 4. 对于仍然存在的缺失值，使用前向填充和后向填充的组合
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 5. 对于仍然存在的缺失值，使用列的平均值
        df = df.fillna(df.mean())
        
        # 6. 最后，使用0填充任何剩余的缺失值
        df = df.fillna(0)
        
        return df

    def _add_periodic_features(self, df):
        """添加周期性特征编码"""
        df = df.copy()
        
        # 从配置文件获取周期性特征定义
        periodic_features = FEATURE_CONFIG['derived_features']['periodic']
        
        # 小时的周期性编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 星期的周期性编码
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 月份的周期性编码
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 一年中的天数的周期性编码
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df

    def _add_interaction_features(self, df):
        """添加特征交互项"""
        df = df.copy()
        
        # 从配置文件获取交互特征定义
        interaction_features = FEATURE_CONFIG['derived_features']['interactions']
        
        # 辐照度和温度的交互
        if 'total_irradiance' in df.columns and 'temperature' in df.columns:
            df['irradiance_temp'] = df['total_irradiance'] * df['temperature']
        
        # 晴空指数和太阳高度角的交互
        if 'clearness_index' in df.columns and 'solar_elevation' in df.columns:
            df['clearness_elevation'] = df['clearness_index'] * df['solar_elevation']
        
        # 风寒指数（考虑风速和温度的影响）
        if 'wind_speed' in df.columns and 'temperature' in df.columns:
            df['wind_chill'] = 13.12 + 0.6215 * df['temperature'] - 11.37 * (df['wind_speed'] ** 0.16) + 0.3965 * df['temperature'] * (df['wind_speed'] ** 0.16)
        
        return df

    def prepare_data(self, df, model_dir=None):
        """准备数据，包括特征选择和标准化"""
        try:
            # 检查缺失列
            missing_columns = [col for col in self.base_feature_columns + [self.target_column] if col not in df.columns]
            if missing_columns:
                logging.warning(f"Missing columns in data: {missing_columns}")
            
            # 处理缺失值
            df = self._handle_missing_values(df)
            
            # 添加衍生特征
            df = self._add_periodic_features(df)
            df = self._add_interaction_features(df)
            
            # 获取所有特征列
            feature_columns = (
                self.base_feature_columns +
                FEATURE_CONFIG['derived_features']['periodic'] +
                FEATURE_CONFIG['derived_features']['interactions']
            )
            
            # 选择特征和目标
            X = df[feature_columns]
            y = df[self.target_column]
            
            # 如果提供了model_dir，尝试加载已保存的scaler
            if model_dir:
                scaler_path = Path(model_dir) / 'scaler.joblib'
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)
                    X_scaled = self.scaler.transform(X)
                else:
                    logging.warning(f"No saved scaler found at {scaler_path}, using new scaler")
                    X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.fit_transform(X)
            
            # 保存scaler（如果在训练模式下）
            if not model_dir:
                scaler_path = Path('models') / 'scaler.joblib'
                scaler_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.scaler, scaler_path)
                logging.info(f"Saved scaler to {scaler_path}")
            
            return X_scaled, y
            
        except Exception as e:
            logging.error(f"Error in prepare_data: {str(e)}")
            raise

    def get_feature_dim(self):
        """返回特征维度"""
        # 基础特征 + 周期性特征 + 交互特征
        return (len(self.base_feature_columns) +
                len(FEATURE_CONFIG['derived_features']['periodic']) +
                len(FEATURE_CONFIG['derived_features']['interactions'])) 