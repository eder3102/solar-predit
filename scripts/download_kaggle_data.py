"""
下载和处理NREL Solar Power数据集
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import requests
from io import StringIO
import zipfile
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def download_dataset():
    """下载数据集"""
    try:
        logger.info("开始下载数据集...")
        
        # 创建临时目录
        os.makedirs('data/raw', exist_ok=True)
        
        # NREL数据集URL（使用示例站点数据）
        base_url = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"
        params = {
            'api_key': 'o642mdsIXUBXKMzHWz0KKflVBvZjGeGkVLBw5joA',
            'wkt': 'POINT(-104.9903 39.7392)',  # Denver的坐标
            'names': '2019',
            'leap_day': 'false',
            'interval': '30',
            'utc': 'false',
            'email': 'eder3102.zhang@gmail.com',
            'attributes': 'air_temperature,dhi,dni,ghi,relative_humidity,solar_zenith_angle,surface_pressure,wind_speed,cloud_type'
        }
        
        # 下载数据
        logger.info("正在发送API请求...")
        logger.info(f"请求URL: {base_url}")
        logger.info(f"请求参数: {params}")
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            # 保存原始数据
            with open('data/raw/solar_data.csv', 'w') as f:
                f.write(response.text)
            logger.info("数据下载完成")
        else:
            # 打印详细的错误信息
            logger.error(f"API响应: {response.text}")
            raise Exception(f"下载失败，状态码: {response.status_code}")
            
    except Exception as e:
        logger.error(f"下载数据集时出错: {str(e)}")
        raise

def calculate_solar_position(df):
    """计算太阳位置相关特征"""
    # 将太阳天顶角转换为太阳高度角
    df['solar_elevation'] = 90 - df['solar_zenith_angle']
    
    # 计算太阳辐照强度
    df['total_irradiance'] = df['direct_irradiance'] + df['diffuse_irradiance']
    
    # 计算直射比例
    df['dni_ratio'] = df['direct_irradiance'] / (df['total_irradiance'] + 1e-6)
    
    return df

def calculate_weather_features(df):
    """计算天气相关特征"""
    # 计算露点温度
    df['dew_point'] = df.apply(lambda x: x['temperature'] - ((100 - x['relative_humidity']) / 5), axis=1)
    
    # 计算大气质量
    df['air_mass'] = 1 / np.cos(np.radians(df['solar_zenith_angle'].clip(0, 89)))
    
    # 计算清晰度指数
    df['clearness_index'] = df['total_irradiance'] / (1367 * np.cos(np.radians(df['solar_zenith_angle'])) + 1e-6)
    df['clearness_index'] = df['clearness_index'].clip(0, 1)
    
    return df

def calculate_power(df):
    """计算更准确的发电量预测"""
    # 基础参数
    efficiency_stc = 0.17  # 标准测试条件下的效率
    temp_coefficient = -0.004  # 温度系数 (%/°C)
    noct = 45  # 标称工作温度
    
    # 计算电池温度
    df['cell_temperature'] = df['temperature'] + (noct - 20) * df['total_irradiance'] / 800
    
    # 计算温度效率损失
    temp_efficiency = 1 + temp_coefficient * (df['cell_temperature'] - 25)
    
    # 计算总效率
    total_efficiency = efficiency_stc * temp_efficiency
    
    # 计算发电量
    panel_area = 100  # 假设面积为100平方米
    df['power'] = df['total_irradiance'] * panel_area * total_efficiency
    
    # 夜间发电量设为0
    df.loc[df['solar_elevation'] <= 0, 'power'] = 0
    
    return df

def process_data():
    """处理数据集"""
    try:
        logger.info("开始处理数据...")
        
        # 读取数据，跳过前两行元数据
        df = pd.read_csv('data/raw/solar_data.csv', skiprows=2)
        
        # 创建时间戳
        df['timestamp'] = pd.to_datetime(
            df[['Year', 'Month', 'Day', 'Hour', 'Minute']].assign(Second=0)
        )
        
        # 添加时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # 重命名列
        columns_mapping = {
            'Temperature': 'temperature',
            'DHI': 'diffuse_irradiance',
            'DNI': 'direct_irradiance',
            'GHI': 'total_irradiance',
            'Relative Humidity': 'relative_humidity',
            'Solar Zenith Angle': 'solar_zenith_angle',
            'Pressure': 'pressure',
            'Wind Speed': 'wind_speed',
            'Cloud Type': 'cloud_type'
        }
        df = df.rename(columns=columns_mapping)
        
        # 计算特征
        df = calculate_solar_position(df)
        df = calculate_weather_features(df)
        df = calculate_power(df)
        
        # 保留需要的列
        columns_to_keep = [
            'timestamp', 'power', 'temperature', 'total_irradiance',
            'wind_speed', 'direct_irradiance', 'diffuse_irradiance',
            'relative_humidity', 'pressure', 'cloud_type',
            'solar_elevation', 'dni_ratio', 'dew_point', 'air_mass',
            'clearness_index', 'cell_temperature', 'hour', 'day_of_week',
            'month', 'day_of_year'
        ]
        df = df[columns_to_keep]
        
        # 按时间排序
        df = df.sort_values('timestamp')
        
        # 分割数据集
        def split_data(df):
            # 使用最后14天的数据作为测试集
            test_days = 14
            val_days = 14
            
            end_date = df['timestamp'].max()
            test_start = end_date - pd.Timedelta(days=test_days)
            val_start = test_start - pd.Timedelta(days=val_days)
            
            test_data = df[df['timestamp'] >= test_start]
            val_data = df[(df['timestamp'] >= val_start) & 
                         (df['timestamp'] < test_start)]
            train_data = df[df['timestamp'] < val_start]
            
            return train_data, val_data, test_data
        
        # 分割并保存数据
        train_data, val_data, test_data = split_data(df)
        
        # 创建保存目录
        for subset in ['train', 'validation', 'test']:
            os.makedirs(f'data/{subset}', exist_ok=True)
        
        # 保存数据
        train_data.to_csv('data/train/data.csv', index=False)
        val_data.to_csv('data/validation/data.csv', index=False)
        test_data.to_csv('data/test/data.csv', index=False)
        
        logger.info(f"数据集大小:")
        logger.info(f"训练集: {len(train_data)} 行")
        logger.info(f"验证集: {len(val_data)} 行")
        logger.info(f"测试集: {len(test_data)} 行")
        
        # 显示数据统计信息
        logger.info("\n数据统计信息:")
        logger.info(df.describe().to_string())
        
        logger.info("数据处理完成")
        
    except Exception as e:
        logger.error(f"处理数据时出错: {str(e)}")
        raise

def main():
    """主函数"""
    try:
        # 创建原始数据目录
        os.makedirs('data/raw', exist_ok=True)
        
        # 下载并处理数据
        download_dataset()
        process_data()
        
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 