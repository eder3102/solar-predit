"""
生成示例数据用于开发环境测试
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

def generate_weather_data(start_date, num_days):
    """生成模拟的天气数据"""
    hours = num_days * 24 * 2  # 30分钟间隔
    timestamps = [start_date + timedelta(minutes=30*i) for i in range(hours)]
    
    # 生成基础天气特征
    temperature = 25 + np.sin(np.linspace(0, 2*np.pi*num_days, hours)) * 5 + np.random.normal(0, 1, hours)
    humidity = 60 + np.sin(np.linspace(0, 2*np.pi*num_days, hours)) * 10 + np.random.normal(0, 2, hours)
    wind_speed = 2 + np.sin(np.linspace(0, 2*np.pi*num_days, hours)) * 1 + np.abs(np.random.normal(0, 0.5, hours))
    pressure = 1013 + np.sin(np.linspace(0, 2*np.pi*num_days, hours)) * 2 + np.random.normal(0, 0.5, hours)
    cloud_type = np.random.randint(0, 10, hours)  # 0-9表示不同云类型
    
    # 生成太阳位置特征
    hour_angles = np.array([ts.hour + ts.minute/60 for ts in timestamps])
    day_angles = np.array([ts.timetuple().tm_yday for ts in timestamps]) * 2 * np.pi / 365
    
    # 计算太阳高度角
    solar_elevation = np.sin(day_angles) * np.sin(np.pi/4) + \
                     np.cos(day_angles) * np.cos(np.pi/4) * \
                     np.cos(hour_angles * np.pi / 12 - np.pi)
    solar_elevation = np.rad2deg(np.arcsin(solar_elevation))
    
    # 生成辐照度
    direct_irradiance = np.zeros(hours)
    diffuse_irradiance = np.zeros(hours)
    
    for i, elevation in enumerate(solar_elevation):
        if elevation > 0:
            # 直接辐照
            direct_irradiance[i] = 1000 * np.sin(np.deg2rad(elevation)) * \
                                 (1 - cloud_type[i]/20) + np.random.normal(0, 20)
            # 散射辐照
            diffuse_irradiance[i] = 100 + cloud_type[i] * 20 + np.random.normal(0, 10)
            
    # 确保非负值
    direct_irradiance = np.maximum(0, direct_irradiance)
    diffuse_irradiance = np.maximum(0, diffuse_irradiance)
    
    # 计算总辐照度
    total_irradiance = direct_irradiance + diffuse_irradiance
    
    # 计算其他特征
    dni_ratio = np.where(total_irradiance > 0, direct_irradiance / total_irradiance, 0)
    dew_point = temperature - ((100 - humidity) / 5)
    air_mass = 1 / np.cos(np.deg2rad(90 - solar_elevation.clip(0, 89)))
    clearness_index = total_irradiance / (1367 * np.sin(np.deg2rad(solar_elevation.clip(0, 90))) + 1e-6)
    clearness_index = clearness_index.clip(0, 1)
    
    # 计算电池温度
    cell_temperature = temperature + (45 - 20) * total_irradiance / 800
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'relative_humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure,
        'cloud_type': cloud_type,
        'solar_elevation': solar_elevation,
        'direct_irradiance': direct_irradiance,
        'diffuse_irradiance': diffuse_irradiance,
        'total_irradiance': total_irradiance,
        'dni_ratio': dni_ratio,
        'dew_point': dew_point,
        'air_mass': air_mass,
        'clearness_index': clearness_index,
        'cell_temperature': cell_temperature,
        'irradiance': total_irradiance  # 总辐照度作为irradiance
    })

def generate_power_data(weather_data):
    """基于天气数据生成发电量数据"""
    # 基础参数
    efficiency_stc = 0.17  # 标准测试条件下的效率
    temp_coefficient = -0.004  # 温度系数 (%/°C)
    panel_area = 100  # 面积为100平方米
    
    # 计算温度效率损失
    temp_efficiency = 1 + temp_coefficient * (weather_data['cell_temperature'] - 25)
    
    # 计算总效率
    total_efficiency = efficiency_stc * temp_efficiency
    
    # 计算发电量
    power = weather_data['irradiance'] * panel_area * total_efficiency
    
    # 添加随机波动
    power = power * (1 + np.random.normal(0, 0.05, len(power)))
    
    # 确保发电量非负且夜间为0
    power = np.maximum(0, power)
    power[weather_data['solar_elevation'] <= 0] = 0
    
    return power

def main():
    # 设置随机种子
    np.random.seed(42)
    
    # 设置日期范围
    start_date = datetime(2024, 1, 1)
    train_days = 30
    val_days = 7
    test_days = 7
    
    # 生成数据
    train_weather = generate_weather_data(start_date, train_days)
    val_weather = generate_weather_data(start_date + timedelta(days=train_days), val_days)
    test_weather = generate_weather_data(start_date + timedelta(days=train_days+val_days), test_days)
    
    # 生成发电量
    train_weather['power'] = generate_power_data(train_weather)
    val_weather['power'] = generate_power_data(val_weather)
    test_weather['power'] = generate_power_data(test_weather)
    
    # 添加时间特征
    for df in [train_weather, val_weather, test_weather]:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # 创建数据目录
    data_dir = Path(__file__).parent.parent / 'data'
    for subset in ['train', 'validation', 'test']:
        (data_dir / subset).mkdir(parents=True, exist_ok=True)
    
    # 保存数据
    train_weather.to_csv(data_dir / 'train/data.csv', index=False)
    val_weather.to_csv(data_dir / 'validation/data.csv', index=False)
    test_weather.to_csv(data_dir / 'test/data.csv', index=False)
    
    print(f"生成的数据集大小:")
    print(f"训练集: {len(train_weather)} 行")
    print(f"验证集: {len(val_weather)} 行")
    print(f"测试集: {len(test_weather)} 行")
    
    # 显示数据样例
    print("\n训练集样例:")
    print(train_weather.head())
    
    # 显示基本统计信息
    print("\n数据统计信息:")
    print(train_weather.describe())

if __name__ == '__main__':
    main() 