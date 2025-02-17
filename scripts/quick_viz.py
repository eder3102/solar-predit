"""
快速可视化预测结果
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import logging
from pathlib import Path
import os

from src.utils.data_processor import DataProcessor
from src.models.filternet.model import FilterNet

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model(model_path: str = 'models/best_model.pth'):
    """加载训练好的模型"""
    # 初始化模型配置
    model_config = {
        'input_dim': 23,
        'hidden_dim': 128,
        'output_dim': 1,
        'num_layers': 4,
        'dropout': 0.3
    }
    
    # 初始化模型
    model = FilterNet(config=model_config)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

def get_predictions(df: pd.DataFrame, model: torch.nn.Module):
    """获取模型预测结果"""
    # 特征工程
    df = df.copy()
    
    print("\n数据检查:")
    print("1. 原始数据:")
    print(df.head())
    print("\n2. 数据统计:")
    print(df.describe())
    
    # 周期性特征编码
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # 交互特征
    df['irradiance_temp'] = df['total_irradiance'] * df['temperature']
    df['clearness_elevation'] = df['clearness_index'] * df['solar_elevation']
    
    # 特征列表
    feature_columns = [
        'temperature', 'relative_humidity', 'wind_speed', 'pressure',
        'cloud_type', 'solar_elevation', 'direct_irradiance', 'diffuse_irradiance',
        'dni_ratio', 'dew_point', 'air_mass', 'clearness_index', 'cell_temperature',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
        'irradiance_temp', 'clearness_elevation'
    ]
    
    print("\n3. 特征列检查:")
    for col in feature_columns:
        if col not in df.columns:
            print(f"缺失特征: {col}")
    
    # 准备特征数据
    data_processor = DataProcessor()
    features = data_processor.scale_features(df[feature_columns], is_training=False)
    features = features.values if isinstance(features, pd.DataFrame) else features
    
    print("\n4. 特征数据形状:", features.shape)
    print("5. 特征数据统计:")
    print(pd.DataFrame(features, columns=feature_columns).describe())
    
    # 转换为tensor
    features_tensor = torch.FloatTensor(features)
    
    # 获取预测结果
    with torch.no_grad():
        predictions = model(features_tensor)
        predictions = predictions.clip(min=-20, max=20)  # 限制预测值范围
        predictions = torch.exp(predictions) - 1  # 使用exp而不是expm1
        predictions = predictions.numpy()
        
        # 确保夜间发电量为0
        night_mask = df['solar_elevation'] <= 0
        predictions[night_mask] = 0
        
        # 限制最大发电量
        max_power = 20000  # 20kW
        predictions = np.clip(predictions, 0, max_power)
    
    print("\n6. 预测结果统计:")
    print(pd.Series(predictions.flatten()).describe())
    
    return predictions

def create_visualizations(test_data_path: str = 'data/test/data.csv',
                        model_path: str = 'models/best_model.pth'):
    """创建预测结果可视化"""
    try:
        # 读取测试数据
        df = pd.read_csv(test_data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 加载模型并生成预测
        model = load_model(model_path)
        df['predictions'] = get_predictions(df, model)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('发电量预测对比', '预测误差分布'),
            vertical_spacing=0.15
        )
        
        # 添加时间序列对比图
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['power'],
                name='实际发电量',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['predictions'],
                name='预测发电量',
                line=dict(color='#ff7f0e', width=2, dash='dot')
            ),
            row=1, col=1
        )
        
        # 添加误差分布图
        errors = df['predictions'] - df['power']
        fig.add_trace(
            go.Histogram(
                x=errors,
                name='预测误差分布',
                nbinsx=50,
                marker_color='#2ca02c'
            ),
            row=2, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title='太阳能发电量预测结果分析',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # 更新坐标轴
        fig.update_xaxes(title_text='时间', row=1, col=1, gridcolor='#f0f0f0')
        fig.update_xaxes(title_text='预测误差 (W)', row=2, col=1, gridcolor='#f0f0f0')
        fig.update_yaxes(title_text='发电量 (W)', row=1, col=1, gridcolor='#f0f0f0')
        fig.update_yaxes(title_text='频数', row=2, col=1, gridcolor='#f0f0f0')
        
        # 创建日发电量对比图
        daily_actual = df.groupby(df['timestamp'].dt.date)['power'].sum()
        daily_pred = df.groupby(df['timestamp'].dt.date)['predictions'].sum()
        
        fig2 = go.Figure()
        
        fig2.add_trace(
            go.Bar(
                x=daily_actual.index,
                y=daily_actual.values,
                name='实际日发电量',
                marker_color='#1f77b4',
                opacity=0.7
            )
        )
        
        fig2.add_trace(
            go.Bar(
                x=daily_pred.index,
                y=daily_pred.values,
                name='预测日发电量',
                marker_color='#ff7f0e',
                opacity=0.7
            )
        )
        
        fig2.update_layout(
            title='日发电量对比',
            xaxis_title='日期',
            yaxis_title='日发电量 (Wh)',
            barmode='group',
            template='plotly_white'
        )
        
        # 保存图表
        os.makedirs('results', exist_ok=True)
        fig.write_html('results/predictions.html')
        fig2.write_html('results/daily_predictions.html')
        
        # 计算评估指标
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        r2 = 1 - np.sum(errors ** 2) / np.sum((df['power'] - df['power'].mean()) ** 2)
        mape = np.mean(np.abs(errors / (df['power'] + 1))) * 100
        
        print("\n评估指标:")
        print(f"MAE: {mae:.2f} W")
        print(f"RMSE: {rmse:.2f} W")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        print("\n可视化结果已保存到:")
        print("- results/predictions.html")
        print("- results/daily_predictions.html")
        
    except Exception as e:
        logging.error(f"创建可视化时出错: {str(e)}")
        raise

if __name__ == '__main__':
    create_visualizations() 