"""
可视化预测结果
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from pathlib import Path
import logging
from datetime import datetime, timedelta
import os
import json

from src.models.trainer import ModelTrainer
from src.utils.data_processor import DataProcessor
from src.utils.dataset import DatasetFactory
from src.models.ensemble.model import EnsembleModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """加载测试数据"""
    data_processor = DataProcessor()
    test_df = data_processor.load_data('data/test')
    test_df = data_processor.prepare_data(test_df, is_training=False)
    test_features, test_targets = data_processor.split_features_target(test_df)
    
    return test_df, test_features, test_targets

def get_predictions(df: pd.DataFrame, model: torch.nn.Module):
    """获取模型预测结果"""
    try:
        # 使用与训练相同的数据处理器
        data_processor = DataProcessor()
        
        # 处理数据
        df = data_processor.prepare_data(df, is_training=False, model_dir=Path('models'))
        features, targets = data_processor.split_features_target(df)
        
        # 转换为tensor
        features_tensor = torch.FloatTensor(features)
        
        # 获取预测结果
        with torch.no_grad():
            predictions = model.predict(features_tensor)
        
        return predictions, targets
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def create_visualizations(test_df, predictions, actuals, power_info):
    """创建可视化图表"""
    # 创建子图
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('发电量预测对比', '预测误差分布', '日发电量对比'),
        vertical_spacing=0.15,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # 添加时间序列对比图
    fig.add_trace(
        go.Scatter(
            x=test_df.index,
            y=actuals.flatten(),
            name='实际发电量',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=test_df.index,
            y=predictions.flatten(),
            name='预测发电量',
            line=dict(color='#ff7f0e', width=2, dash='dot')
        ),
        row=1, col=1
    )
    
    # 添加系统参考功率线
    fig.add_hline(
        y=power_info['system_power'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"系统参考功率: {power_info['system_power']/1000:.1f}kW",
        row=1, col=1
    )
    
    # 添加误差分布图
    errors = predictions.flatten() - actuals.flatten()
    fig.add_trace(
        go.Histogram(
            x=errors,
            name='预测误差分布',
            nbinsx=50,
            marker_color='#2ca02c'
        ),
        row=2, col=1
    )
    
    # 计算每日发电量
    daily_actual = pd.Series(actuals.flatten(), index=test_df.index).resample('D').sum()
    daily_pred = pd.Series(predictions.flatten(), index=test_df.index).resample('D').sum()
    
    # 添加日发电量对比图
    fig.add_trace(
        go.Bar(
            x=daily_actual.index,
            y=daily_actual.values / 1000,  # 转换为kWh
            name='实际日发电量',
            marker_color='#1f77b4',
            opacity=0.7
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=daily_pred.index,
            y=daily_pred.values / 1000,  # 转换为kWh
            name='预测日发电量',
            marker_color='#ff7f0e',
            opacity=0.7
        ),
        row=3, col=1
    )
    
    # 更新布局
    fig.update_layout(
        title='太阳能发电量预测结果分析',
        height=1200,
        showlegend=True,
        template='plotly_white'
    )
    
    # 更新坐标轴
    fig.update_xaxes(title_text='时间', row=1, col=1, gridcolor='#f0f0f0')
    fig.update_xaxes(title_text='预测误差 (W)', row=2, col=1, gridcolor='#f0f0f0')
    fig.update_xaxes(title_text='日期', row=3, col=1, gridcolor='#f0f0f0')
    
    fig.update_yaxes(title_text='发电量 (W)', row=1, col=1, gridcolor='#f0f0f0')
    fig.update_yaxes(title_text='频数', row=2, col=1, gridcolor='#f0f0f0')
    fig.update_yaxes(title_text='日发电量 (kWh)', row=3, col=1, gridcolor='#f0f0f0')
    
    # 保存图表
    os.makedirs('results', exist_ok=True)
    fig.write_html('results/predictions.html')
    
    # 计算并显示评估指标
    metrics = {}
    
    # 总体指标
    metrics['mae'] = np.mean(np.abs(errors))
    metrics['rmse'] = np.sqrt(np.mean(errors ** 2))
    metrics['r2'] = 1 - np.sum(errors ** 2) / np.sum((actuals.flatten() - actuals.mean()) ** 2)
    
    # 白天指标（发电量>10%额定功率）
    day_mask = actuals.flatten() > 0.1 * power_info['system_power']
    if day_mask.any():
        day_errors = errors[day_mask]
        day_actuals = actuals.flatten()[day_mask]
        metrics['day_mae'] = np.mean(np.abs(day_errors))
        metrics['day_rmse'] = np.sqrt(np.mean(day_errors ** 2))
        metrics['day_mape'] = np.mean(np.abs(day_errors / day_actuals)) * 100
    
    # 日发电量指标
    daily_errors = daily_pred - daily_actual
    metrics['daily_mae'] = np.mean(np.abs(daily_errors))
    metrics['daily_rmse'] = np.sqrt(np.mean(daily_errors ** 2))
    metrics['daily_mape'] = np.mean(np.abs(daily_errors / daily_actual)) * 100
    
    print("\n评估指标:")
    print(f"总体MAE: {metrics['mae']:.2f} W")
    print(f"总体RMSE: {metrics['rmse']:.2f} W")
    print(f"总体R²: {metrics['r2']:.4f}")
    if 'day_mae' in metrics:
        print(f"\n白天MAE: {metrics['day_mae']:.2f} W")
        print(f"白天RMSE: {metrics['day_rmse']:.2f} W")
        print(f"白天MAPE: {metrics['day_mape']:.2f}%")
    print(f"\n日发电量MAE: {metrics['daily_mae']/1000:.2f} kWh")
    print(f"日发电量RMSE: {metrics['daily_rmse']/1000:.2f} kWh")
    print(f"日发电量MAPE: {metrics['daily_mape']:.2f}%")
    
    print("\n可视化结果已保存到:")
    print("- results/predictions.html")

def main():
    """主函数"""
    try:
        # 加载数据
        logger.info("加载测试数据...")
        test_df = pd.read_csv('data/test/data.csv')
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        test_df.set_index('timestamp', inplace=True)
        
        # 加载power_info
        logger.info("加载系统参数...")
        with open('data/power_info.json', 'r') as f:
            power_info = json.load(f)
        
        # 加载模型
        logger.info("加载模型...")
        model = EnsembleModel()
        model.load_models('models')
        
        # 获取预测结果
        logger.info("生成预测结果...")
        predictions, actuals = get_predictions(test_df, model)
        
        # 创建可视化
        logger.info("创建可视化图表...")
        create_visualizations(test_df, predictions, actuals, power_info)
        
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 