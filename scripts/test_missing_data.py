"""
测试模型在数据缺失情况下的性能
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import pandas as pd
import numpy as np
import logging
import random
import shutil

from src.data.data_processor import DataProcessor
from src.models.ensemble.model import EnsembleModel
from scripts.visualize_predictions import create_visualizations

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_missing_data(n_features: int = 3):
    """创建带有缺失值的测试数据"""
    try:
        # 创建测试数据目录
        missing_data_dir = Path('data/test_missing')
        missing_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制原始测试数据
        original_data = pd.read_csv('data/test/data.csv')
        logger.info(f"原始数据形状: {original_data.shape}")
        logger.info("原始数据列: %s", original_data.columns.tolist())
        
        # 获取可以设置为缺失值的特征列
        feature_columns = [
            'temperature', 'total_irradiance', 'wind_speed',
            'direct_irradiance', 'diffuse_irradiance',
            'relative_humidity', 'pressure', 'cloud_type',
            'solar_elevation', 'dni_ratio', 'dew_point',
            'air_mass', 'clearness_index', 'cell_temperature'
        ]
        
        # 随机选择n个特征设置为缺失值
        missing_features = random.sample(feature_columns, n_features)
        logger.info(f"将以下特征设置为缺失值: {missing_features}")
        
        # 创建缺失值
        missing_data = original_data.copy()
        for feature in missing_features:
            missing_data[feature] = np.nan
            
        # 检查缺失值
        logger.info("缺失值统计:")
        for col in missing_data.columns:
            missing_count = missing_data[col].isnull().sum()
            if missing_count > 0:
                logger.info(f"- {col}: {missing_count} 个缺失值")
                
        # 保存带有缺失值的数据
        missing_data.to_csv(missing_data_dir / 'data.csv', index=False)
        
        # 复制power_info.json
        shutil.copy2('data/power_info.json', missing_data_dir / 'power_info.json')
        
        return missing_features
        
    except Exception as e:
        logger.error(f"创建缺失数据时出错: {str(e)}")
        raise

def test_missing_data():
    """测试模型在数据缺失情况下的性能"""
    try:
        # 创建缺失数据
        missing_features = create_missing_data(n_features=3)
        
        # 加载数据
        data_processor = DataProcessor()
        test_df = pd.read_csv('data/test_missing/data.csv')
        logger.info("加载的测试数据形状: %s", test_df.shape)
        logger.info("测试数据列: %s", test_df.columns.tolist())
        
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        test_df.set_index('timestamp', inplace=True)
        
        # 检查数据
        logger.info("\n数据统计:")
        logger.info(test_df.describe().to_string())
        
        # 处理数据
        logger.info("\n开始处理数据...")
        test_features, test_targets = data_processor.prepare_data(test_df, model_dir='models')
        logger.info("处理后的特征形状: %s", test_features.shape)
        logger.info("处理后的目标形状: %s", test_targets.shape)
        
        # 检查处理后的数据
        logger.info("\n处理后的特征统计:")
        feature_df = pd.DataFrame(test_features)
        logger.info(feature_df.describe().to_string())
        
        # 加载模型
        logger.info("\n加载模型...")
        model = EnsembleModel()
        model.load_models('models')
        
        # 生成预测
        logger.info("\n生成预测...")
        predictions = model.predict(test_features)
        logger.info("预测形状: %s", predictions.shape)
        logger.info("预测统计: \n%s", pd.Series(predictions.flatten()).describe().to_string())
        
        # 加载power_info
        import json
        with open('data/power_info.json', 'r') as f:
            power_info = json.load(f)
            
        # 转换回原始功率
        predictions = predictions * power_info['system_power']
        actuals = test_targets.values * power_info['system_power']
        
        logger.info("\n转换后的预测统计:")
        logger.info("预测值: \n%s", pd.Series(predictions.flatten()).describe().to_string())
        logger.info("实际值: \n%s", pd.Series(actuals.flatten()).describe().to_string())
        
        # 创建可视化
        create_visualizations(test_df, predictions, actuals, power_info)
        
        # 重命名可视化文件
        os.rename('results/predictions.html', 'results/predictions_missing_data.html')
        
        # 计算评估指标
        errors = predictions.flatten() - actuals.flatten()
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
            
        # 计算每日发电量
        daily_actual = pd.Series(actuals.flatten(), index=test_df.index).resample('D').sum()
        daily_pred = pd.Series(predictions.flatten(), index=test_df.index).resample('D').sum()
        
        # 日发电量指标
        daily_errors = daily_pred - daily_actual
        metrics['daily_mae'] = np.mean(np.abs(daily_errors))
        metrics['daily_rmse'] = np.sqrt(np.mean(daily_errors ** 2))
        metrics['daily_mape'] = np.mean(np.abs(daily_errors / daily_actual)) * 100
        
        # 打印评估指标
        print(f"\n缺失特征: {missing_features}")
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
        print("- results/predictions_missing_data.html")
        
    except Exception as e:
        logger.error(f"测试缺失数据时出错: {str(e)}")
        raise

if __name__ == '__main__':
    test_missing_data() 