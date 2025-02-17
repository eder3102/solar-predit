"""
模型预测主程序
"""
import logging
import torch
import numpy as np
from pathlib import Path
import argparse
import time

from src.config.config import MODEL_CONFIG, DATA_CONFIG, SYSTEM_CONFIG
from src.utils.data_processor import DataProcessor
from src.models.ensemble.model import EnsembleModel
from src.utils.metrics import MetricsCalculator
from src.utils.monitor import SystemMonitor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模型预测程序')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='模型目录')
    parser.add_argument('--data_path', type=str, required=True,
                      help='预测数据路径')
    parser.add_argument('--output_path', type=str, required=True,
                      help='预测结果保存路径')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    return parser.parse_args()

def predict_batch(model: EnsembleModel, features: np.ndarray,
                 monitor: SystemMonitor) -> np.ndarray:
    """
    批量预测
    
    Args:
        model: 模型
        features: 特征数组
        monitor: 监控器
        
    Returns:
        预测结果数组
    """
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 预测
        predictions = model.predict(features)
        
        # 记录预测时间和指标
        prediction_time = time.time() - start_time
        monitor.record_prediction(prediction_time)
        
        # 检查是否超时
        if prediction_time > SYSTEM_CONFIG['prediction_timeout']:
            logger.warning(f"Prediction timeout: {prediction_time:.3f}s")
            
        return predictions
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise

def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_args()
        
        # 初始化数据处理器
        data_processor = DataProcessor()
        
        # 加载数据
        logger.info("Loading data...")
        df = data_processor.load_data(args.data_path)
        df = data_processor.prepare_data(df, is_training=False)
        features, targets = data_processor.split_features_target(df)
        
        # 初始化模型
        logger.info("Loading model...")
        model = EnsembleModel()
        model.load_models(args.model_dir)
        
        # 初始化监控器和指标计算器
        monitor = SystemMonitor()
        metrics_calculator = MetricsCalculator()
        
        # 批量预测
        logger.info("Starting prediction...")
        all_predictions = []
        batch_size = args.batch_size
        
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i + batch_size]
            batch_predictions = predict_batch(model, batch_features, monitor)
            all_predictions.append(batch_predictions)
            
        # 合并预测结果
        predictions = np.concatenate(all_predictions)
        
        # 计算评估指标
        if targets is not None:
            metrics = metrics_calculator.calculate_metrics(targets, predictions)
            logger.info(f"Prediction metrics: {metrics}")
            
        # 保存预测结果
        logger.info("Saving predictions...")
        df['predictions'] = predictions
        df.to_csv(args.output_path, index=True)
        
        logger.info("Prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

if __name__ == '__main__':
    main() 