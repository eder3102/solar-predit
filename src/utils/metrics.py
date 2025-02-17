"""
评估指标类，用于计算模型性能指标
"""
import numpy as np
from typing import Dict, Union, List
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """评估指标计算类"""
    
    @staticmethod
    def calculate_metrics(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            metrics: 需要计算的指标列表
            
        Returns:
            包含各项指标的字典
        """
        try:
            # 转换为numpy数组
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()
                
            # 确保数组形状一致
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)
            
            # 默认指标列表
            if metrics is None:
                metrics = ['mae', 'rmse', 'r2', 'mape']
                
            results = {}
            
            # 计算MAE
            if 'mae' in metrics:
                results['mae'] = float(mean_absolute_error(y_true, y_pred))
                
            # 计算RMSE
            if 'rmse' in metrics:
                results['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                
            # 计算R2
            if 'r2' in metrics:
                results['r2'] = float(r2_score(y_true, y_pred))
                
            # 计算MAPE
            if 'mape' in metrics:
                mask = y_true != 0
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                results['mape'] = float(mape)
                
            return results
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
            
    @staticmethod
    def calculate_running_metrics(
        running_metrics: Dict[str, float],
        batch_metrics: Dict[str, float],
        batch_size: int,
        total_samples: int
    ) -> Dict[str, float]:
        """
        计算运行中的评估指标
        
        Args:
            running_metrics: 当前累积的评估指标
            batch_metrics: 当前批次的评估指标
            batch_size: 批次大小
            total_samples: 总样本数
            
        Returns:
            更新后的运行评估指标
        """
        try:
            # 如果running_metrics为空，初始化
            if not running_metrics:
                running_metrics = {k: 0.0 for k in batch_metrics.keys()}
                
            # 更新运行评估指标
            weight = batch_size / total_samples
            for metric in batch_metrics:
                running_metrics[metric] += batch_metrics[metric] * weight
                
            return running_metrics
            
        except Exception as e:
            logger.error(f"Error calculating running metrics: {str(e)}")
            raise
            
    @staticmethod
    def format_metrics(metrics: Dict[str, float]) -> str:
        """
        格式化评估指标输出
        
        Args:
            metrics: 评估指标字典
            
        Returns:
            格式化的评估指标字符串
        """
        try:
            formatted_metrics = []
            for metric, value in metrics.items():
                if metric == 'mape':
                    formatted_metrics.append(f"{metric.upper()}: {value:.2f}%")
                else:
                    formatted_metrics.append(f"{metric.upper()}: {value:.4f}")
                    
            return ", ".join(formatted_metrics)
            
        except Exception as e:
            logger.error(f"Error formatting metrics: {str(e)}")
            raise
            
    @staticmethod
    def check_metrics_threshold(
        metrics: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> bool:
        """
        检查评估指标是否满足阈值要求
        
        Args:
            metrics: 评估指标字典
            thresholds: 阈值字典
            
        Returns:
            是否满足阈值要求
        """
        try:
            for metric, threshold in thresholds.items():
                if metric in metrics:
                    if metric == 'r2':
                        if metrics[metric] < threshold:
                            logger.warning(f"{metric.upper()} below threshold: "
                                         f"{metrics[metric]:.4f} < {threshold}")
                            return False
                    else:
                        if metrics[metric] > threshold:
                            logger.warning(f"{metric.upper()} above threshold: "
                                         f"{metrics[metric]:.4f} > {threshold}")
                            return False
                            
            logger.info("All metrics within thresholds")
            return True
            
        except Exception as e:
            logger.error(f"Error checking metrics thresholds: {str(e)}")
            raise 