"""
监控工具类，用于收集和导出系统指标
"""
import psutil
import time
import threading
from prometheus_client import start_http_server, Gauge, Counter, Summary
import logging
from typing import Dict, Optional
import torch
from pathlib import Path

from src.config.config import MONITOR_CONFIG, SYSTEM_CONFIG

logger = logging.getLogger(__name__)

class SystemMonitor:
    """系统监控类"""
    
    def __init__(self, port: int = None):
        """
        初始化系统监控
        
        Args:
            port: Prometheus指标导出端口
        """
        self.port = port or MONITOR_CONFIG['metrics_port']
        
        # 初始化Prometheus指标
        self.memory_usage = Gauge('memory_usage_bytes', 
                                'Current memory usage in bytes')
        self.memory_percent = Gauge('memory_usage_percent', 
                                  'Current memory usage percentage')
        self.cpu_percent = Gauge('cpu_usage_percent', 
                               'Current CPU usage percentage')
        self.gpu_memory_usage = Gauge('gpu_memory_usage_bytes', 
                                    'Current GPU memory usage in bytes',
                                    ['device'])
        self.gpu_utilization = Gauge('gpu_utilization_percent',
                                   'Current GPU utilization percentage',
                                   ['device'])
                                   
        # 预测相关指标
        self.prediction_latency = Summary('prediction_latency_seconds',
                                        'Prediction latency in seconds')
        self.prediction_errors = Counter('prediction_errors_total',
                                       'Total number of prediction errors')
        self.prediction_requests = Counter('prediction_requests_total',
                                         'Total number of prediction requests')
                                         
        # 模型相关指标
        self.model_train_loss = Gauge('model_train_loss',
                                    'Current training loss')
        self.model_val_loss = Gauge('model_val_loss',
                                  'Current validation loss')
        self.model_metrics = Gauge('model_metrics',
                                 'Current model metrics',
                                 ['metric'])
                                 
        # 启动监控服务器
        start_http_server(self.port)
        logger.info(f"Started metrics server on port {self.port}")
        
        # 启动后台监控线程
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """监控循环，定期收集系统指标"""
        while not self.stop_monitoring:
            try:
                # 更新内存使用指标
                memory_info = psutil.Process().memory_info()
                self.memory_usage.set(memory_info.rss)
                self.memory_percent.set(psutil.Process().memory_percent())
                
                # 更新CPU使用指标
                self.cpu_percent.set(psutil.Process().cpu_percent())
                
                # 更新GPU指标（如果可用）
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_memory = torch.cuda.memory_allocated(i)
                        gpu_util = torch.cuda.utilization(i)
                        self.gpu_memory_usage.labels(device=f'gpu{i}').set(gpu_memory)
                        self.gpu_utilization.labels(device=f'gpu{i}').set(gpu_util)
                        
                # 检查资源使用是否超过阈值
                self._check_resource_usage()
                
                # 休眠指定时间
                time.sleep(MONITOR_CONFIG['update_interval'])
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(5)  # 发生错误时短暂休眠
                
    def _check_resource_usage(self):
        """检查资源使用是否超过阈值"""
        try:
            # 检查内存使用
            if psutil.Process().memory_info().rss > SYSTEM_CONFIG['max_memory_usage']:
                logger.warning("Memory usage exceeds threshold")
                
            # 检查CPU使用
            if psutil.Process().cpu_percent() > SYSTEM_CONFIG['max_cpu_usage'] * 100:
                logger.warning("CPU usage exceeds threshold")
                
        except Exception as e:
            logger.error(f"Error checking resource usage: {str(e)}")
            
    def record_prediction(self, latency: float, error: Optional[float] = None):
        """
        记录预测相关指标
        
        Args:
            latency: 预测延迟（秒）
            error: 预测误差（可选）
        """
        try:
            self.prediction_latency.observe(latency)
            self.prediction_requests.inc()
            
            if error is not None:
                self.model_metrics.labels(metric='prediction_error').set(error)
                
            # 检查延迟是否超过阈值
            if latency > SYSTEM_CONFIG['prediction_timeout']:
                logger.warning(f"Prediction latency ({latency:.3f}s) exceeds threshold")
                self.prediction_errors.inc()
                
        except Exception as e:
            logger.error(f"Error recording prediction metrics: {str(e)}")
            
    def record_training_metrics(self, train_loss: float, val_loss: float,
                              metrics: Dict[str, float]):
        """
        记录训练相关指标
        
        Args:
            train_loss: 训练损失
            val_loss: 验证损失
            metrics: 评估指标字典
        """
        try:
            self.model_train_loss.set(train_loss)
            self.model_val_loss.set(val_loss)
            
            for metric_name, value in metrics.items():
                self.model_metrics.labels(metric=metric_name).set(value)
                
        except Exception as e:
            logger.error(f"Error recording training metrics: {str(e)}")
            
    def stop(self):
        """停止监控"""
        self.stop_monitoring = True
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()
        logger.info("Stopped system monitoring") 