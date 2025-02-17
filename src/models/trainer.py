"""
模型训练器实现
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import time
from torch.optim import Adam
from torch.nn import MSELoss

from src.config.config import TRAIN_CONFIG, TEST_CONFIG
from src.models.ensemble.model import EnsembleModel
from src.utils.metrics import MetricsCalculator
from src.utils.monitor import SystemMonitor

logger = logging.getLogger(__name__)

class ModelTrainer:
    """模型训练器类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化训练器
        
        Args:
            config: 训练配置字典
        """
        # 使用默认配置或传入配置
        self.config = config or TRAIN_CONFIG
        
        # 初始化模型
        self.model = EnsembleModel()
        
        # 优化器
        self.optimizers = {
            'filternet': Adam(self.model.models['filternet'].parameters(),
                            lr=self.config['learning_rate']),
            'bilstm': Adam(self.model.models['bilstm'].parameters(),
                          lr=self.config['learning_rate'])
        }
        
        # 损失函数
        self.criterion = MSELoss()
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator()
        
        # 系统监控器
        self.monitor = SystemMonitor()
        
        logger.info(f"Initialized ModelTrainer with config: {self.config}")
        
    def train_xgboost(self, train_loader: torch.utils.data.DataLoader,
                      val_loader: Optional[torch.utils.data.DataLoader] = None):
        """
        训练XGBoost模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        logger.info("Training XGBoost model...")
        
        # 收集所有数据
        train_features = []
        train_targets = []
        for batch_features, batch_targets in train_loader:
            # 如果是序列数据，只使用最后一个时间步
            if len(batch_features.shape) == 3:
                batch_features = batch_features[:, -1, :]
            train_features.append(batch_features.numpy())
            train_targets.append(batch_targets.numpy())
            
        train_features = np.concatenate(train_features)
        train_targets = np.concatenate(train_targets)
        
        # 准备验证数据
        eval_set = None
        if val_loader is not None:
            val_features = []
            val_targets = []
            for batch_features, batch_targets in val_loader:
                if len(batch_features.shape) == 3:
                    batch_features = batch_features[:, -1, :]
                val_features.append(batch_features.numpy())
                val_targets.append(batch_targets.numpy())
                
            val_features = np.concatenate(val_features)
            val_targets = np.concatenate(val_targets)
            eval_set = [(val_features, val_targets)]
            
        # 训练XGBoost
        self.model.models['xgboost'].fit(train_features, train_targets, eval_set=eval_set)
        logger.info("XGBoost training completed")
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader,
                   val_loader: Optional[torch.utils.data.DataLoader] = None
                   ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练和验证指标
        """
        try:
            # 训练模式
            self.model.models['filternet'].train()
            self.model.models['bilstm'].train()
            
            total_loss = 0
            batch_count = 0
            
            # 训练循环
            for batch_features, batch_targets in train_loader:
                # 确保targets维度匹配 [batch_size] -> [batch_size, 1]
                batch_targets = batch_targets.unsqueeze(1).float()
                
                # 获取各个模型的预测
                if len(batch_features.shape) == 3:  # 序列数据
                    xgb_features = batch_features[:, -1, :].numpy()  # 只使用最后一个时间步
                    filternet_pred = self.model.models['filternet'](batch_features)
                    bilstm_pred = self.model.models['bilstm'](batch_features)
                    xgb_pred = torch.tensor(self.model.models['xgboost'].predict(xgb_features), dtype=torch.float32).unsqueeze(1)
                else:  # 单步数据
                    xgb_pred = torch.tensor(self.model.models['xgboost'].predict(batch_features.numpy()), dtype=torch.float32).unsqueeze(1)
                    filternet_pred = self.model.models['filternet'](batch_features.unsqueeze(1))
                    bilstm_pred = self.model.models['bilstm'](batch_features.unsqueeze(1))
                
                # 确保所有预测都需要梯度
                xgb_pred = xgb_pred.detach()  # XGBoost预测不需要梯度
                filternet_pred.requires_grad_(True)
                bilstm_pred.requires_grad_(True)
                
                # 计算集成预测
                predictions = (
                    self.model.weights['filternet'] * filternet_pred +
                    self.model.weights['bilstm'] * bilstm_pred +
                    self.model.weights['xgboost'] * xgb_pred
                )
                
                # 计算损失
                loss = self.criterion(predictions, batch_targets)
                
                # 反向传播
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()
                loss.backward()
                for optimizer in self.optimizers.values():
                    optimizer.step()
                    
                total_loss += loss.item()
                batch_count += 1
            
            # 计算平均损失
            avg_train_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            
            # 验证
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                
            # 更新模型权重
            if val_loader is not None:
                val_predictions = {
                    'filternet': [],
                    'bilstm': [],
                    'xgboost': []
                }
                val_targets_list = []
                
                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        # 获取各个模型的预测
                        if len(batch_features.shape) == 3:
                            xgb_features = batch_features[:, -1, :].numpy()
                            filternet_pred = self.model.models['filternet'](batch_features)
                            bilstm_pred = self.model.models['bilstm'](batch_features)
                            xgb_pred = self.model.models['xgboost'].predict(xgb_features).reshape(-1, 1)
                        else:
                            xgb_pred = self.model.models['xgboost'].predict(batch_features.numpy()).reshape(-1, 1)
                            filternet_pred = self.model.models['filternet'](batch_features.unsqueeze(1))
                            bilstm_pred = self.model.models['bilstm'](batch_features.unsqueeze(1))
                            
                        val_predictions['filternet'].extend(filternet_pred.numpy())
                        val_predictions['bilstm'].extend(bilstm_pred.numpy())
                        val_predictions['xgboost'].extend(xgb_pred)
                        val_targets_list.extend(batch_targets.numpy())
                
                # 更新权重
                self.model.update_weights(val_predictions, np.array(val_targets_list))
            
            # 返回训练和验证指标
            metrics = {
                'train_loss': avg_train_loss,
                'val_loss': val_metrics['loss'] if val_metrics else None
            }
            
            logger.info(f"Epoch metrics - Train Loss: {avg_train_loss:.6f}" + 
                       (f", Val Loss: {metrics['val_loss']:.6f}" if val_metrics else ""))
            return metrics
            
        except Exception as e:
            logger.error(f"Error in training epoch: {str(e)}")
            raise
            
    def evaluate(self, data_loader: torch.utils.data.DataLoader
                ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            评估指标
        """
        try:
            # 评估模式
            self.model.models['filternet'].eval()
            self.model.models['bilstm'].eval()
            
            total_loss = 0
            batch_count = 0
            all_predictions = []
            all_targets = []
            
            # 评估循环
            with torch.no_grad():
                for batch_features, batch_targets in data_loader:
                    # 记录开始时间
                    start_time = time.time()
                    
                    # 确保targets维度匹配
                    batch_targets = batch_targets.unsqueeze(1).float()
                    
                    # 获取各个模型的预测
                    if len(batch_features.shape) == 3:  # 序列数据
                        xgb_features = batch_features[:, -1, :].numpy()
                        filternet_pred = self.model.models['filternet'](batch_features)
                        bilstm_pred = self.model.models['bilstm'](batch_features)
                        xgb_pred = torch.tensor(self.model.models['xgboost'].predict(xgb_features), dtype=torch.float32).unsqueeze(1)
                    else:  # 单步数据
                        xgb_pred = torch.tensor(self.model.models['xgboost'].predict(batch_features.numpy()), dtype=torch.float32).unsqueeze(1)
                        filternet_pred = self.model.models['filternet'](batch_features.unsqueeze(1))
                        bilstm_pred = self.model.models['bilstm'](batch_features.unsqueeze(1))
                    
                    # 确保所有预测都需要梯度
                    xgb_pred = xgb_pred.detach()  # XGBoost预测不需要梯度
                    filternet_pred.requires_grad_(True)
                    bilstm_pred.requires_grad_(True)
                    
                    # 计算集成预测
                    predictions = (
                        self.model.weights['filternet'] * filternet_pred +
                        self.model.weights['bilstm'] * bilstm_pred +
                        self.model.weights['xgboost'] * xgb_pred
                    )
                    
                    # 计算损失
                    loss = self.criterion(predictions, batch_targets)
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # 收集预测结果
                    all_predictions.append(predictions.numpy())
                    all_targets.append(batch_targets.numpy())
                    
                    # 记录预测时间
                    batch_time = time.time() - start_time
                    self.monitor.record_prediction(batch_time)
                    
            # 合并所有预测结果
            all_predictions = np.concatenate(all_predictions)
            all_targets = np.concatenate(all_targets)
            
            # 计算评估指标
            metrics = self.metrics_calculator.calculate_metrics(
                all_targets,
                all_predictions,
                metrics=TEST_CONFIG['metrics']
            )
            
            # 检查指标是否满足阈值要求
            meets_threshold = self.metrics_calculator.check_metrics_threshold(
                metrics,
                TEST_CONFIG['threshold']
            )
            
            # 返回评估结果
            results = {
                'loss': total_loss / batch_count,
                'metrics': metrics,
                'meets_threshold': meets_threshold
            }
            
            logger.info(f"Evaluation results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise
            
    def train(self, train_loader: torch.utils.data.DataLoader,
              val_loader: Optional[torch.utils.data.DataLoader] = None,
              save_dir: Optional[Union[str, Path]] = None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            save_dir: 模型保存目录
        """
        try:
            # 收集所有数据用于XGBoost训练
            train_features = []
            train_targets = []
            for batch_features, batch_targets in train_loader:
                # 如果是序列数据，只使用最后一个时间步
                if len(batch_features.shape) == 3:
                    batch_features = batch_features[:, -1, :]
                train_features.append(batch_features.numpy())
                train_targets.append(batch_targets.numpy())
            
            train_features = np.concatenate(train_features)
            train_targets = np.concatenate(train_targets)
            
            # 准备验证数据
            eval_set = None
            if val_loader is not None:
                val_features = []
                val_targets = []
                for batch_features, batch_targets in val_loader:
                    if len(batch_features.shape) == 3:
                        batch_features = batch_features[:, -1, :]
                    val_features.append(batch_features.numpy())
                    val_targets.append(batch_targets.numpy())
                
                val_features = np.concatenate(val_features)
                val_targets = np.concatenate(val_targets)
                eval_set = [(val_features, val_targets)]
            
            # 训练XGBoost
            logger.info("Training XGBoost model...")
            self.model.models['xgboost'].fit(train_features, train_targets, eval_set=eval_set)
            
            # 训练深度学习模型
            best_val_loss = float('inf')
            patience_counter = 0
            
            # 训练循环
            for epoch in range(self.config['num_epochs']):
                logger.info(f"Starting epoch {epoch + 1}")
                
                # 训练一个epoch
                metrics = self.train_epoch(train_loader, val_loader)
                
                # 早停检查
                if val_loader is not None:
                    val_loss = metrics['val_loss']
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # 保存最佳模型
                        if save_dir is not None:
                            self.model.save_models(save_dir)
                            
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config['early_stopping_patience']:
                            logger.info("Early stopping triggered")
                            break
                            
            logger.info("Training completed")
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise
            
    def save_checkpoint(self, save_dir: Union[str, Path], epoch: int,
                       metrics: Dict[str, float]):
        """
        保存检查点
        
        Args:
            save_dir: 保存目录
            epoch: 当前epoch
            metrics: 训练指标
        """
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            self.model.save_models(save_dir)
            
            # 保存优化器状态
            optimizer_states = {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            }
            torch.save(optimizer_states, save_dir / 'optimizers.pth')
            
            # 保存训练状态
            checkpoint = {
                'epoch': epoch,
                'metrics': metrics,
                'model_weights': self.model.weights
            }
            torch.save(checkpoint, save_dir / 'checkpoint.pth')
            
            logger.info(f"Saved checkpoint to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise
            
    def load_checkpoint(self, checkpoint_dir: Union[str, Path]):
        """
        加载检查点
        
        Args:
            checkpoint_dir: 检查点目录
        """
        try:
            checkpoint_dir = Path(checkpoint_dir)
            
            # 加载模型
            self.model.load_models(checkpoint_dir)
            
            # 加载优化器状态
            optimizer_states = torch.load(checkpoint_dir / 'optimizers.pth')
            for name, state in optimizer_states.items():
                self.optimizers[name].load_state_dict(state)
                
            # 加载训练状态
            checkpoint = torch.load(checkpoint_dir / 'checkpoint.pth')
            
            logger.info(f"Loaded checkpoint from {checkpoint_dir}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise 