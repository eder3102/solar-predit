"""
训练太阳能发电预测模型
"""
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import logging
import json
from datetime import datetime
from typing import List
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 根据环境变量选择配置
if os.getenv('SOLAR_ENV') == 'prod':
    from src.config.prod_config import (MODEL_CONFIG, TRAIN_CONFIG, 
                                      DATA_CONFIG, FEATURE_CONFIG,
                                      TEST_CONFIG, ENSEMBLE_CONFIG)
else:
    from src.config.dev_config import (MODEL_CONFIG, TRAIN_CONFIG, 
                                     DATA_CONFIG, FEATURE_CONFIG,
                                     TEST_CONFIG, ENSEMBLE_CONFIG)

from src.models.ensemble.model import EnsembleModel
from src.models.filternet.model import FilterNet
from src.models.bilstm.model import BiLSTM
from src.models.xgboost.model import XGBoostModel
from src.data.data_processor import DataProcessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class WeightedMSELoss(nn.Module):
    """加权MSE损失函数"""
    def __init__(self, day_weight: float = 1.0, night_weight: float = 0.5):
        super().__init__()
        self.day_weight = day_weight
        self.night_weight = night_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                is_daytime: torch.Tensor) -> torch.Tensor:
        # 计算MSE
        mse = (pred - target) ** 2
        
        # 应用权重
        weights = torch.where(is_daytime, self.day_weight, self.night_weight)
        weighted_mse = mse * weights.unsqueeze(1)
        
        return weighted_mse.mean()

class RelativeMSELoss(nn.Module):
    """相对MSE损失函数"""
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        relative_error = (pred - target) / (target + self.epsilon)
        return (relative_error ** 2).mean()

class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, l1_weight: float = 0.01, model=None):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_weight = l1_weight
        self.model = model
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MSE损失
        mse_loss = self.mse_loss(pred, target)
        
        # L1正则化
        l1_reg = torch.tensor(0., device=pred.device)
        if self.model is not None:
            for param in self.model.models['filternet'].parameters():
                l1_reg += torch.norm(param, p=1)
            for param in self.model.models['bilstm'].parameters():
                l1_reg += torch.norm(param, p=1)
        
        # 组合损失
        total_loss = mse_loss + self.l1_weight * l1_reg
        
        return total_loss

def add_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    添加滑动窗口特征
    
    Args:
        df: 输入数据
        columns: 需要添加滞后特征的列
        lags: 滞后步数列表
        
    Returns:
        添加滞后特征后的数据
    """
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # 添加差分特征
            if lag > 0:
                df[f'{col}_diff_{lag}'] = df[col] - df[f'{col}_lag_{lag}']
    
    return df

def add_rolling_features(df: pd.DataFrame, columns: List[str], 
                        windows: List[int]) -> pd.DataFrame:
    """
    添加滚动统计特征
    
    Args:
        df: 输入数据
        columns: 需要添加滚动特征的列
        windows: 窗口大小列表
        
    Returns:
        添加滚动特征后的数据
    """
    df = df.copy()
    
    for col in columns:
        for window in windows:
            # 滚动平均
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
            # 滚动标准差
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
            # 滚动最大值
            df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
            # 滚动最小值
            df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
    
    return df

def prepare_data(data_dir):
    """准备训练数据"""
    # 特征列表
    feature_columns = DATA_CONFIG['feature_columns']
    logging.info(f"特征列表: {feature_columns}")
    logging.info(f"特征数量: {len(feature_columns)}")
    
    # 读取数据
    train_data = pd.read_csv(data_dir / 'train/data.csv')
    val_data = pd.read_csv(data_dir / 'validation/data.csv')
    test_data = pd.read_csv(data_dir / 'test/data.csv')
    
    logging.info(f"数据集大小:")
    logging.info(f"训练集: {len(train_data)} 行")
    logging.info(f"验证集: {len(val_data)} 行")
    logging.info(f"测试集: {len(test_data)} 行")
    
    # 设置时间索引
    for df in [train_data, val_data, test_data]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
    
    # 特征工程
    data_processor = DataProcessor()
    train_features, train_targets = data_processor.prepare_data(train_data)
    val_features, val_targets = data_processor.prepare_data(val_data, model_dir='models')
    test_features, test_targets = data_processor.prepare_data(test_data, model_dir='models')
    
    # 目标变量归一化
    train_max_power = train_targets.max()
    logging.info(f"训练集最大发电量: {train_max_power:.2f} W")
    
    train_95th_power = train_targets.quantile(0.95)
    logging.info(f"训练集95%分位数发电量: {train_95th_power:.2f} W")
    
    daytime_data = train_data[train_data['solar_elevation'] > 0]
    daytime_max_power = daytime_data['power'].max()
    logging.info(f"白天最大发电量: {daytime_max_power:.2f} W")
    
    system_power = np.median([train_max_power, train_95th_power, daytime_max_power])
    logging.info(f"估计的系统参考功率: {system_power:.2f} W")
    
    # 保存系统参考功率
    power_info = {
        'system_power': float(system_power),
        'train_max': float(train_max_power),
        'train_95th': float(train_95th_power),
        'daytime_max': float(daytime_max_power)
    }
    with open(data_dir / 'power_info.json', 'w') as f:
        json.dump(power_info, f, indent=4)
    
    # 归一化目标变量
    train_targets = (train_targets.values / system_power).reshape(-1, 1)
    val_targets = (val_targets.values / system_power).reshape(-1, 1)
    test_targets = (test_targets.values / system_power).reshape(-1, 1)
    
    # 裁剪目标值
    train_targets = np.clip(train_targets, 0, 1.1)
    val_targets = np.clip(val_targets, 0, 1.1)
    test_targets = np.clip(test_targets, 0, 1.1)
    
    logging.info(f"目标变量范围:")
    logging.info(f"训练集: [{train_targets.min():.4f}, {train_targets.max():.4f}]")
    logging.info(f"验证集: [{val_targets.min():.4f}, {val_targets.max():.4f}]")
    logging.info(f"测试集: [{test_targets.min():.4f}, {test_targets.max():.4f}]")
    
    return (train_features, train_targets), (val_features, val_targets), \
           (test_features, test_targets), power_info

def save_models(model, save_dir):
    """保存所有模型"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存FilterNet
    torch.save(model.models['filternet'].state_dict(), save_dir / 'filternet.pth')
    
    # 保存BiLSTM
    torch.save(model.models['bilstm'].state_dict(), save_dir / 'bilstm.pth')
    
    # 保存XGBoost
    model.models['xgboost'].save_model(save_dir / 'xgboost.pkl')
    
    # 保存权重
    np.save(save_dir / 'weights.npy', model.weights)
    
    logging.info(f"所有模型已保存到 {save_dir}")

def train_model(train_loader, val_loader, model_config, train_config):
    """三阶段训练模型"""
    logging.info("初始化模型...")
    model = EnsembleModel(ENSEMBLE_CONFIG)  # 使用ENSEMBLE_CONFIG而不是model_config
    
    # 第一阶段：分别训练各个模型
    logging.info("第一阶段：单独训练各个模型...")
    
    # 训练FilterNet
    logging.info("训练FilterNet...")
    filternet_optimizer = torch.optim.Adam(
        model.models['filternet'].parameters(),
        lr=train_config['learning_rate']['stage1']['initial']
    )
    filternet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        filternet_optimizer,
        mode='min',
        factor=train_config['learning_rate']['stage1']['factor'],
        patience=train_config['learning_rate']['stage1']['patience'],
        min_lr=train_config['learning_rate']['stage1']['min']
    )
    train_single_model('filternet', model, train_loader, val_loader, 
                      filternet_optimizer, filternet_scheduler, train_config)
    
    # 训练BiLSTM
    logging.info("训练BiLSTM...")
    bilstm_optimizer = torch.optim.Adam(
        model.models['bilstm'].parameters(),
        lr=train_config['learning_rate']['stage1']['initial']
    )
    bilstm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        bilstm_optimizer,
        mode='min',
        factor=train_config['learning_rate']['stage1']['factor'],
        patience=train_config['learning_rate']['stage1']['patience'],
        min_lr=train_config['learning_rate']['stage1']['min']
    )
    train_single_model('bilstm', model, train_loader, val_loader, 
                      bilstm_optimizer, bilstm_scheduler, train_config)
    
    # 训练XGBoost
    logging.info("训练XGBoost...")
    train_features = []
    train_targets = []
    for batch_features, batch_targets in train_loader:
        train_features.append(batch_features.numpy())
        train_targets.append(batch_targets.numpy())
    train_features = np.concatenate(train_features)
    train_targets = np.concatenate(train_targets)
    
    val_features = []
    val_targets = []
    for batch_features, batch_targets in val_loader:
        val_features.append(batch_features.numpy())
        val_targets.append(batch_targets.numpy())
    val_features = np.concatenate(val_features)
    val_targets = np.concatenate(val_targets)
    
    # 更新XGBoost训练参数
    xgb_params = model.models['xgboost'].model.get_params()
    xgb_params.update({
        'early_stopping_rounds': train_config['early_stopping_patience']['stage1'],
        'eval_metric': 'rmse',
        'verbose': True
    })
    model.models['xgboost'].model.set_params(**xgb_params)
    
    # 训练XGBoost模型
    model.models['xgboost'].model.fit(
        train_features, train_targets.ravel(),
        eval_set=[(val_features, val_targets.ravel())],
        verbose=True
    )
    
    # 第二阶段：固定权重训练
    logging.info("第二阶段：固定权重训练...")
    model.weights = train_config['initial_weights']
    logging.info(f"使用固定权重: {model.weights}")
    
    optimizer = torch.optim.Adam([
        {'params': model.models['filternet'].parameters()},
        {'params': model.models['bilstm'].parameters()}
    ], lr=train_config['learning_rate']['stage2']['initial'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_config['learning_rate']['stage2']['factor'],
        patience=train_config['learning_rate']['stage2']['patience'],
        min_lr=train_config['learning_rate']['stage2']['min']
    )
    
    criterion = CombinedLoss(l1_weight=train_config['l1_weight'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(train_config['stage2_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, train_config)
        val_loss = validate_epoch(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_models(train_config['model_save_path'])
            patience_counter = 0
            logging.info(f"保存新的最佳模型，验证损失: {val_loss:.6f}")
        else:
            patience_counter += 1
            
        if patience_counter >= train_config['early_stopping_patience']['stage2']:
            logging.info(f"Early stopping at epoch {epoch}")
            break
            
        logging.info(f"Stage 2 - Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    # 第三阶段：动态权重训练
    logging.info("第三阶段：启用动态权重训练...")
    optimizer = torch.optim.Adam([
        {'params': model.models['filternet'].parameters()},
        {'params': model.models['bilstm'].parameters()}
    ], lr=train_config['learning_rate']['stage3']['initial'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_config['learning_rate']['stage3']['factor'],
        patience=train_config['learning_rate']['stage3']['patience'],
        min_lr=train_config['learning_rate']['stage3']['min']
    )
    
    remaining_epochs = train_config['max_epochs'] - train_config['stage3_start_epoch']
    patience_counter = 0
    
    for epoch in range(remaining_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, train_config)
        val_loss, val_predictions = validate_epoch_with_predictions(model, val_loader, criterion)
        
        # 更新模型权重
        update_ensemble_weights(model, val_predictions, val_loader, train_config)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_models(train_config['model_save_path'])
            patience_counter = 0
            logging.info(f"保存新的最佳模型，验证损失: {val_loss:.6f}")
        else:
            patience_counter += 1
            
        if patience_counter >= train_config['early_stopping_patience']['stage3']:
            logging.info(f"Early stopping at epoch {epoch}")
            break
            
        logging.info(f"Stage 3 - Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        logging.info(f"当前集成权重: {model.weights}")
    
    logging.info("训练完成")
    return model

def train_single_model(model_name, model, train_loader, val_loader, optimizer, scheduler, train_config):
    """训练单个模型"""
    best_val_loss = float('inf')
    patience_counter = 0
    criterion = nn.MSELoss()
    
    for epoch in range(train_config['stage1_epochs']):
        model.models[model_name].train()
        train_loss = 0
        batch_count = 0
        
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            predictions = model.models[model_name](batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.models[model_name].parameters(), train_config['gradient_clip'])
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = train_loss / batch_count
        
        # 验证
        model.models[model_name].eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                predictions = model.models[model_name](batch_features)
                loss = criterion(predictions, batch_targets)
                val_loss += loss.item()
                val_batch_count += 1
        
        avg_val_loss = val_loss / val_batch_count
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 只保存当前正在训练的模型
            if model_name == 'filternet':
                torch.save(model.models['filternet'].state_dict(), 
                         Path(train_config['model_save_path']) / 'filternet.pth')
            elif model_name == 'bilstm':
                torch.save(model.models['bilstm'].state_dict(), 
                         Path(train_config['model_save_path']) / 'bilstm.pth')
            patience_counter = 0
            logging.info(f"{model_name} - 保存新的最佳模型，验证损失: {avg_val_loss:.6f}")
        else:
            patience_counter += 1
            
        if patience_counter >= train_config['early_stopping_patience']['stage1']:
            logging.info(f"{model_name} - Early stopping at epoch {epoch}")
            break
            
        logging.info(f"{model_name} - Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

def update_ensemble_weights(model, val_predictions, val_loader, train_config):
    """更新集成模型权重"""
    val_targets = []
    for _, batch_targets in val_loader:
        val_targets.extend(batch_targets.numpy())
    val_targets = np.array(val_targets)
    
    # 计算每个模型的MAE
    scores = {}
    for model_name, preds in val_predictions.items():
        mae = np.mean(np.abs(preds - val_targets))
        scores[model_name] = mae + 1e-6  # 添加小的平滑因子
    
    # 使用softmax计算新权重
    total_error = sum(np.exp(-score) for score in scores.values())
    new_weights = {
        model_name: max(
            train_config['weight_update']['min_weight'],
            np.exp(-score) / total_error
        )
        for model_name, score in scores.items()
    }
    
    # 重新归一化权重
    total_weight = sum(new_weights.values())
    new_weights = {k: v/total_weight for k, v in new_weights.items()}
    
    # 使用平滑系数更新权重
    alpha = train_config['weight_update']['smoothing_factor']
    model.weights = {
        k: alpha * model.weights[k] + (1-alpha) * new_weights[k]
        for k in model.weights
    }
    
    logging.info(f"更新后的集成权重: {model.weights}")

def validate_epoch_with_predictions(model, val_loader, criterion):
    """验证轮次并返回各模型的预测结果"""
    model.models['filternet'].eval()
    model.models['bilstm'].eval()
    val_loss = 0
    batch_count = 0
    predictions = {
        'filternet': [],
        'bilstm': [],
        'xgboost': []
    }
    
    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            # 获取各个模型的预测
            filternet_pred = model.models['filternet'](batch_features)
            bilstm_pred = model.models['bilstm'](batch_features)
            xgboost_pred = torch.FloatTensor(
                model.models['xgboost'].predict(
                    batch_features.numpy()
                ).reshape(-1, 1)
            )
            
            # 收集预测结果
            predictions['filternet'].extend(filternet_pred.numpy())
            predictions['bilstm'].extend(bilstm_pred.numpy())
            predictions['xgboost'].extend(xgboost_pred.numpy())
            
            # 计算融合预测和损失
            ensemble_pred = (
                model.weights['filternet'] * filternet_pred +
                model.weights['bilstm'] * bilstm_pred +
                model.weights['xgboost'] * xgboost_pred
            )
            
            loss = criterion(ensemble_pred, batch_targets)
            val_loss += loss.item()
            batch_count += 1
    
    avg_val_loss = val_loss / batch_count
    predictions = {k: np.array(v) for k, v in predictions.items()}
    
    return avg_val_loss, predictions

def evaluate_model(model, test_loader, criterion, device, power_info):
    """评估模型"""
    model.models['filternet'].eval()
    model.models['bilstm'].eval()
    test_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            # 获取各个模型的预测
            filternet_pred = model.models['filternet'](batch_features)
            bilstm_pred = model.models['bilstm'](batch_features)
            
            if isinstance(batch_features, torch.Tensor):
                features_np = batch_features.detach().numpy()
            else:
                features_np = batch_features
                
            xgboost_pred = torch.FloatTensor(
                model.models['xgboost'].predict(features_np).reshape(-1, 1)
            )
            
            # 加权融合
            predictions_batch = (
                model.weights['filternet'] * filternet_pred +
                model.weights['bilstm'] * bilstm_pred +
                model.weights['xgboost'] * xgboost_pred
            )
            
            # 计算损失
            loss = criterion(predictions_batch, batch_targets)
            test_loss += loss.item()
            
            # 转换回原始功率
            pred = predictions_batch.cpu().numpy() * power_info['system_power']
            actual = batch_targets.cpu().numpy() * power_info['system_power']
            
            predictions.extend(pred)
            actuals.extend(actual)
    
    test_loss /= len(test_loader)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 计算评估指标
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - actuals.mean()) ** 2)
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1))) * 100
    
    # 计算每个时段的指标
    daytime_mask = actuals > 0.1 * power_info['system_power']  # 发电量大于10%额定功率视为白天
    if daytime_mask.any():
        day_metrics = {
            'day_mae': float(np.mean(np.abs(predictions[daytime_mask] - actuals[daytime_mask]))),
            'day_rmse': float(np.sqrt(np.mean((predictions[daytime_mask] - actuals[daytime_mask]) ** 2))),
            'day_mape': float(np.mean(np.abs((predictions[daytime_mask] - actuals[daytime_mask]) / 
                                            actuals[daytime_mask])) * 100)
        }
    else:
        day_metrics = {'day_mae': None, 'day_rmse': None, 'day_mape': None}
    
    metrics = {
        'test_loss': float(test_loss),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        **day_metrics,
        'max_actual': float(actuals.max()),
        'max_predicted': float(predictions.max()),
        'mean_actual': float(actuals.mean()),
        'mean_predicted': float(predictions.mean())
    }
    
    return metrics, predictions, actuals

def create_visualizations(predictions, actuals, power_info):
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
            x=np.arange(len(actuals)),
            y=actuals.flatten(),
            name='实际发电量',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(predictions)),
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
    # 由于没有时间戳，我们按照48个点为一天来计算
    points_per_day = 48
    num_days = len(actuals) // points_per_day
    daily_actual = []
    daily_pred = []
    
    for i in range(num_days):
        start_idx = i * points_per_day
        end_idx = (i + 1) * points_per_day
        daily_actual.append(np.sum(actuals[start_idx:end_idx]))
        daily_pred.append(np.sum(predictions[start_idx:end_idx]))
    
    # 添加日发电量对比图
    fig.add_trace(
        go.Bar(
            x=np.arange(len(daily_actual)),
            y=np.array(daily_actual) / 1000,  # 转换为kWh
            name='实际日发电量',
            marker_color='#1f77b4',
            opacity=0.7
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=np.arange(len(daily_pred)),
            y=np.array(daily_pred) / 1000,  # 转换为kWh
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
    fig.update_xaxes(title_text='时间点', row=1, col=1, gridcolor='#f0f0f0')
    fig.update_xaxes(title_text='预测误差 (W)', row=2, col=1, gridcolor='#f0f0f0')
    fig.update_xaxes(title_text='天数', row=3, col=1, gridcolor='#f0f0f0')
    
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
    daily_errors = np.array(daily_pred) - np.array(daily_actual)
    metrics['daily_mae'] = np.mean(np.abs(daily_errors))
    metrics['daily_rmse'] = np.sqrt(np.mean(daily_errors ** 2))
    metrics['daily_mape'] = np.mean(np.abs(daily_errors / np.array(daily_actual))) * 100
    
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

def train_epoch(model, train_loader, optimizer, criterion, train_config):
    """训练一个epoch"""
    model.models['filternet'].train()
    model.models['bilstm'].train()
    train_loss = 0
    batch_count = 0
    
    for batch_features, batch_targets in train_loader:
        # 确保数据是FloatTensor类型
        if not isinstance(batch_features, torch.FloatTensor):
            batch_features = torch.FloatTensor(batch_features)
        if not isinstance(batch_targets, torch.FloatTensor):
            batch_targets = torch.FloatTensor(batch_targets)
        
        optimizer.zero_grad()
        
        # 获取各个模型的预测
        filternet_pred = model.models['filternet'](batch_features)
        bilstm_pred = model.models['bilstm'](batch_features)
        xgboost_pred = torch.FloatTensor(
            model.models['xgboost'].predict(
                batch_features.numpy()
            ).reshape(-1, 1)
        )
        
        # 加权融合
        predictions = (
            model.weights['filternet'] * filternet_pred +
            model.weights['bilstm'] * bilstm_pred +
            model.weights['xgboost'] * xgboost_pred
        )
        
        # 计算损失
        loss = criterion(predictions, batch_targets)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            model.models['filternet'].parameters(),
            train_config['gradient_clip']
        )
        torch.nn.utils.clip_grad_norm_(
            model.models['bilstm'].parameters(),
            train_config['gradient_clip']
        )
        
        optimizer.step()
        train_loss += loss.item()
        batch_count += 1
    
    return train_loss / batch_count

def validate_epoch(model, val_loader, criterion):
    """验证一个epoch"""
    model.models['filternet'].eval()
    model.models['bilstm'].eval()
    val_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            # 确保数据是FloatTensor类型
            if not isinstance(batch_features, torch.FloatTensor):
                batch_features = torch.FloatTensor(batch_features)
            if not isinstance(batch_targets, torch.FloatTensor):
                batch_targets = torch.FloatTensor(batch_targets)
            
            # 获取各个模型的预测
            filternet_pred = model.models['filternet'](batch_features)
            bilstm_pred = model.models['bilstm'](batch_features)
            xgboost_pred = torch.FloatTensor(
                model.models['xgboost'].predict(
                    batch_features.numpy()
                ).reshape(-1, 1)
            )
            
            # 加权融合
            predictions = (
                model.weights['filternet'] * filternet_pred +
                model.weights['bilstm'] * bilstm_pred +
                model.weights['xgboost'] * xgboost_pred
            )
            
            # 计算损失
            loss = criterion(predictions, batch_targets)
            val_loss += loss.item()
            batch_count += 1
    
    return val_loss / batch_count

def main():
    """主函数"""
    try:
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 创建必要的目录
        Path('models').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)
        
        # 准备数据
        data_dir = Path(DATA_CONFIG['train_data_path']).parent
        (train_features, train_targets), (val_features, val_targets), \
        (test_features, test_targets), power_info = prepare_data(data_dir)
        
        # 创建数据加载器
        train_dataset = TimeSeriesDataset(
            torch.FloatTensor(train_features),
            torch.FloatTensor(train_targets)
        )
        val_dataset = TimeSeriesDataset(
            torch.FloatTensor(val_features),
            torch.FloatTensor(val_targets)
        )
        test_dataset = TimeSeriesDataset(
            torch.FloatTensor(test_features),
            torch.FloatTensor(test_targets)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=TEST_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {device}')
        
        # 训练模型
        model = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model_config=MODEL_CONFIG,
            train_config=TRAIN_CONFIG
        )
        
        # 评估模型
        metrics, predictions, actuals = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=CombinedLoss(l1_weight=0.01, model=model),
            device=device,
            power_info=power_info
        )
        
        # 创建可视化
        create_visualizations(
            predictions=predictions,
            actuals=actuals,
            power_info=power_info
        )
        
        # 保存结果
        results = {
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_config': MODEL_CONFIG,
            'train_config': TRAIN_CONFIG,
            'feature_config': FEATURE_CONFIG,
            'power_info': power_info
        }
        
        with open('results/training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        logging.info("训练完成!")
        
    except Exception as e:
        logging.error(f"训练出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 