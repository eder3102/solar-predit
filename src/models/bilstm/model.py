"""
BiLSTM模型实现
"""
import torch
import torch.nn as nn
from typing import Dict
import logging
import os

# 根据环境变量选择配置
if os.getenv('SOLAR_ENV') == 'prod':
    from src.config.prod_config import MODEL_CONFIG
else:
    from src.config.dev_config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class BiLSTM(nn.Module):
    """双向LSTM模型类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化BiLSTM模型
        
        Args:
            config: 模型配置字典
        """
        super().__init__()
        
        # 使用默认配置或传入配置
        self.config = config or MODEL_CONFIG['bilstm']
        
        # 模型参数
        self.input_dim = self.config['input_dim']
        self.hidden_dim = self.config['hidden_dim']
        self.num_layers = self.config['num_layers']
        self.dropout = self.config['dropout']
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(self.hidden_dim * 2, 1)  # *2是因为双向
        
        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"Initialized BiLSTM with config: {self.config}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim] 或 [batch_size, sequence_length, input_dim]
            
        Returns:
            输出张量 [batch_size, 1]
        """
        try:
            # 如果输入是2D，添加序列维度
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
                
            # LSTM层
            lstm_out, _ = self.lstm(x)
            
            # 只使用最后一个时间步的输出
            last_hidden = lstm_out[:, -1, :]
            
            # Dropout
            last_hidden = self.dropout_layer(last_hidden)
            
            # 全连接层
            output = self.fc(last_hidden)
            
            return output
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
            
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def get_model_size(self) -> int:
        """
        获取模型大小(参数数量)
        
        Returns:
            模型参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 