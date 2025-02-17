"""
改进的FilterNet模型实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import logging
import os

# 根据环境变量选择配置
if os.getenv('SOLAR_ENV') == 'prod':
    from src.config.prod_config import MODEL_CONFIG
else:
    from src.config.dev_config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class FilterNet(nn.Module):
    """改进的FilterNet模型"""
    
    def __init__(self, config: Dict = None):
        """
        初始化FilterNet模型
        
        Args:
            config: 模型配置字典
        """
        super().__init__()
        
        # 使用默认配置或传入配置
        self.config = config or MODEL_CONFIG['filternet']
        
        # 模型参数
        self.input_dim = self.config['input_dim']
        self.hidden_dim = self.config['hidden_dim']
        self.output_dim = self.config['output_dim']
        self.num_layers = self.config['num_layers']
        self.dropout = self.config['dropout']
        
        # 特征嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),  # 使用GELU激活函数
            nn.LayerNorm(self.hidden_dim),  # 使用LayerNorm
            nn.Dropout(self.dropout)
        )
        
        # 特征交互层
        self.interaction_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(self.num_layers):
            # 每层使用两个线性层
            layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Dropout(self.dropout)
            )
            self.interaction_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim))
        
        # 残差连接
        self.residual = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        
        # 多头自注意力（2头）
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Softmax(dim=1)
            ) for _ in range(2)
        ])
        
        # 特征金字塔
        self.pyramid = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim // (2 ** i))
            for i in range(3)  # 3层金字塔
        ])
        
        # 输出层
        pyramid_dim = self.hidden_dim + self.hidden_dim // 2 + self.hidden_dim // 4
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + pyramid_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"Initialized FilterNet with config: {self.config}")
        
    def forward(self, x):
        # 特征嵌入
        x = self.embedding(x)
        
        # 保存原始嵌入用于残差连接
        identity = x
        
        # 特征交互和残差连接
        for interact, residual, norm in zip(self.interaction_layers, self.residual, self.layer_norms):
            # 交互层
            out = interact(x)
            # 残差连接
            res = residual(identity)
            # 组合并归一化
            x = norm(out + res)
            # 更新残差连接
            identity = x
        
        # 多头注意力
        attention_outputs = []
        for attention in self.attention_heads:
            weights = attention(x)
            attended = x * weights
            attention_outputs.append(attended)
        
        # 特征金字塔
        pyramid_outputs = []
        pyramid_input = x
        for pyramid_layer in self.pyramid:
            pyramid_output = pyramid_layer(pyramid_input)
            pyramid_outputs.append(pyramid_output)
        
        # 合并所有特征
        x = torch.cat([
            *attention_outputs,  # 注意力输出
            *pyramid_outputs,    # 金字塔输出
        ], dim=1)
        
        # 输出预测
        return self.output_layers(x)
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用He初始化
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_model_size(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """获取注意力权重（用于可视化）"""
        weights = []
        with torch.no_grad():
            # 特征嵌入
            x = self.embedding(x)
            # 获取每个注意力头的权重
            for attention in self.attention_heads:
                w = attention(x)
                weights.append(w)
        return torch.cat(weights, dim=1) 