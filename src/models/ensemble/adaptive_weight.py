class AdaptiveWeightLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 三个模型的权重
        )
        
    def forward(self, x):
        uncertainties = self.uncertainty_net(x.mean(dim=1))
        weights = F.softmax(-uncertainties, dim=1)  # 不确定性越小权重越大
        return weights 