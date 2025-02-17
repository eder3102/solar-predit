class DistillTrainer:
    def __init__(self, teacher, student):
        self.teacher = teacher.eval()
        self.student = student.train()
        
    def compute_loss(self, inputs, targets):
        with torch.no_grad():
            t_logits = self.teacher(inputs)
            
        s_logits = self.student(inputs)
        
        # 软目标损失
        loss_soft = KLDivLoss()(F.log_softmax(s_logits/T, dim=1),
                               F.softmax(t_logits/T, dim=1)) * (T**2)
        
        # 真实标签损失
        loss_hard = F.mse_loss(s_logits, targets)
        
        return 0.7*loss_soft + 0.3*loss_hard 