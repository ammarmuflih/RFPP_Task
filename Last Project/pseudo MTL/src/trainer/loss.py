import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, weight_task1=1.0, weight_task2=1.0):
        super(MultiTaskLoss, self).__init__()
        # Menggunakan CrossEntropy standar sebagai basis
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.w1 = weight_task1
        self.w2 = weight_task2

    def forward(self, pred1, target1, pred2, target2):
        loss1 = self.criterion1(pred1, target1.long())
        loss2 = self.criterion2(pred2 ,target2.long())
        
        # Menggabungkan dengan bobot kustom
        total_loss = (self.w1 * loss1) + (self.w2 * loss2)
        
        return total_loss, loss1, loss2