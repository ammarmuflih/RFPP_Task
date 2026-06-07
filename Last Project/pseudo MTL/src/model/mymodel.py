import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPMTLModel(nn.Module):
    def __init__(self, input_size, num_classes_task1, num_classes_task2):
        super(MLPMTLModel, self).__init__()
        
        self.hidden = nn.Linear(input_size, 32)
        self.out_task1 = nn.Linear(32, num_classes_task1) # digit
        self.out_task2 = nn.Linear(32, num_classes_task2) # speaker
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.dropout(x)
    
        out1 = self.out_task1(x)
        out2 = self.out_task2(x)
        
        return out1, out2