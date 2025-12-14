import math
import torch

class LORALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.a_mat = torch.nn.Parameter(torch.empty((in_dim, rank)))
        torch.nn.init.kaiming_uniform_(self.a_mat, a=math.sqrt(5))
        self.b_mat = torch.nn.Parameter(torch.zeros((rank, out_dim)))
        self.alpha = alpha
        self.rank = rank
    
    def forward(self, x):
        x = self.alpha * (x @ self.a_mat @ self.b_mat)
        return x
    
class LoRAInjectedLinear(torch.nn.Module):
    def __init__(self, original_linear, rank, alpha):
        super().__init__()
        self.original_linear = original_linear
        self.lora_layer = LORALayer(original_linear.in_features, original_linear.out_features, rank, alpha)
    
    def forward(self, x):
        return self.original_linear(x) + self.lora_layer(x)
