import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype) 

        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
            self.lora_scaling = self.lora_scaling = lora_alpha / lora_rank

            #TODO: Fill in the "..."
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank, device=device, dtype=dtype))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            #TODO: Initialize both lora_A and lora_B with torch.nn.init. Refer to the paper to see how each is initialize
            #Hint: lora_A is initialized using kaiming_uniform_ using negative slope (a) as math.sqrt(5)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #TODO: return input after the forward pass
        #TODO: Remember to use dropout on the input before multiplying with lora_B and lora_A if the weights are not merged

        if self.has_weights_merged:
            result = nn.Linear.forward(self, input) 
            return result
        
        base_output = nn.Linear.forward(self, input)
        if self.is_lora():
            input_dropped = self.lora_dropout(input)
            lora_adjustment = F.linear(F.linear(input_dropped, self.lora_A), self.lora_B)
            base_output = base_output + (self.lora_scaling * lora_adjustment)  
        return base_output  

        
    def train(self, mode: bool = True) -> "LoRALinear":
        
        #TODO: Set the linear layer into train mode
        #Hint: Make sure to demerge LORA matrices if already merged
        if mode and self.has_weights_merged and self.is_lora():
            self.weight.data -= (self.lora_B @ self.lora_A) * self.lora_scaling
            self.has_weights_merged = False

        nn.Linear.train(self, mode) 
        return self if mode else super().train(mode)

    def eval(self) -> "LoRALinear":
        #TODO: Set the linear layer into eval mode
        #Hint: Make sure to merge LORA matrices if already demerged
        
        if not self.has_weights_merged and self.is_lora():
            self.weight.data += (self.lora_B @ self.lora_A) * self.lora_scaling
            self.has_weights_merged = True

        eval_result = nn.Linear.eval(self)  
        return eval_result or self  
        
    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:
    #TODO: Loop through parameters and mark some as trainable. Which ones should these be?
    #Hint: How do you mark a parameter as trainable (or not trainable)?
    for param in model.parameters():
        param.requires_grad = False  

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:  
            param.requires_grad = True

    return model