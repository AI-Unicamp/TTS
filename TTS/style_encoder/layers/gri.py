import torch
from torch.autograd import Function

'''
    Gradient Reversal Layer implementation (based on: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py)

    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)

    
    Gradient Inverter Layer is a gradient norm adaptation based on: https://www.isca-speech.org/archive/pdfs/ssw_2021/schnell21b_ssw.pdf

    Here we use the only the exponential norm, to avoid infinity collapse
'''

# Function
class GradientInverterLayer_(Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_ 
        lambda_ = grad_output.new_tensor(lambda_)
        grad_input = -lambda_ * grad_output

        l2_norm = torch.exp(grad_output.norm(dim=1, p = 2)) # calculate l2 norm along the batches and apply exp to avoid infinity collapse
        grad_input = grad_input*(1/l2_norm.unsqueeze(1).repeat(1,grad_input.shape[-1])) # Normalize gradients by the norm


        return grad_input, None

# Class to use in models
class GradientInverterLayer(torch.nn.Module):
    def __init__(self, lambda_ = 1):
        super(GradientInverterLayer, self).__init__()
        self.lambda_ = lambda_ 

    def forward(self, x):
        return GradientInverterLayer_.apply(x, self.lambda_)