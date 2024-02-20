import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from copy import deepcopy
import math

class Replace(Function):
    @staticmethod
    def forward(ctx, x, x_r):
        return x_r

    @staticmethod
    def backward(ctx, grad):
        return (grad, grad)


class LinearProjGradFunction(Function):
    @staticmethod
    def forward(ctx, input, orth_input, weight, bias=None):
        ctx.save_for_backward(orth_input, weight, bias)
        out = torch.matmul(input, weight.t())
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad):
        orth_input, weight, bias = ctx.saved_variables
        grad_input = grad_orth_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad, weight)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.matmul(grad.t(), orth_input)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = torch.sum(grad, dim=0)

        return grad_input, grad_orth_input, grad_weight, grad_bias


# for feedback alignment or sign symmetric
class DecoupledLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, weight_back, bias=None):
        ctx.save_for_backward(input, weight_back, bias)
        out = torch.matmul(input, weight.t())
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad):
        input, weight_back, bias = ctx.saved_variables
        grad_input = grad_weight = grad_weight_back = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad, weight_back)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad.t(), input)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = torch.sum(grad, dim=0)

        return grad_input, grad_weight, grad_weight_back, grad_bias


# for projection with feedback alignment or sign symmetric
class DecoupledLinearProjGradFunction(Function):
    @staticmethod
    def forward(ctx, input, orth_input, weight, weight_back, bias=None):
        ctx.save_for_backward(orth_input, weight_back, bias)
        out = torch.matmul(input, weight.t())
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad):
        orth_input, weight_back, bias = ctx.saved_variables
        grad_input = grad_orth_input = grad_weight = grad_weight_back = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad, weight_back)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.matmul(grad.t(), orth_input)
        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = torch.sum(grad, dim=0)

        return grad_input, grad_orth_input, grad_weight, grad_weight_back, grad_bias


class LinearProj(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super(LinearProj, self).__init__(in_features, out_features, bias, device, dtype)

    def _forward_with_proj(self, input, weight, bias, proj_func):
        # input: B*C, proj_func: B*C => B*C
        # originally y=Wx calculates the gradient of W by x, replace x by proj_x for gradients
        # this is a simple but costing implementation, TODO: consider the bottom implentation

        with torch.no_grad():
            proj_input = proj_func(input)
            orth_input = (input - proj_input).detach()

        out = LinearProjGradFunction.apply(input, orth_input, weight, bias)
        return out

    # for OTTT-SNN
    def _forward_with_replace(self, input, replace_input, weight, bias):
        out = LinearProjGradFunction.apply(input, replace_input, weight, bias)

        return out

    # for OTTT-SNN
    def _forward_with_proj_replace(self, input, replace_input, weight, bias, proj_func):
        with torch.no_grad():
            proj_replace_input = proj_func(replace_input)
            orth_replace_input = (replace_input - proj_replace_input).detach()

        out = LinearProjGradFunction.apply(input, orth_replace_input, weight, bias)
        return out

    def forward(self, input, projection=False, proj_func=None, replace_input=None):
        if replace_input is not None:
            if projection:
                assert proj_func is not None
                return self._forward_with_proj_replace(input, replace_input, self.weight, self.bias, proj_func)
            else:
                return self._forward_with_replace(input, replace_input, self.weight, self.bias)

        if projection:
            assert proj_func is not None
            return self._forward_with_proj(input, self.weight, self.bias, proj_func)
        else:
            return F.linear(input, self.weight, self.bias)


class SSLinearProj(LinearProj):

    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super(SSLinearProj, self).__init__(in_features, out_features, bias, device, dtype)
        self.scale = 1. / np.sqrt(self.weight.size(1))

    def _forward_with_proj(self, input, weight, weight_back, bias, proj_func):
        # input: B*C, proj_func: B*C => B*C
        # originally y=Wx calculates the gradient of W by x, replace x by proj_x for gradients
        # this is a simple but costing implementation, TODO: consider the bottom implentation

        with torch.no_grad():
            proj_input = proj_func(input)
            orth_input = (input - proj_input).detach()

        out = DecoupledLinearProjGradFunction.apply(input, orth_input, weight, weight_back, bias)
        return out

    def _forward_decouple(self, input, weight, weight_back, bias):
        out = DecoupledLinearFunction.apply(input, weight, weight_back, bias)
        return out

    def forward(self, input, projection=False, proj_func=None):
        weight_back = torch.sign(self.weight) * self.scale
        if projection:
            assert proj_func is not None
            return self._forward_with_proj(input, self.weight, weight_back, self.bias, proj_func)
        else:
            return self._forward_decouple(input, self.weight, weight_back, self.bias)


class SSLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(SSLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.scale = 1. / np.sqrt(self.weight.size(1))

    def _forward_decouple(self, input, weight, weight_back, bias):
        out = DecoupledLinearFunction.apply(input, weight, weight_back, bias)
        return out

    def forward(self, input, projection=False, proj_func=None):
        weight_back = torch.sign(self.weight) * self.scale
        return self._forward_decouple(input, self.weight, weight_back, self.bias)


class FALinearProj(LinearProj):

    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super(FALinearProj, self).__init__(in_features, out_features, bias, device, dtype)
        self.weight_back = nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight_back, a=math.sqrt(5))

    def _forward_with_proj(self, input, weight, weight_back, bias, proj_func):
        # input: B*C, proj_func: B*C => B*C
        # originally y=Wx calculates the gradient of W by x, replace x by proj_x for gradients
        # this is a simple but costing implementation, TODO: consider the bottom implentation

        with torch.no_grad():
            proj_input = proj_func(input)
            orth_input = (input - proj_input).detach()

        out = DecoupledLinearProjGradFunction.apply(input, orth_input, weight, weight_back, bias)
        return out

    def _forward_decouple(self, input, weight, weight_back, bias):
        out = DecoupledLinearFunction.apply(input, weight, weight_back, bias)
        return out

    def forward(self, input, projection=False, proj_func=None):
        weight_back = self.weight_back
        if projection:
            assert proj_func is not None
            return self._forward_with_proj(input, self.weight, weight_back, self.bias, proj_func)
        else:
            return self._forward_decouple(input, self.weight, weight_back, self.bias)


class FALinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(SSLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.weight_back = nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight_back, a=math.sqrt(5))

    def _forward_decouple(self, input, weight, weight_back, bias):
        out = DecoupledLinearFunction.apply(input, weight, weight_back, bias)
        return out

    def forward(self, input, projection=False, proj_func=None):
        weight_back = self.weight_back
        return self._forward_decouple(input, self.weight, weight_back, self.bias)

