import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from copy import deepcopy
import time

class Replace(Function):
    @staticmethod
    def forward(ctx, x, x_r):
        return x_r

    @staticmethod
    def backward(ctx, grad):
        return (grad, grad)


class ConvProjGradFunction(Function):
    @staticmethod
    def forward(ctx, input, orth_input, weight, bias=None):
        ctx.save_for_backward(orth_input, weight, bias)
        out = torch.matmul(weight.reshape(weight.size(0), -1), input)
        if bias is not None:
            out = out + bias.unsqueeze(1)
        return out

    @staticmethod
    def backward(ctx, grad):
        orth_input, weight, bias = ctx.saved_variables
        grad_input = grad_orth_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(weight.reshape(weight.size(0), -1).t(), grad)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.sum(torch.matmul(grad, orth_input.transpose(1, 2)), dim=0).reshape(weight.shape)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = torch.sum(grad, dim=(0, 2))

        return grad_input, grad_orth_input, grad_weight, grad_bias


# for OTTT-SNN
# we can also use the above function, this is for possible acceleration of conv operations
class ReplaceConvProjGradFunction(Function):
    @staticmethod
    def forward(ctx, conv_output, replace_input, weight, bias=None):
        # conv_output: B*C_out*H*W, replace_input: B*(C_in*ksize^2)*(H*W), weight: C_out*C_in*ksize^2
        ctx.save_for_backward(replace_input, weight, bias)
        return conv_output

    @staticmethod
    def backward(ctx, grad):
        replace_input, weight, bias = ctx.saved_variables
        grad_conv_output = grad_replace_input = grad_weight = grad_bias = None
        grad = grad.reshape(grad.size(0), grad.size(1), -1)
        if ctx.needs_input_grad[1]:
            grad_replace_input = torch.matmul(weight.reshape(weight.size(0), -1).t(), grad)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.sum(torch.matmul(grad, replace_input.transpose(1, 2)), dim=0).reshape(weight.shape)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = torch.sum(grad, dim=(0, 2))

        return grad_conv_output, grad_replace_input, grad_weight, grad_bias


# for sign symmetric
class DecoupledConvFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, weight_back, bias=None):
        ctx.save_for_backward(input, weight_back, bias)
        out = torch.matmul(weight.reshape(weight.size(0), -1), input)
        if bias is not None:
            out = out + bias.unsqueeze(1)
        return out

    @staticmethod
    def backward(ctx, grad):
        input, weight_back, bias = ctx.saved_variables
        grad_input = grad_weight = grad_weight_back = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(weight_back.reshape(weight_back.size(0), -1).t(), grad)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.sum(torch.matmul(grad, input.transpose(1, 2)), dim=0).reshape(weight_back.shape)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = torch.sum(grad, dim=(0, 2))

        return grad_input, grad_weight, grad_weight_back, grad_bias


# for projection with sign symmetric
class DecoupledConvProjGradFunction(Function):
    @staticmethod
    def forward(ctx, input, orth_input, weight, weight_back, bias=None):
        ctx.save_for_backward(orth_input, weight_back, bias)
        out = torch.matmul(weight.reshape(weight.size(0), -1), input)
        if bias is not None:
            out = out + bias.unsqueeze(1)
        return out

    @staticmethod
    def backward(ctx, grad):
        orth_input, weight_back, bias = ctx.saved_variables
        grad_input = grad_orth_input = grad_weight = grad_weight_back = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(weight_back.reshape(weight_back.size(0), -1).t(), grad)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.sum(torch.matmul(grad, orth_input.transpose(1, 2)), dim=0).reshape(weight_back.shape)
        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = torch.sum(grad, dim=(0, 2))

        return grad_input, grad_orth_input, grad_weight, grad_weight_back, grad_bias


class Conv2dProj(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', device=None, dtype=None, proj_type='input'):
        super(Conv2dProj, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        # now only support common settings
        assert groups == 1
        self.proj_type = proj_type
        if self.proj_type != 'input':
            self.h = None

    def _conv_forward_with_proj(self, input, weight, bias, proj_func):
        # input: B*C*H*W, proj_func: (B*NH*NW) * (C*K*K) => (B*NH*NW) * (C*K*K)
        # originally y=Wx calculates the gradient of W by x, replace x by proj_x for gradients
        # this is a simple but costing implementation, TODO: consider the bottom implentation
        #start = time.time()

        # update: proj_type=='weight' enables acceleration to project weight gradients instead of presynaptic activities
        # this can leverage bottom acceleration for convolutional operations

        if self.proj_type == 'input':
            _, _, H, W = input.shape

            if self.padding_mode != 'zeros':
                input_pad = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
                # input_unfold: B * (C * kernel_size^2) * L
                input_unfold = F.unfold(input_pad, self.kernel_size, dilation=self.dilation, stride=self.stride)
                _, _, H, W = input_pad.shape
                H_ = int(np.floor((H - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / float(self.stride[0]) + 1))
                W_ = int(np.floor((W - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / float(self.stride[1]) + 1))
            else:
                input_unfold = F.unfold(input, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

                H_ = int(np.floor((H + self.padding[0] * 2 - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / float(self.stride[0]) + 1))
                W_ = int(np.floor((W + self.padding[1] * 2 - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / float(self.stride[1]) + 1))

            #print(time.time() - start)

            with torch.no_grad():
                # shape: B, L, C * kernel_size^2
                shape = input_unfold.transpose(1, 2).shape
                proj_input = proj_func(input_unfold.transpose(1, 2).reshape(-1, shape[2])).reshape(shape).transpose(1, 2)
                orth_input = (input_unfold - proj_input).detach()

            out = ConvProjGradFunction.apply(input_unfold, orth_input, weight, bias)

            #print(time.time() - start)
            # F.fold function is slow
            #out = F.fold(out, (H_, W_), (1, 1))
            out = out.reshape(out.shape[0], out.shape[1], H_, W_)
            #print(time.time() - start)
        else:
            def hook_func(grad):
                return (grad.reshape(grad.size(0), -1) - proj_func(grad.reshape(grad.size(0), -1))).reshape(grad.shape)
            self.h = weight.register_hook(hook_func)
            out = self._conv_forward(input, weight, bias)

        return out

    # for OTTT-SNN
    def _conv_forward_replace(self, input, replace_input, weight, bias):

        with torch.no_grad():
            conv_output = self._conv_forward(input, weight, bias).detach()

        in_for_grad = Replace.apply(input, replace_input)
        out_for_grad = self._conv_forward(in_for_grad, weight, bias)
        out = Replace.apply(out_for_grad, conv_output)

        return out

    # for OTTT-SNN, TODO: consider proj_type=='weight'
    def _conv_forward_with_proj_replace(self, input, replace_input, weight, bias, proj_func):
        # input: B*C*H*W, proj_func: (B*NH*NW) * (C*K*K) => (B*NH*NW) * (C*K*K)
        # originally y=Wx calculates the gradient of W by x, replace x by proj_x for gradients
        # this is a simple but costing implementation, TODO: consider the bottom implentation
        #start = time.time()

        with torch.no_grad():
            conv_output = self._conv_forward(input, weight, bias).detach()

        # for gradient propagation
        replace_input = Replace.apply(input, replace_input)

        _, _, H, W = replace_input.shape

        if self.padding_mode != 'zeros':
            input_pad = F.pad(replace_input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            # input_unfold: B * (C * kernel_size^2) * L
            input_unfold = F.unfold(input_pad, self.kernel_size, dilation=self.dilation, stride=self.stride)
            _, _, H, W = input_pad.shape
            H_ = int(np.floor((H - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / float(self.stride[0]) + 1))
            W_ = int(np.floor((W - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / float(self.stride[1]) + 1))
        else:
            input_unfold = F.unfold(replace_input, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

            H_ = int(np.floor((H + self.padding[0] * 2 - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / float(self.stride[0]) + 1))
            W_ = int(np.floor((W + self.padding[1] * 2 - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / float(self.stride[1]) + 1))

        #print(time.time() - start)

        with torch.no_grad():
            shape = input_unfold.transpose(1, 2).shape
            proj_input = proj_func(input_unfold.transpose(1, 2).reshape(-1, shape[2])).reshape(shape).transpose(1, 2)
            orth_input = (input_unfold - proj_input).detach()

        orth_input = Replace.apply(input_unfold, orth_input)

        out = ReplaceConvProjGradFunction.apply(conv_output, orth_input, weight, bias)

        return out

    def forward(self, input, projection=False, proj_func=None, replace_input=None):
        if self.proj_type != 'input' and self.h is not None:
            self.h.remove()

        if replace_input is not None:
            if projection:
                assert proj_func is not None
                return self._conv_forward_with_proj_replace(input, replace_input, self.weight, self.bias, proj_func)
            else:
                return self._conv_forward_replace(input, replace_input, self.weight, self.bias)

        if projection:
            assert proj_func is not None
            return self._conv_forward_with_proj(input, self.weight, self.bias, proj_func)
        else:
            return self._conv_forward(input, self.weight, self.bias)


# weight standardization
class WSConv2dProj(Conv2dProj):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, gain=True, eps=1e-4, proj_type='input'):
        super(WSConv2dProj, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype, proj_type)
        # now only support common settings
        assert groups == 1

        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps
        self.mean = nn.Parameter(torch.zeros(self.weight.shape[0]), requires_grad=False)
        self.var = nn.Parameter(torch.ones(self.weight.shape[0]), requires_grad=False)
        self.fix_affine = False

    def fix_ws(self):
        if not self.fix_affine:
            self.gain.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
            self.mean.data = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True).detach().clone()
            self.var.data = torch.var(self.weight, axis=[1, 2, 3], keepdims=True).detach().clone()
            self.fix_affine = True

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        if self.fix_affine:
            weight = (self.weight - self.mean) / ((self.var * fan_in + self.eps) ** 0.5)
        else:
            mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
            var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
            weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)

        if self.gain is not None:
            weight = weight * self.gain

        return weight

    def forward(self, input, projection=False, proj_func=None, replace_input=None):
        if self.proj_type != 'input' and self.h is not None:
            self.h.remove()

        if replace_input is not None:
            if projection:
                assert self.fix_affine
                assert proj_func is not None
                return self._conv_forward_with_proj_replace(input, replace_input, self.get_weight(), self.bias, proj_func)
            else:
                return self._conv_forward_replace(input, replace_input, self.get_weight(), self.bias)

        if projection:
            assert self.fix_affine
            assert proj_func is not None
            return self._conv_forward_with_proj(input, self.get_weight(), self.bias, proj_func)
        else:
            return self._conv_forward(input, self.get_weight(), self.bias)


# TODO implement for OTTT-SNN
# TODO implement for proj_type=='weight'
class SSConv2dProj(Conv2dProj):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', device=None, dtype=None):
        super(SSConv2dProj, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        # now only support common settings
        assert groups == 1
        self.scale = np.sqrt(2 / (self.kernel_size[0] * self.kernel_size[1] * self.out_channels))

    def _conv_forward_with_proj(self, input, weight, weight_back, bias, proj_func):
        # input: B*C*H*W, proj_func: (B*NH*NW) * (C*K*K) => (B*NH*NW) * (C*K*K)
        # originally y=Wx calculates the gradient of W by x, replace x by proj_x for gradients
        # this is a simple but costing implementation, TODO: consider the bottom implentation

        _, _, H, W = input.shape

        if self.padding_mode != 'zeros':
            input_pad = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            # input_unfold: B * (C * kernel_size^2) * L
            input_unfold = F.unfold(input_pad, self.kernel_size, dilation=self.dilation, stride=self.stride)
            _, _, H, W = input_pad.shape
            H_ = int(np.floor((H - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / float(self.stride[0]) + 1))
            W_ = int(np.floor((W - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / float(self.stride[1]) + 1))
        else:
            input_unfold = F.unfold(input, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

            H_ = int(np.floor((H + self.padding[0] * 2 - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / float(self.stride[0]) + 1))
            W_ = int(np.floor((W + self.padding[1] * 2 - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / float(self.stride[1]) + 1))

        with torch.no_grad():
            # shape: B, L, C * kernel_size^2
            shape = input_unfold.transpose(1, 2).shape
            proj_input = proj_func(input_unfold.transpose(1, 2).reshape(-1, shape[2])).reshape(shape).transpose(1, 2)
            orth_input = (input_unfold - proj_input).detach()

        out = DecoupledConvProjGradFunction.apply(input_unfold, orth_input, weight, weight_back, bias)
        #out = F.fold(out, (H_, W_), (1, 1))
        out = out.reshape(out.shape[0], out.shape[1], H_, W_)

        return out

    def _conv_forward_decouple(self, input, weight, weight_back, bias):
        _, _, H, W = input.shape

        if self.padding_mode != 'zeros':
            input_pad = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            # input_unfold: B * (C * kernel_size^2) * L
            input_unfold = F.unfold(input_pad, self.kernel_size, dilation=self.dilation, stride=self.stride)
            _, _, H, W = input_pad.shape
            H_ = int(np.floor((H - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / float(self.stride[0]) + 1))
            W_ = int(np.floor((W - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / float(self.stride[1]) + 1))
        else:
            input_unfold = F.unfold(input, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

            H_ = int(np.floor((H + self.padding[0] * 2 - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / float(self.stride[0]) + 1))
            W_ = int(np.floor((W + self.padding[1] * 2 - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / float(self.stride[1]) + 1))

        out = DecoupledConvFunction.apply(input_unfold, weight, weight_back, bias)
        #out = F.fold(out, (H_, W_), (1, 1))
        out = out.reshape(out.shape[0], out.shape[1], H_, W_)

        return out

    def forward(self, input, projection=False, proj_func=None):
        weight_back = torch.sign(self.weight) * self.scale
        if projection:
            assert proj_func is not None
            return self._conv_forward_with_proj(input, self.weight, weight_back, self.bias, proj_func)
        else:
            return self._conv_forward_decouple(input, self.weight, weight_back, self.bias)


