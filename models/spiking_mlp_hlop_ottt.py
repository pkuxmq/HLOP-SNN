import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.neuron_ottt import OnlineIFNode, OnlineLIFNode
import modules.surrogate as surrogate
from modules.proj_conv import Conv2dProj, SSConv2dProj
from modules.proj_linear import LinearProj, SSLinear, SSLinearProj, FALinear, FALinearProj
from modules.hlop_module import HLOP
import numpy as np


__all__ = [
    'spiking_MLP_ottt'
]


class WrapedSNNOp(nn.Module):

    def __init__(self, op):
        super(WrapedSNNOp, self).__init__()
        self.op = op

    def forward(self, x, **kwargs):
        require_wrap = kwargs.get('require_wrap', True)
        projection = kwargs.get('projection', False)
        proj_func = kwargs.get('proj_func', None)
        if require_wrap:
            B = x.shape[0] // 2
            spike = x[:B]
            rate = x[B:]
            if isinstance(self.op, Conv2dProj) or isinstance(self.op, LinearProj):
                output = self.op(spike, projection, proj_func, rate)
            else:
                with torch.no_grad():
                    out = self.op(spike).detach()
                in_for_grad = Replace.apply(spike, rate)
                out_for_grad = self.op(in_for_grad)
                output = Replace.apply(out_for_grad, out)
            return output
        else:
            if isinstance(self.op, Conv2dProj) or isinstance(self.op, LinearProj):
                return self.op(x, projection, proj_func)
            else:
                return self.op(x)


class spiking_MLP(nn.Module):
    def __init__(self, num_classes=10, n_hidden=800, share_classifier=True, neuron_type='lif', ss=False, fa=False, hlop_spiking=False, hlop_spiking_scale=20., hlop_spiking_timesteps=1000., **kwargs):
        super(spiking_MLP, self).__init__()
        self.neuron_type = neuron_type
        if self.neuron_type == 'lif':
            self.single_step_neuron = OnlineLIFNode
        elif self.neuron_type == 'if':
            self.single_step_neuron = OnlineIFNode
        else:
            raise NotImplementedError('Please use IF or LIF model.')
        self.hlop_spiking = hlop_spiking
        self.hlop_spiking_scale = hlop_spiking_scale
        self.hlop_spiking_timesteps = hlop_spiking_timesteps

        self.grad_with_rate = kwargs.get('grad_with_rate', True)
        self.share_classifier = share_classifier
        self.n_hidden = n_hidden
        self.ss = ss
        self.fa = fa
        # TODO not implemented
        assert ss == False
        assert fa == False
        self.hlop_modules = nn.ModuleList([])


        if ss:
            # TODO not implemented
            self.fc1 = SSLinearProj(784, n_hidden, bias=False)
            self.fc2 = SSLinearProj(n_hidden, n_hidden, bias=False)
        elif fa:
            # TODO not implemented
            self.fc1 = FALinearProj(784, n_hidden, bias=False)
            self.fc2 = FALinearProj(n_hidden, n_hidden, bias=False)
        else:
            self.fc1 = LinearProj(784, n_hidden, bias=False)
            #self.fc2 = LinearProj(n_hidden, n_hidden, bias=False)
            self.fc2 = WrapedSNNOp(LinearProj(n_hidden, n_hidden, bias=False))

        self.sn1 = self.single_step_neuron(**kwargs)
        self.sn2 = self.single_step_neuron(**kwargs)

        self.hlop_modules.append(HLOP(784, lr=0.001, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
        self.hlop_modules.append(HLOP(n_hidden, lr=0.01, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
        if share_classifier:
            self.hlop_modules.append(HLOP(n_hidden, lr=0.01, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
            if ss:
                # TODO not implemented
                self.classifiers = nn.ModuleList([SSLinearProj(n_hidden, num_classes, bias=False)])
            elif fa:
                # TODO not implemented
                self.classifiers = nn.ModuleList([FALinearProj(n_hidden, num_classes, bias=False)])
            else:
                self.classifiers = nn.ModuleList([WrapedSNNOp(LinearProj(n_hidden, num_classes, bias=False))])
        else:
            if ss:
                # TODO not implemented
                self.classifiers = nn.ModuleList([SSLinear(n_hidden, num_classes, bias=False)])
            elif fa:
                # TODO not implemented
                self.classifiers = nn.ModuleList([FALinear(n_hidden, num_classes, bias=False)])
            else:
                self.classifiers = nn.ModuleList([WrapedSNNOp(nn.Linear(n_hidden, num_classes, bias=False))])
        self.classifier_num = 1

    def set_hlop_value(self, weight, index=0, **kwargs):
        self.hlop_modules[index].set_subspace(weight, **kwargs)

    def get_hlop_value(self, index=0, **kwargs):
        return self.hlop_modules[index].get_weight_value(**kwargs)

    def forward(self, x, task_id=None, projection=False, proj_id_list=[0], update_hlop=False, fix_subspace_id_list=None, **kwargs):
        require_wrap = self.grad_with_rate and self.training
        x = x.view(-1, 784)
        if projection:
            proj_func = self.hlop_modules[0].get_proj_func(subspace_id_list=proj_id_list)
            x_ = self.fc1(x, projection=True, proj_func=proj_func)
        else:
            x_ = self.fc1(x, projection=False)
        if update_hlop:
            with torch.no_grad():
                self.hlop_modules[0].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)
        if require_wrap:
            x = self.sn1(x_, output_type='spike_rate', **kwargs)
        else:
            x = self.sn1(x_, **kwargs)
        if projection:
            proj_func = self.hlop_modules[1].get_proj_func(subspace_id_list=proj_id_list)
            x_ = self.fc2(x, projection=True, proj_func=proj_func, require_wrap=require_wrap)
        else:
            x_ = self.fc2(x, projection=False, require_wrap=require_wrap)
        if update_hlop:
            with torch.no_grad():
                self.hlop_modules[1].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)
        if require_wrap:
            x = self.sn2(x_, output_type='spike_rate', **kwargs)
        else:
            x = self.sn2(x_, **kwargs)
        if not self.share_classifier:
            assert task_id is not None
            x = self.classifiers[task_id](x, require_wrap=require_wrap)
        else:
            m = self.classifiers[0]
            if projection:
                proj_func = self.hlop_modules[2].get_proj_func(subspace_id_list=proj_id_list)
                x_ = m(x, projection=True, proj_func=proj_func, require_wrap=require_wrap)
            else:
                x_ = m(x, projection=False, require_wrap=require_wrap)
            if update_hlop:
                with torch.no_grad():
                    self.hlop_modules[2].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)
            x = x_

        out = x
        return out

    def add_classifier(self, num_classes):
        self.classifier_num += 1
        if self.ss:
            self.classifiers.append(SSLinear(self.n_hidden, num_classes).to(self.classifiers[0].weight.device))
        elif self.fa:
            self.classifiers.append(FALinear(self.n_hidden, num_classes).to(self.classifiers[0].weight.device))
        else:
            self.classifiers.append(nn.Linear(self.n_hidden, num_classes).to(self.classifiers[0].weight.device))

    def merge_hlop_subspace(self):
        for m in self.hlop_modules:
            m.merge_subspace()

    def add_hlop_subspace(self, out_numbers):
        if isinstance(out_numbers, list):
            for i in range(len(self.hlop_modules)):
                self.hlop_modules[i].add_subspace(out_numbers[i])
        else:
            for m in self.hlop_modules:
                m.add_subspace(out_numbers)

    #def adjust_hlop_lr(self, gamma):
    #    for m in self.hlop_modules:
    #        m.adjust_lr(gamma)

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False


def spiking_MLP_ottt(tau=2., surrogate_function=surrogate.Sigmoid(alpha=4.), track_rate=True, grad_with_rate=True, v_reset=None, **kwargs):
    return spiking_MLP(tau=tau, surrogate_function=surrogate_function, track_rate=track_rate, grad_with_rate=grad_with_rate, v_reset=v_reset, **kwargs)
