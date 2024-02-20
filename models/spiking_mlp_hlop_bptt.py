import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.neuron_spikingjelly import MultiStepIFNode, MultiStepLIFNode
import modules.surrogate as surrogate
from modules.proj_conv import Conv2dProj, SSConv2dProj
from modules.proj_linear import LinearProj, SSLinear, SSLinearProj, FALinear, FALinearProj
from modules.hlop_module import HLOP
import numpy as np


__all__ = [
    'spiking_MLP_bptt'
]


class spiking_MLP(nn.Module):
    def __init__(self, num_classes=10, n_hidden=800, neuron_type='lif', share_classifier=True, ss=False, fa=False, timesteps=6, hlop_spiking=False, hlop_spiking_scale=20., hlop_spiking_timesteps=1000., **kwargs):
        super(spiking_MLP, self).__init__()
        self.neuron_type = neuron_type
        if self.neuron_type == 'lif':
            self.multi_step_neuron = MultiStepLIFNode
        elif self.neuron_type == 'if':
            self.multi_step_neuron = MultiStepIFNode
        else:
            raise NotImplementedError('Please use IF or LIF model.')
        self.timesteps = timesteps
        self.hlop_spiking = hlop_spiking
        self.hlop_spiking_scale = hlop_spiking_scale
        self.hlop_spiking_timesteps = hlop_spiking_timesteps

        self.n_hidden = n_hidden
        self.ss = ss
        self.fa = fa
        self.hlop_modules = nn.ModuleList([])
        self.share_classifier = share_classifier

        if ss:
            self.fc1 = SSLinearProj(784, n_hidden, bias=False)
            self.fc2 = SSLinearProj(n_hidden, n_hidden, bias=False)
        elif fa:
            self.fc1 = FALinearProj(784, n_hidden, bias=False)
            self.fc2 = FALinearProj(n_hidden, n_hidden, bias=False)
        else:
            self.fc1 = LinearProj(784, n_hidden, bias=False)
            self.fc2 = LinearProj(n_hidden, n_hidden, bias=False)

        self.sn1 = self.multi_step_neuron(**kwargs)
        self.sn2 = self.multi_step_neuron(**kwargs)

        self.hlop_modules.append(HLOP(784, lr=0.001, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
        self.hlop_modules.append(HLOP(n_hidden, lr=0.01, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
        if share_classifier:
            self.hlop_modules.append(HLOP(n_hidden, lr=0.01, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
            if ss:
                self.classifiers = nn.ModuleList([SSLinearProj(n_hidden, num_classes, bias=False)])
            elif fa:
                self.classifiers = nn.ModuleList([FALinearProj(n_hidden, num_classes, bias=False)])
            else:
                self.classifiers = nn.ModuleList([LinearProj(n_hidden, num_classes, bias=False)])
        else:
            if ss:
                self.classifiers = nn.ModuleList([SSLinear(n_hidden, num_classes, bias=False)])
            elif fa:
                self.classifiers = nn.ModuleList([FALinear(n_hidden, num_classes, bias=False)])
            else:
                self.classifiers = nn.ModuleList([nn.Linear(n_hidden, num_classes, bias=False)])
        self.classifier_num = 1

    def set_hlop_value(self, weight, index=0, **kwargs):
        self.hlop_modules[index].set_subspace(weight, **kwargs)

    def get_hlop_value(self, index=0, **kwargs):
        return self.hlop_modules[index].get_weight_value(**kwargs)

    def forward(self, x, task_id=None, projection=False, proj_id_list=[0], update_hlop=False, fix_subspace_id_list=None, **kwargs):
        x = x.view(-1, 784)
        B = x.size(0)
        T = self.timesteps
        x.unsqueeze_(0)
        x = x.repeat(T, 1, 1)
        x = x.view(T*B, -1)
        if projection:
            proj_func = self.hlop_modules[0].get_proj_func(subspace_id_list=proj_id_list)
            x_ = self.fc1(x, projection=True, proj_func=proj_func)
        else:
            x_ = self.fc1(x, projection=False)
        if update_hlop:
            with torch.no_grad():
                self.hlop_modules[0].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)
        x = x_.reshape(T, B, -1)
        x = self.sn1(x)
        x = x.view(T*B, -1)
        if projection:
            proj_func = self.hlop_modules[1].get_proj_func(subspace_id_list=proj_id_list)
            x_ = self.fc2(x, projection=True, proj_func=proj_func)
        else:
            x_ = self.fc2(x, projection=False)
        if update_hlop:
            with torch.no_grad():
                self.hlop_modules[1].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)

        x = x_.reshape(T, B, -1)
        x = self.sn2(x)
        x = x.view(T*B, -1)
        if not self.share_classifier:
            assert task_id is not None
            x = self.classifiers[task_id](x)
        else:
            m = self.classifiers[0]
            if projection:
                proj_func = self.hlop_modules[2].get_proj_func(subspace_id_list=proj_id_list)
                x_ = m(x, projection=True, proj_func=proj_func)
            else:
                x_ = m(x, projection=False)
            if update_hlop:
                with torch.no_grad():
                    self.hlop_modules[2].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)
            x = x_

        out = x.reshape(T, B, -1)
        out = torch.mean(out, dim=0)
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


def spiking_MLP_bptt(tau=2., surrogate_function=surrogate.Sigmoid(alpha=4.), v_reset=None, detach_reset=True, decay_input=False, **kwargs):
    return spiking_MLP(tau=tau, surrogate_function=surrogate_function, v_reset=v_reset, detach_reset=detach_reset, decay_input=decay_input, **kwargs)
