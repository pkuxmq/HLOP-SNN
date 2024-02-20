import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.neuron_dsr import LIFNeuron, IFNeuron
from modules.neuron_dsr import rate_spikes, weight_rate_spikes
from modules.proj_conv import Conv2dProj, SSConv2dProj
from modules.proj_linear import LinearProj, SSLinear, SSLinearProj, FALinear, FALinearProj
from modules.hlop_module import HLOP
import numpy as np


__all__ = [
    'spiking_MLP'
]


class spiking_MLP(nn.Module):
    def __init__(self, snn_setting, num_classes=10, n_hidden=800, share_classifier=True, neuron_type='lif', ss=False, fa=False, hlop_with_wfr=True, hlop_spiking=False, hlop_spiking_scale=20., hlop_spiking_timesteps=1000.):
        super(spiking_MLP, self).__init__()
        self.timesteps = snn_setting['timesteps']
        self.snn_setting = snn_setting
        self.neuron_type = neuron_type

        self.share_classifier = share_classifier
        self.n_hidden = n_hidden
        self.ss = ss
        self.fa = fa
        self.hlop_modules = nn.ModuleList([])
        self.hlop_with_wfr = hlop_with_wfr
        self.hlop_spiking = hlop_spiking
        self.hlop_spiking_scale = hlop_spiking_scale
        self.hlop_spiking_timesteps = hlop_spiking_timesteps

        if self.neuron_type == 'lif':
            self.tau = snn_setting['tau']
            self.delta_t = snn_setting['delta_t']
            self.weight_avg = True
        elif self.neuron_type == 'if':
            self.weight_avg = False
        else:
            raise NotImplementedError('Please use IF or LIF model.')

        if ss:
            self.fc1 = SSLinearProj(784, n_hidden, bias=False)
            self.fc2 = SSLinearProj(n_hidden, n_hidden, bias=False)
        elif fa:
            self.fc1 = FALinearProj(784, n_hidden, bias=False)
            self.fc2 = FALinearProj(n_hidden, n_hidden, bias=False)
        else:
            self.fc1 = LinearProj(784, n_hidden, bias=False)
            self.fc2 = LinearProj(n_hidden, n_hidden, bias=False)

        if self.neuron_type == 'lif':
            self.sn1 = LIFNeuron(self.snn_setting)
            self.sn2 = LIFNeuron(self.snn_setting)
        elif self.neuron_type == 'if':
            self.sn1 = IFNeuron(self.snn_setting)
            self.sn2 = IFNeuron(self.snn_setting)
        else:
            raise NotImplementedError('Please use IF or LIF model.')

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

    def forward(self, x, task_id=None, projection=False, proj_id_list=[0], update_hlop=False, fix_subspace_id_list=None):
        x = torch.cat([x[:,_,:,:,:] for _ in range(self.timesteps)], 0)
        x = x.view(-1, 784)
        if projection:
            proj_func = self.hlop_modules[0].get_proj_func(subspace_id_list=proj_id_list)
            x_ = self.fc1(x, projection=True, proj_func=proj_func)
        else:
            x_ = self.fc1(x, projection=False)
        if update_hlop:
            if self.hlop_with_wfr:
                # update hlop by weighted firing rate
                x = weight_rate_spikes(x, self.timesteps, self.tau, self.delta_t)
            with torch.no_grad():
                self.hlop_modules[0].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)
        x = self.sn1(x_)
        if projection:
            proj_func = self.hlop_modules[1].get_proj_func(subspace_id_list=proj_id_list)
            x_ = self.fc2(x, projection=True, proj_func=proj_func)
        else:
            x_ = self.fc2(x, projection=False)
        if update_hlop:
            if self.hlop_with_wfr:
                # update hlop by weighted firing rate
                x = weight_rate_spikes(x, self.timesteps, self.tau, self.delta_t)
            with torch.no_grad():
                self.hlop_modules[1].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)
        x = self.sn2(x_)
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
                if self.hlop_with_wfr:
                    # update hlop by weighted firing rate
                    x = weight_rate_spikes(x, self.timesteps, self.tau, self.delta_t)
                with torch.no_grad():
                    self.hlop_modules[2].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)
            x = x_

        out = x
        if self.weight_avg:
            out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
        else:
            out = rate_spikes(out, self.timesteps)
        return out

    def forward_features(self, x):
        x = torch.cat([x[:,_,:,:,:] for _ in range(self.timesteps)], 0)
        inputs = x.view(-1, 784)
        feature_list = []
        x_ = self.fc1(inputs, projection=False)
        if self.hlop_with_wfr:
            # calculate weighted firing rate
            inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
        feature_list.append(inputs.detach().cpu())
        inputs = self.sn1(x_)
        x_ = self.fc2(inputs, projection=False)
        if self.hlop_with_wfr:
            # calculate weighted firing rate
            inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
        feature_list.append(inputs.detach().cpu())
        inputs = self.sn2(x_)
        if self.share_classifier:
            if self.hlop_with_wfr:
                # calculate weighted firing rate
                inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
            feature_list.append(inputs.detach().cpu())

        return feature_list

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

            if isinstance(m, LIFNeuron) or isinstance(m, IFNeuron):
                if self.snn_setting['train_Vth']:
                    m.Vth.requires_grad = False
