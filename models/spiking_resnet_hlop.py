import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.neuron_dsr import LIFNeuron, IFNeuron
from modules.neuron_dsr import rate_spikes, weight_rate_spikes
from modules.proj_conv import Conv2dProj, SSConv2dProj
from modules.proj_linear import LinearProj, SSLinear, SSLinearProj
from modules.hlop_module import HLOP
import numpy as np
from typing import Callable, Optional, List


__all__ = [
    'SpikingResNet',
    'spiking_resnet18',
]


def update_conv_hlop(x: torch.Tensor, hlop_module, iteration=5, fix_subspace_id_list: Optional[List[int]] = None, kernel_size: int = 3, padding: int = 1, stride: int = 1):
    with torch.no_grad():
        x = F.unfold(x, kernel_size, padding=padding, stride=stride).transpose(1, 2)
        x = x.reshape(-1, x.shape[2])
        hlop_module.forward_with_update(x, iteration=iteration, fix_subspace_id_list=fix_subspace_id_list)


class BasicBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, snn_setting, neuron_type='lif', ss=False, hlop_with_wfr=True, hlop_spiking=False, hlop_spiking_scale=20., hlop_spiking_timesteps=1000., proj_type='input'):
        super(BasicBlock, self).__init__()

        self.timesteps = snn_setting['timesteps']
        self.snn_setting = snn_setting
        self.neuron_type = neuron_type

        self.ss = ss
        self.hlop_with_wfr = hlop_with_wfr
        self.hlop_spiking = hlop_spiking
        self.hlop_spiking_scale = hlop_spiking_scale
        self.hlop_spiking_timesteps = hlop_spiking_timesteps
        hlop_modules = []
        self.bn1 = nn.BatchNorm2d(in_channels)

        if self.ss:
            self.conv1 = SSConv2dProj(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, proj_type=proj_type)
        else:
            self.conv1 = Conv2dProj(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, proj_type=proj_type)
        hlop_modules.append(HLOP(in_channels*3*3, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
        self.bn2 = nn.BatchNorm2d(out_channels)

        if self.ss:
            self.conv2 = SSConv2dProj(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1, bias=False, proj_type=proj_type)
        else:
            self.conv2 = Conv2dProj(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1, bias=False, proj_type=proj_type)
        hlop_modules.append(HLOP(out_channels*3*3, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))

        self.stride = stride
        if stride != 1 or in_channels != self.expansion * out_channels:
            if self.ss:
                self.downsample = SSConv2dProj(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0, proj_type=proj_type)
            else:
                self.downsample = Conv2dProj(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0, proj_type=proj_type)
            hlop_modules.append(HLOP(in_channels*1*1, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
        else:
            self.downsample = None

        if neuron_type == 'lif':
            self.sn1 = LIFNeuron(snn_setting)
            self.sn2 = LIFNeuron(snn_setting)
            self.tau = snn_setting['tau']
            self.delta_t = snn_setting['delta_t']
            self.weight_avg = True
        elif neuron_type == 'if':
            self.sn1 = IFNeuron(snn_setting)
            self.sn2 = IFNeuron(snn_setting)
            self.weight_avg = False
        else:
            raise NotImplementedError('Please use IF or LIF model.')

        self.hlop_modules = nn.ModuleList(hlop_modules)

    def forward(self, x, task_id=None, projection=False, proj_id_list=[0], update_hlop=False, fix_subspace_id_list=None, hlop_iteration=5):

        out = self.sn1(self.bn1(x))

        if self.downsample is None:
            identity = x
        else:
            if projection:
                proj_func = self.hlop_modules[2].get_proj_func(subspace_id_list=proj_id_list)
                identity = self.downsample(out, projection=True, proj_func=proj_func)
            else:
                identity = self.downsample(out, projection=False)
            if update_hlop:
                if self.hlop_with_wfr:
                    # update hlop by weighted firing rate
                    out_ = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
                else:
                    out_ = out
                update_conv_hlop(out_, self.hlop_modules[2], hlop_iteration, fix_subspace_id_list, kernel_size=1, padding=0, stride=self.stride)

        if projection:
            proj_func = self.hlop_modules[0].get_proj_func(subspace_id_list=proj_id_list)
            out_ = self.conv1(out, projection=True, proj_func=proj_func)
        else:
            out_ = self.conv1(out, projection=False)

        if update_hlop:
            if self.hlop_with_wfr:
                # update hlop by weighted firing rate
                out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
            update_conv_hlop(out, self.hlop_modules[0], hlop_iteration, fix_subspace_id_list, kernel_size=3, padding=1, stride=1)
        out = out_
        out = self.sn2(self.bn2(out))
        if projection:
            proj_func = self.hlop_modules[1].get_proj_func(subspace_id_list=proj_id_list)
            out_ = self.conv2(out, projection=True, proj_func=proj_func)
        else:
            out_ = self.conv2(out, projection=False)
        if update_hlop:
            if self.hlop_with_wfr:
                # update hlop by weighted firing rate
                out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
            update_conv_hlop(out, self.hlop_modules[1], hlop_iteration, fix_subspace_id_list, kernel_size=3, padding=1, stride=1)
        out = out_

        out = out + identity

        return out


    def forward_with_features(self, x):
        features = []
        out = self.sn1(self.bn1(x))

        if self.downsample is None:
            identity = x
        else:
            identity = self.downsample(out, projection=False)
            if self.hlop_with_wfr:
                # calculate weighted firing rate
                out_ = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
            else:
                out_ = out
            out_ = F.unfold(out_, 1, padding=0, stride=self.stride).transpose(1, 2)
            out_ = out_.reshape(-1, out_.shape[2])
            features.append(out_.detach().cpu())

        out_ = self.conv1(out, projection=False)
        if self.hlop_with_wfr:
            # calculate weighted firing rate
            out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
        out = F.unfold(out, 3, padding=1, stride=1).transpose(1, 2)
        out = out.reshape(-1, out.shape[2])
        features.append(out.detach().cpu())
        out = out_
        out = self.sn2(self.bn2(out))
        out_ = self.conv2(out, projection=False)
        if self.hlop_with_wfr:
            # calculate weighted firing rate
            out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
        out = F.unfold(out, 3, padding=1, stride=1).transpose(1, 2)
        out = out.reshape(-1, out.shape[2])
        features.append(out.detach().cpu())
        out = out_

        out = out + identity
        return out, features


    def add_hlop_subspace(self, out_numbers):
        if isinstance(out_numbers, list):
            for i in range(len(self.hlop_modules)):
                self.hlop_modules[i].add_subspace(out_numbers[i])
        else:
            for m in self.hlop_modules:
                m.add_subspace(out_numbers)

    def adjust_hlop_lr(self, gamma):
        for m in self.hlop_modules:
            m.adjust_lr(gamma)


class SpikingResNet(nn.Module):

    def __init__(self, block, num_blocks, snn_setting, neuron_type='lif', nf=64, num_classes=10, share_classifier=False, ss=False, hlop_with_wfr=True, hlop_spiking=False, hlop_spiking_scale=20., hlop_spiking_timesteps=1000., proj_type='input', first_conv_stride2=False):
        super(SpikingResNet, self).__init__()
        self.neuron_type = neuron_type
        self.snn_setting = snn_setting
        self.timesteps = snn_setting['timesteps']
        self.num_blocks = num_blocks
        self.hlop_modules = []
        self.ss = ss
        self.hlop_with_wfr = hlop_with_wfr
        self.hlop_spiking = hlop_spiking
        self.hlop_spiking_scale = hlop_spiking_scale
        self.hlop_spiking_timesteps = hlop_spiking_timesteps
        # choice for projection type in bottom implementation
        # it is theoretically equivalent for input and weight, while weight enables acceleration of convolutional operations
        self.proj_type = proj_type

        self.in_planes = nf
        self.first_conv_stride2 = first_conv_stride2
        if first_conv_stride2:
            if self.ss:
                self.conv1 = SSConv2dProj(3, nf, kernel_size=5, stride=2, padding=2, bias=False, proj_type=proj_type)
            else:
                self.conv1 = Conv2dProj(3, nf, kernel_size=5, stride=2, padding=2, bias=False, proj_type=proj_type)
            self.hlop_modules.append(HLOP(3*5*5, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
        else:
            if self.ss:
                self.conv1 = SSConv2dProj(3, nf, kernel_size=3, stride=1, padding=1, bias=False, proj_type=proj_type)
            else:
                self.conv1 = Conv2dProj(3, nf, kernel_size=3, stride=1, padding=1, bias=False, proj_type=proj_type)
            self.hlop_modules.append(HLOP(3*3*3, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
        self.layers = []
        self.layers += self._make_layer(block, nf, num_blocks[0], 1, snn_setting)
        self.layers += self._make_layer(block, nf*2, num_blocks[1], 2, snn_setting)
        self.layers += self._make_layer(block, nf*4, num_blocks[2], 2, snn_setting)
        self.layers += self._make_layer(block, nf*8, num_blocks[3], 2, snn_setting)
        self.layers = nn.ModuleList(self.layers)

        self.bn1 = nn.BatchNorm2d(nf*8*block.expansion)
        if neuron_type == 'lif':
            self.sn1 = LIFNeuron(snn_setting)
            self.weight_avg = True
            self.tau = snn_setting['tau']
            self.delta_t = snn_setting['delta_t']
        elif neuron_type == 'if':
            self.sn1 = IFNeuron(snn_setting)
            self.weight_avg = False
        else:
            raise NotImplementedError('Please use IF or LIF model.')
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        fc_size = nf*8*block.expansion
        self.fc_size = fc_size
        self.share_classifier = share_classifier
        if share_classifier:
            self.hlop_modules.append(HLOP(fc_size, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
            if self.ss:
                self.classifiers = nn.ModuleList([SSLinearProj(fc_size, num_classes, bias=False)])
            else:
                self.classifiers = nn.ModuleList([LinearProj(fc_size, num_classes, bias=False)])
        else:
            if self.ss:
                self.classifiers = nn.ModuleList([SSLinear(fc_size, num_classes)])
            else:
                self.classifiers = nn.ModuleList([nn.Linear(fc_size, num_classes)])
        self.classifier_num = 1
        self.hlop_modules = nn.ModuleList(self.hlop_modules)
        self._initialize_weights()


    def _make_layer(self, block, planes, num_blocks, stride, snn_setting):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, snn_setting, self.neuron_type, self.ss, self.hlop_with_wfr, self.hlop_spiking, self.hlop_spiking_scale, self.hlop_spiking_timesteps, self.proj_type))
            self.in_planes = planes * block.expansion
        return layers


    def forward(self, x, task_id=None, projection=False, proj_id_list=[0], update_hlop=False, fix_subspace_id_list=None, hlop_iteration=5):

        x = torch.cat([x[:,_,:,:,:] for _ in range(self.timesteps)], 0)
        if projection:
            proj_func = self.hlop_modules[0].get_proj_func(subspace_id_list=proj_id_list)
            out = self.conv1(x, projection=True, proj_func=proj_func)
        else:
            out = self.conv1(x, projection=False)
        if update_hlop:
            if self.hlop_with_wfr:
                # update hlop by weighted firing rate
                x = weight_rate_spikes(x, self.timesteps, self.tau, self.delta_t)
            if self.first_conv_stride2:
                update_conv_hlop(x, self.hlop_modules[0], hlop_iteration, fix_subspace_id_list, kernel_size=5, padding=2, stride=2)
            else:
                update_conv_hlop(x, self.hlop_modules[0], hlop_iteration, fix_subspace_id_list, kernel_size=3, padding=1, stride=1)

        for block in self.layers:
            out = block(out, task_id, projection, proj_id_list, update_hlop, fix_subspace_id_list, hlop_iteration)

        out = self.sn1(self.bn1(out))

        out = self.pool(out)
        out = out.view(out.size(0), -1)

        if not self.share_classifier:
            assert task_id is not None
            out = self.classifiers[task_id](out)
        else:
            m = self.classifiers[0]
            if projection:
                proj_func = self.hlop_modules[index].get_proj_func(subspace_id_list=proj_id_list)
                out_ = m(out, projection=True, proj_func = proj_func)
            else:
                out_ = m(out, projection=False)
            if update_hlop:
                if self.hlop_with_wfr:
                    # update hlop by weighted firing rate
                    out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
                with torch.no_grad():
                    self.hlop_modules[index].forward_with_update(out, hlop_iteration, fix_subspace_id_list)
            out = out_
        
        if self.weight_avg:
            out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
        else:
            out = rate_spikes(out, self.timesteps)

        return out

    def forward_features(self, x):
        inputs = torch.cat([x[:,_,:,:,:] for _ in range(self.timesteps)], 0)
        feature_list = []
        x_ = self.conv1(inputs, projection=False)
        if self.hlop_with_wfr:
            # calculate weighted firing rate
            inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
        if self.first_conv_stride2:
            inputs = F.unfold(inputs, 5, padding=2, stride=2).transpose(1, 2)
        else:
            inputs = F.unfold(inputs, 3, padding=1, stride=1).transpose(1, 2)
        inputs = inputs.reshape(-1, inputs.shape[2])
        feature_list.append(inputs.detach().cpu())
        inputs = x_
        
        for block in self.layers:
            inputs, features = block.forward_with_features(inputs)
            feature_list += features

        inputs = self.sn1(self.bn1(inputs))

        if self.share_classifier:
            inputs = self.pool(inputs)
            if self.hlop_with_wfr:
                # calculate weighted firing rate
                inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
            inputs = inputs.view(inputs.size(0), -1)
            feature_list.append(inputs.detach().cpu())

        return feature_list

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2dProj):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or isinstance(m, LinearProj):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def add_classifier(self, num_classes):
        self.classifier_num += 1
        if self.ss:
            self.classifiers.append(SSLinear(self.fc_size, num_classes).to(self.classifiers[0].weight.device))
        else:
            self.classifiers.append(nn.Linear(self.fc_size, num_classes).to(self.classifiers[0].weight.device))
        m = self.classifiers[-1]
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

    def merge_hlop_subspace(self):
        for m in self.hlop_modules:
            m.merge_subspace()
        for block in self.layers:
            for m in block.hlop_modules:
                m.merge_subspace()

    def add_hlop_subspace(self, out_numbers):
        if isinstance(out_numbers, list):
            self.hlop_modules[0].add_subspace(out_numbers[0])
            for i in range(len(self.layers)):
                self.layers[i].add_hlop_subspace(out_numbers[i+1])
            if self.share_classifier:
                self.hlop_modules[1].add_subspace(out_numbers[-1])
        else:
            for m in self.hlop_modules:
                m.add_subspace(out_numbers)
            for block in self.layers:
                for m in block.hlop_modules:
                    m.add_subspace(out_numbers)

    def adjust_hlop_lr(self, gamma):
        for m in self.hlop_modules:
            m.adjust_lr(gamma)
        for block in self.layers:
            block.adjust_hlop_lr(gamma)

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

            if isinstance(m, LIFNeuron) or isinstance(m, IFNeuron):
                if self.snn_setting['train_Vth']:
                    m.Vth.requires_grad = False


def spiking_resnet18(snn_setting, nf=32, **kwargs):
    model = SpikingResNet(BasicBlock, [2, 2, 2, 2], snn_setting, nf=nf, **kwargs)
    return model

