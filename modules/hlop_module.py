import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# module for hebbian learning based orthogonal projection
class HLOP(nn.Module):

    def __init__(self, in_features, lr=0.01, momentum=True, spiking=False, spiking_scale=20., spiking_timesteps=1000.):
        super(HLOP, self).__init__()
        self.in_features = in_features

        self.subspace_num = 0
        self.out_numbers_list = []
        self.index_list = [0]
        self.weight = None
        self.momentum = momentum
        if self.momentum:
            self.delta_weight_momentum = None
            self.m = 0.9
        self.spiking = spiking
        self.spiking_scale = spiking_scale
        self.spiking_timesteps = spiking_timesteps

        self.lr = lr

    def add_subspace(self, out_numbers):
        if out_numbers > 0:
            self.subspace_num += 1
            self.out_numbers_list.append(out_numbers)
            self.index_list.append(self.index_list[-1] + out_numbers)

            if self.subspace_num == 1:
                self.weight = nn.Parameter(torch.zeros((out_numbers, self.in_features)))
                if self.momentum:
                    self.delta_weight_momentum = nn.Parameter(torch.zeros((out_numbers, self.in_features)))

                # initialize weights
                torch.nn.init.orthogonal_(self.weight.data)
                #torch.nn.init.xavier_normal_(self.weight.data)
                
            else:
                dim = self.weight.size(0) + out_numbers
                weight_new = torch.zeros((dim, self.in_features)).to(self.weight.device)
                if self.momentum:
                    delta_weight_momentum_new = torch.zeros((dim, self.in_features)).to(self.weight.device)

                weight_new[:self.weight.size(0), :] = self.weight.data
                if self.momentum:
                    delta_weight_momentum_new[:self.weight.size(0), :] = self.delta_weight_momentum.data

                # initialize new weights
                torch.nn.init.orthogonal_(weight_new[self.weight.size(0):, :])

                self.weight = nn.Parameter(weight_new)
                if self.momentum:
                    self.delta_weight_momentum = nn.Parameter(delta_weight_momentum_new)

    def merge_subspace(self):
        assert self.subspace_num > 0
        self.subspace_num = 1
        self.out_numbers_list = [self.index_list[-1]]
        self.index_list = [0, self.out_numbers_list[0]]

    def update_weights(self, x, y, xhat, fix_subspace_id_list=None):
        # x: B*N, y: B*M, weight: M*N
        weight = self.weight.data
        if self.momentum:
            delta_weight_momentum = self.delta_weight_momentum.data
        
        m, n = weight.size()
        assert n == x.size(1) and m == y.size(1)

        fix_index = []
        if fix_subspace_id_list != None:
            for sid in fix_subspace_id_list:
                fix_index.extend(range(self.index_list[sid], self.index_list[sid + 1]))

        delta_weight = (torch.mm(y.t(), x - xhat) / x.shape[0])
        delta_weight = torch.clamp(delta_weight, -10, 10)
        delta_weight[fix_index, :] = 0.
        lr = self.lr
        if self.momentum:
            fix_term = delta_weight_momentum[fix_index, :]
            delta_weight_momentum[fix_index, :] = 0
            delta_weight_momentum = self.m * delta_weight_momentum + (1 - self.m) * delta_weight
            weight = weight + lr * delta_weight_momentum
            delta_weight_momentum[fix_index, :] = fix_term
        else:
            weight = weight + lr * delta_weight

        self.weight.data = weight
        if self.momentum:
            self.delta_weight_momentum.data = delta_weight_momentum

    def set_subspace(self, weight, id_list=[0]):
        index = []
        for i in id_list:
            index.extend(range(self.index_list[i], self.index_list[i + 1]))
        self.weight.data[index, :] = weight.clone()

    def get_weight_value(self, id_list=[0]):
        index = []
        for i in id_list:
            index.extend(range(self.index_list[i], self.index_list[i + 1]))
        weight_ = self.weight.data[index, :].clone()
        return weight_

    def inference(self, x, subspace_id_list=[0]):
        index = []
        for sid in subspace_id_list:
            index.extend(range(self.index_list[sid], self.index_list[sid + 1]))

        weight = self.weight.data[index, :]

        y0 = torch.mm(x, weight.t())
        y = y0

        if self.spiking:
            y = (torch.clamp(y, -self.spiking_scale, self.spiking_scale) / self.spiking_scale * self.spiking_timesteps).round() / self.spiking_timesteps * self.spiking_scale

        return y

    def inference_back(self, y, subspace_id_list=[0]):
        index = []
        for sid in subspace_id_list:
            index.extend(range(self.index_list[sid], self.index_list[sid + 1]))

        weight = self.weight.data[index, :]

        x = torch.mm(y, weight)

        return x

    def projection(self, x, subspace_id_list=[0]):
        y = self.inference(x, subspace_id_list)
        x_proj = self.inference_back(y, subspace_id_list)

        return x_proj

    def forward_with_update(self, x, iteration=5, fix_subspace_id_list=None):
        subspace_id_list = list(range(self.subspace_num))
        for i in range(iteration):
            y = self.inference(x, subspace_id_list)
            xhat = self.inference_back(y, subspace_id_list)
            self.update_weights(x, y, xhat, fix_subspace_id_list)

    def projection_with_update(self, x, iteration=5, subspace_id_list=[0], fix_subspace_id_list=None):
        x_proj = self.projection(x, subspace_id_list)
        self.forward_with_update(x, iteration, fix_subspace_id_list)

        return x_proj

    def get_proj_func(self, iteration=5, subspace_id_list=[0], forward_with_update=False, fix_subspace_id_list=None):
        if forward_with_update:
            return lambda x: self.projection_with_update(x, iteration, subspace_id_list, fix_subspace_id_list)
        else:
            return lambda x: self.projection(x, subspace_id_list)

    def adjust_lr(self, gamma):
        self.lr = self.lr * gamma


