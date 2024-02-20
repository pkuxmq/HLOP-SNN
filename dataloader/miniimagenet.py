from __future__ import print_function
from PIL import Image
import os
import os.path
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import numpy as np

import torch
from torchvision import transforms

from dataloader.utils import *



class MiniImageNet(torch.utils.data.Dataset):

    def __init__(self, root, train):
        super(MiniImageNet, self).__init__()
        if train:
            self.name='train'
        else:
            self.name='test'
        root = os.path.join(root, 'miniimagenet')
        with open(os.path.join(root,'{}.pkl'.format(self.name)), 'rb') as f:
            data_dict = pickle.load(f)

        self.data = data_dict['images']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.labels[i]
        return img, label


class iMiniImageNet(MiniImageNet):

    def __init__(self, root, classes, memory_classes, memory, task_num, train, transform=None):
        super(iMiniImageNet, self).__init__(root=root, train=train)

        self.transform = transform
        if not isinstance(classes, list):
            classes = [classes]

        self.class_mapping = {c: i for i, c in enumerate(classes)}
        self.class_indices = {}

        for cls in classes:
            self.class_indices[self.class_mapping[cls]] = []

        data = []
        labels = []
        tt = []  # task module labels
        td = []  # disctiminator labels

        for i in range(len(self.data)):
            if self.labels[i] in classes:
                data.append(self.data[i])
                labels.append(self.class_mapping[self.labels[i]])
                tt.append(task_num)
                td.append(task_num+1)
                self.class_indices[self.class_mapping[self.labels[i]]].append(i)

        if memory_classes:
            for task_id in range(task_num):
                for i in range(len(memory[task_id]['x'])):
                    if memory[task_id]['y'][i] in range(len(memory_classes[task_id])):
                        data.append(memory[task_id]['x'][i])
                        labels.append(memory[task_id]['y'][i])
                        tt.append(memory[task_id]['tt'][i])
                        td.append(memory[task_id]['td'][i])

        self.data = np.array(data)
        self.labels = labels
        self.tt = tt
        self.td = td



    def __getitem__(self, index):

        img, target, tt, td = self.data[index], self.labels[index], self.tt[index], self.td[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not torch.is_tensor(img):
            img = Image.fromarray(img)
            img = self.transform(img)
        # return img, target, tt, td
        return img, target





    def __len__(self):
        return len(self.data)




class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, data_dir, seed):
        super(DatasetGen, self).__init__()

        self.seed = seed
        self.batch_size=64
        self.root = data_dir
        self.use_memory = 'yes'

        self.num_tasks = 20
        self.num_classes = 100

        self.num_samples = 0

        self.inputsize = [3,84,84]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transformation = transforms.Compose([
                                    transforms.Resize((84,84)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])

        self.taskcla = [[t, int(self.num_classes/self.num_tasks)] for t in range(self.num_tasks)]

        self.indices = {}
        self.dataloaders = {}
        self.idx={}

        self.num_workers = 4
        self.pin_memory = True

        np.random.seed(self.seed)
        task_ids = np.split(np.random.permutation(self.num_classes),self.num_tasks)
        self.task_ids = [list(arr) for arr in task_ids]

        self.train_set = {}
        self.train_split = {}
        self.test_set = {}


        self.task_memory = {}
        for i in range(self.num_tasks):
            self.task_memory[i] = {}
            self.task_memory[i]['x'] = []
            self.task_memory[i]['y'] = []
            self.task_memory[i]['tt'] = []
            self.task_memory[i]['td'] = []



    def get(self, task_id):

        self.dataloaders[task_id] = {}

        sys.stdout.flush()

        if task_id == 0:
            memory_classes = None
            memory=None
        else:
            memory_classes = self.task_ids
            memory = self.task_memory


        self.train_set[task_id] = iMiniImageNet(root=self.root, classes=self.task_ids[task_id],
                                                memory_classes=memory_classes, memory=memory,
                                                task_num=task_id, train=True, transform=self.transformation)

        self.test_set[task_id] = iMiniImageNet(root=self.root, classes=self.task_ids[task_id], memory_classes=None,
                                        memory=None, task_num=task_id, train=False, transform=self.transformation)


        train_loader = torch.utils.data.DataLoader(self.train_set[task_id], batch_size=2500, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=500, num_workers=self.num_workers,
                                                  pin_memory=self.pin_memory, shuffle=True)

        self.dataloaders[task_id]['train'] = {'x': [],'y': []}
        for data, label in train_loader:
            self.dataloaders[task_id]['train']['x'] = data
            self.dataloaders[task_id]['train']['y'] = label

        self.dataloaders[task_id]['test'] = {'x': [],'y': []}
        for data, label in test_loader:
            self.dataloaders[task_id]['test']['x'] = data 
            self.dataloaders[task_id]['test']['y']= label


        self.dataloaders[task_id]['name'] = 'iMiniImageNet-{}-{}'.format(task_id,self.task_ids[task_id])

        print ("Task ID: ", task_id)
        print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        if self.use_memory == 'yes' and self.num_samples > 0 :
            self.update_memory(task_id)


        return self.dataloaders



    def update_memory(self, task_id):
        num_samples_per_class = self.num_samples // len(self.task_ids[task_id])
        mem_class_mapping = {i: i for i, c in enumerate(self.task_ids[task_id])}

        for i in range(len(self.task_ids[task_id])):
            data_loader = torch.utils.data.DataLoader(self.train_split[task_id], batch_size=1,
                                                        num_workers=self.num_workers,
                                                        pin_memory=self.pin_memory)

            randind = torch.randperm(len(data_loader.dataset))[:num_samples_per_class]  # randomly sample some data


            for ind in randind:
                self.task_memory[task_id]['x'].append(data_loader.dataset[ind][0])
                self.task_memory[task_id]['y'].append(mem_class_mapping[i])
                self.task_memory[task_id]['tt'].append(data_loader.dataset[ind][2])
                self.task_memory[task_id]['td'].append(data_loader.dataset[ind][3])

        print ('Memory updated by adding {} images'.format(len(self.task_memory[task_id]['x'])))
