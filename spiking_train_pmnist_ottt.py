import datetime
import os
import time
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import sys
from torch.cuda import amp
import models
import argparse
import math
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from copy import deepcopy

_seed_ = 2022
import random
random.seed(_seed_)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

torch.set_num_threads(4)


def test(args, model, x, y, task_id):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=((x.size(0)-1)//args.b+1))

    test_loss = 0
    test_acc = 0
    test_samples = 0
    batch_idx = 0

    r=np.arange(x.size(0))
    with torch.no_grad():
        for i in range(0, len(r), args.b):
            if i + args.b <= len(r):
                index = r[i : i + args.b]
            else:
                index = r[i:]
            batch_idx += 1
            input = x[index].float().cuda()

            label = y[index].cuda()

            loss = 0.
            for t in range(args.timesteps):
                if t == 0:
                    out_fr = model(input, task_id, projection=False, update_hlop=False, init=True)
                    total_fr = out_fr.clone().detach()
                else:
                    out_fr = model(input, task_id, projection=False, update_hlop=False)
                    total_fr += out_fr.clone().detach()
                loss += F.cross_entropy(out_fr, label).detach() / args.timesteps
            out = total_fr
                
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out.argmax(1) == label).float().sum().item()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
            losses.update(loss, input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx,
                        size=((x.size(0)-1)//args.b+1),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
    bar.finish()

    test_loss /= test_samples
    test_acc /= test_samples

    return test_loss, test_acc


def main():

    parser = argparse.ArgumentParser(description='Classify PMNIST')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='./data')
    parser.add_argument('-out_dir', type=str, help='root dir for saving logs and checkpoint', default='./logs')

    parser.add_argument('-opt', type=str, help='use which optimizer. SGD or Adam', default='SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-lr_scheduler', default='StepLR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=200, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-warmup', default=0, type=int, help='warmup epochs for learning rate')
    parser.add_argument('-cnf', type=str)

    parser.add_argument('-hlop_start_epochs', default=0, type=int, help='the start epoch to update hlop')

    parser.add_argument('-sign_symmetric', action='store_true', help='sign symmetric')
    parser.add_argument('-feedback_alignment', action='store_true', help='feedback alignment')

    parser.add_argument('-baseline', action='store_true', help='baseline')

    parser.add_argument('-replay', action='store_true', help='replay few-shot previous tasks')
    parser.add_argument('-memory_size', default=50, type=int, help='memory size for replay')
    parser.add_argument('-replay_epochs', default=1, type=int, help='epochs for replay')
    parser.add_argument('-replay_b', default=50, type=int, help='batch size per task for replay')
    parser.add_argument('-replay_lr', default=0.01, type=float, help='learning rate for replay')
    parser.add_argument('-replay_T_max', default=20, type=int, help='T_max for CosineAnnealingLR for replay')

    parser.add_argument('-gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    # SNN settings
    parser.add_argument('-timesteps', default=6, type=int)
    parser.add_argument('-online_update', action='store_true', help='online update')

    parser.add_argument('-hlop_spiking', action='store_true', help='use hlop with lateral spiking neurons')
    parser.add_argument('-hlop_spiking_scale', default=20., type=float)
    parser.add_argument('-hlop_spiking_timesteps', default=1000., type=float)




    args = parser.parse_args()

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    from dataloader import pmnist as pmd
    data, taskcla, inputsize = pmd.get(data_dir=args.data_dir, seed=_seed_)

    acc_matrix=np.zeros((10,10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    
    hlop_out_num = [80, 200, 100]
    hlop_out_num_inc = [70, 70, 70]

    if args.replay:
        replay_data = {}

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print(out_dir)

    pt_dir = os.path.join(out_dir, 'models')
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)
        print(f'Mkdir {pt_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    for k, ncla in taskcla:
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)

        writer = SummaryWriter(os.path.join(out_dir, 'logs_task{task_id}'.format(task_id=task_id)))

        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        if args.replay:
            # save samples for memory replay
            replay_data[task_id] = {'x': [], 'y': []}
            for c in range(ncla):
                num = args.memory_size
                index = 0
                while num > 0:
                    if ytrain[index] == c:
                        replay_data[task_id]['x'].append(xtrain[index])
                        replay_data[task_id]['y'].append(ytrain[index])
                        num -= 1
                    index += 1
            replay_data[task_id]['x'] = torch.stack(replay_data[task_id]['x'], dim=0)
            replay_data[task_id]['y'] = torch.stack(replay_data[task_id]['y'], dim=0)

        if task_id == 0:
            model = models.spiking_MLP_ottt(num_classes=ncla, n_hidden=800, ss=args.sign_symmetric, fa=args.feedback_alignment, timesteps=args.timesteps, hlop_spiking=args.hlop_spiking, hlop_spiking_scale=args.hlop_spiking_scale, hlop_spiking_timesteps=args.hlop_spiking_timesteps)

            model.add_hlop_subspace(hlop_out_num)
            model = model.cuda()
        else:
            if task_id % 3 == 0:
                hlop_out_num_inc[0] -= 20
                hlop_out_num_inc[1] -= 20
                hlop_out_num_inc[2] -= 20
            model.add_hlop_subspace(hlop_out_num_inc)

        params = []
        for name, p in model.named_parameters():
            if 'hlop' not in name:
                if task_id != 0:
                    if len(p.size()) != 1:
                        params.append(p)
                else:
                    params.append(p)
        if args.opt == 'SGD':
            optimizer = torch.optim.SGD(params, lr=args.lr)
        elif args.opt == 'Adam':
            optimizer = torch.optim.Adam(params, lr=args.lr)
        else:
            raise NotImplementedError(args.opt)

        lr_scheduler = None
        if args.lr_scheduler == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.lr_scheduler == 'CosALR':
            #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
            lr_lambda = lambda cur_epoch: (cur_epoch + 1) / args.warmup if cur_epoch < args.warmup else 0.5 * (1 + math.cos((cur_epoch - args.warmup) / (args.T_max - args.warmup) * math.pi))
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise NotImplementedError(args.lr_scheduler)

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            model.train()
            if task_id != 0:
                model.fix_bn()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            end = time.time()

            bar = Bar('Processing', max=((xtrain.size(0)-1)//args.b+1))

            train_loss = 0
            train_acc = 0
            train_samples = 0
            batch_idx = 0

            r = np.arange(xtrain.size(0))
            np.random.shuffle(r)
            for i in range(0, len(r), args.b):
                if i + args.b <= len(r):
                    index = r[i : i + args.b]
                else:
                    index = r[i:]
                batch_idx += 1
                x = xtrain[index].float().cuda()

                label = ytrain[index].cuda()

                total_loss = 0.
                if not args.online_update:
                    optimizer.zero_grad()
                for t in range(args.timesteps):
                    if args.online_update:
                        optimizer.zero_grad()
                    init = (t == 0)
                    if task_id == 0:
                        if args.baseline:
                            out_fr = model(x, task_id, projection=False, update_hlop=False, init=init)
                        else:
                            if epoch <= args.hlop_start_epochs:
                                out_fr = model(x, task_id, projection=False, update_hlop=False, init=init)
                            else:
                                out_fr = model(x, task_id, projection=False, update_hlop=True, init=init)
                    else:
                        if args.baseline:
                            out_fr = model(x, task_id, projection=False, proj_id_list=[0], update_hlop=False, fix_subspace_id_list=[0], init=init)
                        else:
                            if epoch <= args.hlop_start_epochs:
                                out_fr = model(x, task_id, projection=True, proj_id_list=[0], update_hlop=False, fix_subspace_id_list=[0], init=init)
                            else:
                                out_fr = model(x, task_id, projection=True, proj_id_list=[0], update_hlop=True, fix_subspace_id_list=[0], init=init)
                    if t == 0:
                        total_fr = out_fr.clone().detach()
                    else:
                        total_fr += out_fr.clone().detach()
                    loss = F.cross_entropy(out_fr, label) / args.timesteps
                    loss.backward()
                    total_loss += loss.detach()
                    if args.online_update:
                        optimizer.step()
                if not args.online_update:
                    optimizer.step()

                train_loss += total_loss.item() * label.numel()
                out = total_fr

                # measure accuracy and record loss
                prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                losses.update(loss, x.size(0))
                top1.update(prec1.item(), x.size(0))
                top5.update(prec5.item(), x.size(0))


                train_samples += label.numel()
                train_acc += (out.argmax(1) == label).float().sum().item()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx,
                            size=((xtrain.size(0)-1)//args.b+1),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                            )
                bar.next()
            bar.finish()

            train_loss /= train_samples
            train_acc /= train_samples

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            lr_scheduler.step()

            test_loss, test_acc = test(args, model, xtest, ytest, task_id)

            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)

            total_time = time.time() - start_time
            print(f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')

        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            _, acc_matrix[task_id,jj] = test(args, model, xtest, ytest, ii) 
            jj +=1
        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]*100),end='')
            print()

        model.merge_hlop_subspace()

        # save model
        torch.save(model.state_dict(), os.path.join(pt_dir, 'model_task{task_id}.pth'.format(task_id=task_id)))

        # update task id 
        task_id +=1

    print('-'*50)
    # Simulation Results 
    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()*100)) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt*100))
    print('-'*50)
    # Plots
    #array = acc_matrix
    #df_cm = pd.DataFrame(array, index = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]],
    #                  columns = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]])
    #sn.set(font_scale=1.4) 
    #sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    #plt.show()

if __name__ == '__main__':
    main()
