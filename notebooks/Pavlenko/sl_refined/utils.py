import os
import random
import math
import numpy as np
from scipy.special import logsumexp

import torch
import torchvision
from PIL import Image
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def kNN(net, trainloader, testloader, K, sigma=0.1, dim=128,use_pca=False):
    net.eval()
    # this part is ugly but made to be backwards-compatible. there was a change in cifar dataset's structure.
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]) # .cuda()
    elif hasattr(trainloader.dataset, 'indices'):
        trainLabels = torch.LongTensor([k for path,k in trainloader.dataset.dataset.dt.imgs])[trainloader.dataset.indices]
    elif hasattr(trainloader.dataset, 'train_labels'):
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels)  # .cuda()
    if hasattr(trainloader.dataset, 'dt'):
        if hasattr(trainloader.dataset.dt, 'targets'):
            trainLabels = torch.LongTensor(trainloader.dataset.dt.targets) # .cuda()
        else: #  hasattr(trainloader.dataset.dt, 'imgs'):
            trainLabels = torch.LongTensor([k for path,k in trainloader.dataset.dt.imgs]) # .cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.targets) # .cuda()
    C = trainLabels.max() + 1

    if hasattr(trainloader.dataset, 'transform'):
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
    elif hasattr(trainloader.dataset.dataset.dt, 'transform'):
        transform_bak = trainloader.dataset.dataset.dt.transform
        trainloader.dataset.dataset.dt.transform = testloader.dataset.dt.transform
    else:
        transform_bak = trainloader.dataset.dt.transform
        trainloader.dataset.dt.transform = testloader.dataset.dt.transform

    temploader = torch.utils.data.DataLoader(trainloader.dataset,
                                             batch_size=64, num_workers=1)
    if hasattr(trainloader.dataset, 'indices'):
        LEN = len(trainloader.dataset.indices)
    else:
        LEN = len(trainloader.dataset)
    trainFeatures = torch.zeros((dim, LEN))  # , device='cuda:0')
    normalize = Normalize()
    for batch_idx, (inputs, targets, _) in enumerate(temploader):
        batchSize = inputs.size(0)
        inputs = inputs.cuda()
        features = net(inputs)
        if not use_pca:
            features = normalize(features)
        trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t().cpu()
    if hasattr(temploader.dataset, 'imgs'):
        trainLabels = torch.LongTensor(temploader.dataset.train_labels) # .cuda()
    elif hasattr(temploader.dataset, 'indices'):
        trainLabels = torch.LongTensor([k for path,k in temploader.dataset.dataset.dt.imgs])[temploader.dataset.indices]
    elif hasattr(temploader.dataset, 'train_labels'):
        trainLabels = torch.LongTensor(temploader.dataset.train_labels) # .cuda()
    elif hasattr(temploader.dataset, 'targets'):
        trainLabels = torch.LongTensor(temploader.dataset.targets) # .cuda()
    elif hasattr(temploader.dataset.dt, 'imgs'):
        trainLabels = torch.LongTensor([k for path,k in temploader.dataset.dt.imgs]) #.cuda()
    elif hasattr(temploader.dataset.dt, 'targets'):
        trainLabels = torch.LongTensor(temploader.dataset.dt.targets) #.cuda()
    else:
        trainLabels = torch.LongTensor(temploader.dataset.labels) #.cuda()
    trainLabels = trainLabels.cpu()
    if hasattr(trainloader.dataset, 'transform'):
        trainloader.dataset.transform = transform_bak
    elif hasattr(trainloader.dataset, 'indices'):
        trainloader.dataset.dataset.dt.transform = transform_bak
    else:
        trainloader.dataset.dt.transform = transform_bak

    if use_pca:
        comps = 128
        print('doing PCA with %s components'%comps, end=' ')
        from sklearn.decomposition import PCA
        pca = PCA(n_components=comps, whiten=False)
        trainFeatures = pca.fit_transform(trainFeatures.numpy().T)
        trainFeatures = torch.Tensor(trainFeatures)
        trainFeatures = normalize(trainFeatures).t()
        print('..done')
    def eval_k_s(K_,sigma_):
        total = 0
        top1 = 0.
        top5 = 0.

        with torch.no_grad():
            retrieval_one_hot = torch.zeros(K_, C)# .cuda()
            for batch_idx, (inputs, targets, _) in enumerate(testloader):
                targets = targets # .cuda(async=True) # or without async for py3.7
                inputs = inputs.cuda()
                batchSize = inputs.size(0)
                features = net(inputs)
                if use_pca:
                    features = pca.transform(features.cpu().numpy())
                    features = torch.Tensor(features).cuda()
                features = normalize(features).cpu()

                dist = torch.mm(features, trainFeatures)


                yd, yi = dist.topk(K_, dim=1, largest=True, sorted=True)
                candidates = trainLabels.view(1, -1).expand(batchSize, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batchSize * K_, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(sigma_).exp_()
                probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C),
                                            yd_transform.view(batchSize, -1, 1)),
                                  1)
                _, predictions = probs.sort(1, True)

                # Find which predictions match the target
                correct = predictions.eq(targets.data.view(-1, 1))

                top1 = top1 + correct.narrow(1, 0, 1).sum().item()
                top5 = top5 + correct.narrow(1, 0, 5).sum().item()

                total += targets.size(0)

        print(f"{K_}-NN,s={sigma_}: TOP1: ", top1 * 100. / total)
        return top1 / total

    if isinstance(K, list):
        res = []
        for K_ in K:
            for sigma_ in sigma:
                res.append(eval_k_s(K_, sigma_))
        return res
    else:
        res = eval_k_s(K, sigma)
        return res
    
    
def py_softmax(x, axis=None):
    """stable softmax"""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
