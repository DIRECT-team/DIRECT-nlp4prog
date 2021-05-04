import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader, IterableDataset

import pickle
import random
import time
import os
import copy
import time
import math

from PIL import Image

import argparse

from pprint import pprint
from collections import defaultdict as ddict
from collections import OrderedDict

import logging, uuid, sys

import numbers

import tarfile, glob, json

import editdistance
import nltk

class RemapClasses(object):
    def __init__(self, class_remapping):
        self.class_remapping = class_remapping

    def __call__(self, target):
        assert isinstance(target, torch.Tensor)

        for i in self.class_remapping:
            target[target == i] = self.class_remapping[i]

        return target

def get_logger(name, log_dir):
	logger = logging.getLogger(name)
	logFormatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")

	fileHandler = logging.FileHandler("{0}/{1}.log".format(log_dir, name.replace('/', '-')))
	fileHandler.setFormatter(logFormatter)
	logger.addHandler(fileHandler)

	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logFormatter)
	logger.addHandler(consoleHandler)

	logger.setLevel(logging.INFO)

	return logger

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def fast_hist(Y_pred, Y, n):
    assert n == Y_pred.shape[1]
    pred    = torch.argmax(Y_pred, dim=1)
    pred    = pred.cpu().detach().numpy().flatten()
    label   = Y.cpu().detach().numpy().flatten()
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def iou_from_hist(hist):
    ious = 100 * np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    mask = ((hist.sum(1) > 0) & (~np.isnan(ious))) # If a particular class is missing from labels, don't include it
    mIoU = np.nansum(ious * mask) / np.sum(mask)
    return mIoU

def per_image_fast_hist(Y_pred, Y, n):
    pred        = torch.argmax(Y_pred, dim=1)
    pred        = pred.cpu().detach().numpy().reshape(pred.shape[0], -1)
    label       = Y.cpu().detach().numpy().reshape(Y.shape[0], -1)

    k           = (label >= 0) & (label < n)
    label[~k]   = n
    temp        = np.apply_along_axis(np.bincount, 1, n * label.astype(int) + pred, minlength=(n**2) + n)
    temp        = temp[:, : n**2]

    return temp.reshape(pred.shape[0], n, n)

def per_image_iou(Y_pred, Y):
    num_classes = Y_pred.shape[1]

    hist        = per_image_fast_hist(Y_pred, Y, num_classes)
    diag        = np.diagonal(hist, axis1=1, axis2=2)
    ious        = 100 * diag / (hist.sum(2) + hist.sum(1) - diag + 1e-10)
    mask        = ((hist.sum(2) > 0) & (~np.isnan(ious))) # If a particular class is missing from labels, don't include it
    mIoU        = np.nansum(ious * mask, axis=1) / np.sum(mask, axis=1)
    # mIoU        = np.nanmean(ious, axis=1)
    mIoU        = torch.Tensor(mIoU).to(Y_pred.device)

    return mIoU
