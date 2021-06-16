#!/usr/bin/env python3

import torch
import numpy as np
import sys
sys.path.insert(1, "/home/zxz147/git_forks/adversarial-robustness-toolbox")
sys.path.insert(1, "/home/zxz147/projects/De-Certification")

from smoothing.architectures import get_architecture
import models.resnet as resnet

def remove_misclassify(data, preds, label): 
    repeat_idx = np.equal(preds if len(preds.shape) == 1 else np.argmax(preds, axis=1), 
                          label if len(label.shape) == 1 else np.argmax(label, axis=1))
    return data[repeat_idx], label[repeat_idx]

def load_classifier(checkpoint_path: str, smoothed: bool = False): 
    base_checkpoint = torch.load(checkpoint_path)
    if smoothed: 
        base_classifier = get_architecture(base_checkpoint["arch"], "cifar10")
    else: 
        base_classifier = torch.nn.DataParallel(resnet.__dict__["resnet110"]())
    base_classifier.load_state_dict(base_checkpoint["state_dict"])
    return base_classifier

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')