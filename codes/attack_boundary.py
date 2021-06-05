from __future__ import absolute_import, division, print_function, unicode_literals
from logging import error

import os 
import sys
sys.path.insert(1, "/home/zxz147/git_forks/adversarial-robustness-toolbox")
sys.path.insert(1, "/home/zxz147/projects/De-Certification")
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torch.backends.cudnn as cudnn
from torchvision.datasets import CIFAR10
from torchvision import transforms

from art.attacks.evasion.boundary_orig import BoundaryAttackOrig
from art.attacks.evasion.boundary_cert import BoundaryAttackCert
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10, to_categorical

from smoothing.architectures import get_architecture
import models.resnet as resnet


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def imshow(img):
    img = img / 2 + 0.5   # unnormalize
    # npimg = img   # convert from tensor
    # plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    # plt.show()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    

def load_classifier(checkpoint_path: str, smoothed: bool = False): 
    base_checkpoint = torch.load(checkpoint_path)
    if smoothed: 
        base_classifier = get_architecture(base_checkpoint["arch"], "cifar10")
    else: 
        base_classifier = torch.nn.DataParallel(resnet.__dict__["resnet110"]())
    base_classifier.load_state_dict(base_checkpoint["state_dict"])
    return base_classifier


def attack(model_path, smooth_N, smooth_N0, failure_prob, targeted=False, model_architecture="cifar_resnet110", dataset="cifar10", report_acc=True, smoothed=True, attack_method="cert"): 
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
    base_classifier = load_classifier(model_path, smoothed)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_classifier.parameters(), lr=0.01)
    pytorch_classifier = PyTorchClassifier(
        model=base_classifier, 
        clip_values=(min_pixel_value, max_pixel_value), 
        loss=criterion, 
        optimizer=optimizer, 
        input_shape=(3, 32, 32), 
        nb_classes=10)
    
    if report_acc: 
        predictions = pytorch_classifier.predict(x_test, training_mode=False)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    
    # Obtaining the init (cat) and target (dog) images. 
    img_init, img_targ = None, None
    imgs_found = [False, False]

    for img, label in zip(x_test, y_test):
        if np.argmax(label) == 1 and not imgs_found[0]: 
            img_init = img
            imgs_found[0] = True
        if np.argmax(label) == 5 and not imgs_found[1]: 
            img_targ = img
            imgs_found[1] = True
    
    print("Init image (cat): {}".format(np.argmax(pytorch_classifier.predict(np.expand_dims(img_init, 0)))))
    print("Target image (dog): {}".format(np.argmax(pytorch_classifier.predict(np.expand_dims(img_targ, 0)))))
    
    # exit()
    
    if attack_method == "cert": 
        attack = BoundaryAttackCert(estimator=pytorch_classifier, targeted=targeted, max_iter=0, delta=0.001, epsilon=0.001, smooth_N=smooth_N, smooth_N0=smooth_N0, smooth_alpha=failure_prob)
    elif attack_method == "orig": 
        attack = BoundaryAttackOrig(estimator=pytorch_classifier, targeted=targeted, max_iter=0, delta=0.001, epsilon=0.001)
    else: 
        error("Invalid attack_method parameters")
        exit()
    
    iter_step = 30
    x_adv = np.array([img_init[..., ::-1]])

    for i in range(5):
        x_adv = attack.generate(x=torch.tensor([img_init[..., ::-1]]), y=to_categorical([1], 10), x_adv_init=x_adv)
        #clear_output()    
        print("Adversarial image at step %d." % (i * iter_step), "L2 error", 
              np.linalg.norm(np.reshape(x_adv[0] - img_targ[..., ::-1], [-1])),
              "and class label %d." % np.argmax(pytorch_classifier.predict(x_adv)[0]))
        # plt.imshow(x_adv[0][..., ::-1].astype(np.uint))
        # plt.show(block=False)

        if hasattr(attack, 'curr_delta') and hasattr(attack, 'curr_epsilon'):
            attack.max_iter = iter_step 
            attack.delta = attack.curr_delta
            attack.epsilon = attack.curr_epsilon
        else:
            break
    
    
        
        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", dest="checkpoint_path", type=str)
    parser.add_argument("-s", "--smoothed", dest="smoothed", default=True, type=str2bool, nargs='?')
    parser.add_argument("-a", "--attack_method", dest="attack_method", default="cert")
    parser.add_argument("-sn", "--smooth_N", dest="smooth_N", default=10000, type=int)
    parser.add_argument("-sn0", "--smooth_N0", dest="smooth_N0", default=100, type=int)
    parser.add_argument("-f", "--failure_prob", dest="failure_prob", default=0.001, type=float)
    parser.add_argument("-t", "--targeted", dest="targeted", default=False, type=str2bool, nargs='?')
    parser.add_argument("-n", "--noise-level", dest="noise_level", default="0.50", type=str)
    parser.add_argument("-r", "--report-acc", dest="report_acc", default=False, type=str2bool, nargs='?')
    args = parser.parse_args()
    
    # checkpoint_path = "/home/zxz147/git_clones/adversarial-robustness-toolbox/test/de-certification/models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar"
    
    print(args)
    smoothed_path = f"/home/zxz147/git_clones/adversarial-robustness-toolbox/test/de-certification/models/cifar10/resnet110/noise_{args.noise_level}/checkpoint.pth.tar"
    attack(
        model_path=smoothed_path, 
        smoothed=args.smoothed, 
        report_acc=args.report_acc, 
        attack_method=args.attack_method, 
        smooth_N=args.smooth_N, 
        smooth_N0=args.smooth_N0, 
        failure_prob=args.failure_prob, 
        targeted=args.targeted
    )