# CUDA_VISIBLE_DEVICES=3 python codes/attack.py -m /home/zxz147/git_clones/adversarial-robustness-toolbox/test/de-certification/models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar 2>&1 | tee experiments/logs/21-05-11-run2-smtmodels.txt
# CUDA_VISIBLE_DEVICES=3 python codes/attack.py -m /data/zxz147/model/image/cifar-10/resnet/resnet-110/model_best.pth.tar 2>&1 | tee experiments/logs/21-05-11-run2-accmodels.txt

from __future__ import absolute_import, division, print_function, unicode_literals

import os 
import sys
sys.path.insert(1, "/home/zxz147/git_clones/adversarial-robustness-toolbox")
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10, to_categorical

from smoothing.architectures import get_architecture
import classifications.models.cifar as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def imshow(img):
    img = img / 2 + 0.5   # unnormalize
    # npimg = img   # convert from tensor
    # plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    # plt.show()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    

def attack(model_path, targeted=True, model_architecture="cifar_resnet110", dataset="cifar10", report_acc=True, smoothed=True): 
    if smoothed: 
        base_checkpoint = torch.load(model_path)
        base_classifier = get_architecture(model_architecture, dataset)
        base_classifier.load_state_dict(base_checkpoint["state_dict"])
        # base_classifier.eval()
    else: 
        base_checkpoint = torch.load(model_path)
        base_classifier = models.__dict__["resnet"](num_classes=10, depth=110, block_name="BasicBlock")
        state_dict = base_checkpoint['state_dict']
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k]=v

        base_classifier.load_state_dict(new_state_dict)
        # base_classifier.load_state_dict(base_checkpoint["state_dict"])

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

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
        predictions = pytorch_classifier.predict(x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    
    # Obtaining the init (cat) and target (dog) images. 
    img_init, img_targ = None, None
    imgs_found = [False, False]

    for img, label in zip(x_test, y_test):
        if np.argmax(label) == 3 and not imgs_found[0]: 
            img_init = img
            imgs_found[0] = True
        if np.argmax(label) == 5 and not imgs_found[1]: 
            img_targ = img
            imgs_found[1] = True
            
    # HopSkipJump targeted Attack, without masking
    attack = HopSkipJump(classifier=pytorch_classifier, targeted=True, max_iter=0, max_eval=1000, init_eval=10, verbose=True)
    iter_step = 1000
    x_adv = np.array([img_init]) if targeted else None
    for i in range(20): 
        x_adv = attack.generate(x=np.array([img_init]), y=to_categorical([5], 10), x_adv_init=x_adv, resume=True)
        print("Adversarial image at step %d." % (i * iter_step), "L2 error", 
              np.linalg.norm(np.reshape(x_adv[0] - img_targ, [-1])),
              "and class label %d." % np.argmax(pytorch_classifier.predict(x_adv)[0]))
        # plt.imshow(x_adv[0])
        # plt.show(block=False)

        attack.max_iter = iter_step
    
    
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="checkpoint_path", type=str)
    parser.add_argument("-s", "--smoothed", dest="smoothed", default=False, action="store_true")
    
    args = parser.parse_args()
    
    # checkpoint_path = "/home/zxz147/git_clones/adversarial-robustness-toolbox/test/de-certification/models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar"
    attack(args.checkpoint_path, smoothed=args.smoothed)