from __future__ import absolute_import, division, print_function, unicode_literals
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from smoothing.core import Smooth
from utils import remove_misclassify, load_classifier, str2bool
from art.utils import load_cifar10, to_categorical
from art.estimators.classification import PyTorchClassifier, pytorch
from art.attacks.evasion.boundary import BoundaryAttack
from art.attacks.evasion.boundary_swim import BoundaryAttackSwim
from art.attacks.evasion.boundary_cert import BoundaryAttackCert
from art.attacks.evasion.boundary_orig import BoundaryAttackOrig
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import OrderedDict
import argparse
import matplotlib.pyplot as plt
from logging import error

import sys
sys.path.insert(1, "/home/zxz147/git_forks/adversarial-robustness-toolbox")
sys.path.insert(1, "/home/zxz147/projects/De-Certification")


def attack(init_class: int, target_class: int, noise_level: float,
           model_path: str, smooth_N: int, smooth_N0: int, failure_prob: float,
           targeted: bool = False, model_architecture: str = "cifar_resnet110",
           dataset: str = "cifar10", report_acc: str = "False", smoothed: bool = True,
           attack_method: str = "cert"):
    # data preparation
    (x_train, y_train), (x_test,
                         y_test), min_pixel_value, max_pixel_value = load_cifar10()
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
    base_classifier = load_classifier(model_path, smoothed)

    # create art classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_classifier.parameters(), lr=0.01)
    pytorch_classifier = PyTorchClassifier(
        model=base_classifier,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10)

    # Examine the model accuracy under different settings.
    if report_acc == "True":
        predictions = pytorch_classifier.predict(x_test, training_mode=False)
        accuracy = np.sum(np.argmax(predictions, axis=1) ==
                          np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    elif report_acc == "True_Cert":
        smoothed_clf = Smooth(pytorch_classifier, 10, 0.12)
        predictions = []
        # predictions, _ = smoothed_clf.certify(x_test, smooth_N0, smooth_N, failure_prob, 128)
        for i in tqdm(range(x_test.shape[0])):
            data = x_test[i]
            pred, _ = smoothed_clf.certify(torch.tensor(
                data), smooth_N0, smooth_N, failure_prob, 1)
            predictions.append(pred)
            if (i + 1) % 10 == 0:
                cumu_acc = np.sum([x == y for x, y in zip(
                    predictions, np.argmax(y_test[:i], axis=1))]) / i
                print(
                    f"\npredictions: \n{predictions}\nground truth: \n{np.argmax(y_test[:i], axis=1)}")
                print(f"\nCumulative prob at step {i}: {cumu_acc}")
                # print(f"Predictions: {predictions}\nActual: {np.argmax(y_test[:i], axis=1)}")
        accuracy = np.sum(predictions == np.argmax(
            y_test, axis=1)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Removing the misclassified samples.
    _input = np.concatenate((x_train, x_test), axis=0)
    _label = np.concatenate((y_train, y_test), axis=0)
    _preds = pytorch_classifier.predict(_input, training_mode=False)
    _input, _label = remove_misclassify(_input, _preds, _label)

    # Obtaining the init and target images.
    img_init, img_targ = None, None
    imgs_found = [False, False]

    for img, label in zip(_input, _label):
        if np.argmax(label) == init_class and not imgs_found[0]:
            img_init = img
            imgs_found[0] = True
        if np.argmax(label) == target_class and not imgs_found[1]:
            img_targ = img
            imgs_found[1] = True

    print("Init image ({}): {}".format(init_class, 
        np.argmax(pytorch_classifier.predict(np.expand_dims(img_init, 0)))))
    print("Target image ({}): {}".format(target_class, 
        np.argmax(pytorch_classifier.predict(np.expand_dims(img_targ, 0)))))

    # exit()

    if attack_method == "cert":
        print("Using BoundaryAttackCert class as attack. ")
        attack = BoundaryAttackCert(estimator=pytorch_classifier, targeted=targeted, max_iter=0,
                                    delta=0.001, epsilon=0.001, smooth_N=smooth_N, smooth_N0=smooth_N0, smooth_alpha=failure_prob)
    elif attack_method == "orig_cert" or attack_method == "orig":
        print("Using BoundaryAttackOrig class as attack. ")
        attack = BoundaryAttackOrig(estimator=pytorch_classifier, targeted=targeted,
                                    max_iter=0, delta=0.001, epsilon=0.001, cert_pred=attack_method)
    elif attack_method == "swim":
        print("Using BoundaryAttackSwim class as attack. ")
        attack = BoundaryAttackSwim(estimator=pytorch_classifier, targeted=targeted, max_iter=0,
                                    delta=0.001, epsilon=0.001, smooth_N=smooth_N, smooth_N0=smooth_N0, smooth_alpha=failure_prob)
    # elif attack_method == "orig": 
    #     print("Using BoundaryAttack class as attack. ")
    #     attack = BoundaryAttack(estimator=pytorch_classifier, targeted=targeted, max_iter=0,
    #                                 delta=0.001, epsilon=0.001)
    else:
        error("Invalid attack_method parameters")
        exit()

    iter_step = 20
    x_adv = np.array([img_targ[..., ::-1]]) if targeted else None

    for i in range(10):
        x_adv = attack.generate(x=torch.tensor([img_init[..., ::-1]]), 
                                y=to_categorical([target_class], 10), 
                                x_adv_init=x_adv)
        # clear_output()
        print("Adversarial image at step %d." % (i * iter_step), "L2 error",
              np.linalg.norm(np.reshape(x_adv[0] - img_init[..., ::-1], [-1])),
              "and class label %d." % np.argmax(pytorch_classifier.predict(x_adv)[0]))
        # plt.imshow(x_adv[0][..., ::-1].astype(np.uint))
        # plt.show(block=False)

        if hasattr(attack, 'curr_delta') and hasattr(attack, 'curr_epsilon'):
            attack.max_iter = iter_step
            attack.delta = attack.curr_delta
            attack.epsilon = attack.curr_epsilon
        else:
            print("breaked")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", dest="checkpoint_path", type=str)
    parser.add_argument("-ic", "--init_class",
                        dest="init_class", default=1, type=int)
    parser.add_argument("-tc", "--target_class",
                        dest="target_class", default=5, type=int)
    parser.add_argument("-s", "--smoothed", dest="smoothed",
                        default=True, type=str2bool, nargs='?')
    parser.add_argument("-a", "--attack_method",
                        dest="attack_method", default="cert")
    parser.add_argument("-sn", "--smooth_N",
                        dest="smooth_N", default=10000, type=int)
    parser.add_argument("-sn0", "--smooth_N0",
                        dest="smooth_N0", default=100, type=int)
    parser.add_argument("-f", "--failure_prob",
                        dest="failure_prob", default=0.001, type=float)
    parser.add_argument("-t", "--targeted", dest="targeted",
                        default=False, type=str2bool, nargs='?')
    parser.add_argument("-n", "--noise_level",
                        dest="noise_level", default="0.50", type=str)
    parser.add_argument("-r", "--report_acc",
                        dest="report_acc", default="False")
    args = parser.parse_args()

    # checkpoint_path = "/home/zxz147/git_clones/adversarial-robustness-toolbox/test/de-certification/models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar"

    print(args)
    smoothed_path = f"/home/zxz147/git_clones/adversarial-robustness-toolbox/test/de-certification/models/cifar10/resnet110/noise_{args.noise_level}/checkpoint.pth.tar"
    attack(
        init_class=args.init_class,
        target_class=args.target_class,
        noise_level=args.noise_level,
        model_path=smoothed_path,
        smoothed=args.smoothed,
        report_acc=args.report_acc,
        attack_method=args.attack_method,
        smooth_N=args.smooth_N,
        smooth_N0=args.smooth_N0,
        failure_prob=args.failure_prob,
        targeted=args.targeted
    )
