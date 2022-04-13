import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import albumentations
import albumentations.pytorch
import numpy as np
import math
import pandas as pd
import random
import os
import matplotlib
import argparse
import wandb

from EnD import *
from configs import *
from collections import defaultdict
import colour_mnist
import celeba
import utils

import models
from tqdm import tqdm

device = torch.device('cpu')

def num_correct(outputs,labels):
    _, preds = torch.max(outputs, dim=1)
    correct = preds.eq(labels).sum()
    return correct

def train(model, dataloader, criterion, weights, optimizer, scheduler):
    num_samples = 0
    tot_correct = 0
    tot_loss = 0
    tot_bce = 0.
    tot_abs = 0.
    model.train()

    for data, labels, color_labels,_ in tqdm(dataloader, leave=False):
        data, labels, color_labels = data.to(device=device), labels.to(device=device,dtype=torch.long), color_labels.to(device=device,dtype=torch.long)

        optimizer.zero_grad()
        with torch.enable_grad():
            outputs = model(data)
        bce, abs = criterion(outputs, labels, color_labels, weights)
        loss = bce+abs
        loss.backward()
        optimizer.step()

        batch_size = data.shape[0]
        tot_correct += num_correct(outputs, labels).item()
        num_samples += batch_size
        tot_loss += loss.item() * batch_size
        tot_bce += bce.item() * batch_size
        tot_abs += abs.item() * batch_size

    if scheduler is not None:
        scheduler.step()

    avg_accuracy = tot_correct / num_samples
    avg_loss = tot_loss / num_samples
    return avg_accuracy, avg_loss, tot_bce/num_samples, tot_abs/num_samples

def test(model, dataloader, criterion, weights):
    num_samples = 0
    tot_correct = 0
    tot_loss = 0

    y_all = []
    scores_all = []

    model.eval()

    for data, labels, color_labels,both_labels in tqdm(dataloader, leave=False):
        data, labels, color_labels, both_labels = data.to(device), labels.to(device=device,dtype=torch.long), color_labels.to(device=device,dtype=torch.long), both_labels.to(device=device,dtype=torch.long)

        with torch.no_grad():
            outputs = model(data)

        scores,_ = outputs.max(dim=1)
        scores = torch.sigmoid(scores).squeeze()
        scores_all.append(scores.detach().cpu().numpy())

        y_all.append(both_labels.detach().cpu().numpy())

        loss = criterion(outputs, labels, color_labels, weights)

        batch_size = data.shape[0]
        tot_correct += num_correct(outputs, labels).item()
        num_samples += batch_size
        tot_loss += loss.item() * batch_size

    y_all = np.concatenate(y_all)
    pred_all = np.concatenate(scores_all)
    avg_accuracy = tot_correct / num_samples
    avg_loss = tot_loss / num_samples
    return avg_accuracy, avg_loss, y_all, pred_all


def main(config):
    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

    if config.dataset == 'mnist':
        train_loader, valid_loader = colour_mnist.get_biased_mnist_dataloader(
            f'{os.path.expanduser("~")}/data',
            config.batch_size,
            config.rho,
            train=True
        )

        biased_test_loader = colour_mnist.get_biased_mnist_dataloader(
            f'{os.path.expanduser("~")}/data',
            config.batch_size,
            1.0,
            train=False
        )

        unbiased_test_loader = colour_mnist.get_biased_mnist_dataloader(
            f'{os.path.expanduser("~")}/data',
            config.batch_size,
            0.1,
            train=False
        )

    elif config.dataset == 'celeba':
        train_loader = celeba.create_dataset(config.batch_size,
            config.dataset_path,
            config.attribute,
            config.protected_attribute,
            True, split='train')

        valid_loader = celeba.create_dataset(config.batch_size,
            config.dataset_path,
            config.attribute,
            config.protected_attribute,
            False, split='valid')

        test_loader = celeba.create_dataset(config.batch_size,
            config.dataset_path,
            config.attribute,
            config.protected_attribute,
            False, split='test')

    print('Training debiased model')
    print('Config:', config)

    if config.model == 'simple_convnet':
        model = models.simple_convnet()
    elif config.model == 'resnet18':
        model = models.resnet18()
    elif config.model == 'resnet50':
        model = models.resnet50()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, verbose=True)
    hook = Hook(model.avgpool, backward=False)

    def ce(outputs, labels, color_labels, weights):
        return F.cross_entropy(outputs, labels)

    def ce_abs(outputs, labels, color_labels, weights):
        loss = ce(outputs, labels, color_labels, weights)
        abs = abs_regu(hook, labels, color_labels, config.alpha, config.beta)
        return loss, abs

    best = defaultdict(float)

    for i in range(config.epochs):
        train_acc, train_loss, train_bce, train_abs = train(model, train_loader, ce_abs, None, optimizer, scheduler=None)
        scheduler.step()


        valid_acc, valid_loss,val_targets, val_scores = test(model, valid_loader, ce, None)
        test_acc, test_loss,test_targets, test_scores = test(model, test_loader, ce, None)

        print(f'Epoch {i} - Train acc: {train_acc:.4f}, train_loss: {train_loss:.4f} (bce: {train_bce:.4f} abs: {train_abs:.4f});')

        cal_thresh = utils.calibrated_threshold(val_targets[:, 0], val_scores)
        f1_score, f1_thresh = utils.get_threshold(val_targets[:, 0], val_scores)
        val_pred = np.where(val_scores > cal_thresh, 1, 0)
        test_pred = np.where(test_scores > cal_thresh, 1, 0)

        ap_val, ap_std_val = utils.bootstrap_ap(val_targets[:, 0], val_scores)
        deo_val, deo_std_val = utils.bootstrap_deo(val_targets[:, 1], val_targets[:, 0], val_pred)
        ba_val, ba_std_val = utils.bootstrap_bias_amp(val_targets[:, 1], val_targets[:, 0], val_pred)
        kl_val, kl_std_val = utils.bootstrap_kl(val_targets[:, 1], val_targets[:, 0], val_scores)


        print('Validation results: ')
        print('AP : {:.1f} +- {:.1f}', 100 * ap_val, 200 * ap_std_val)
        print('DEO : {:.1f} +- {:.1f}', 100 * deo_val, 200 * deo_std_val)
        print('BA : {:.1f} +- {:.1f}', 100 * ba_val, 200 * ba_std_val)
        print('KL : {:.1f} +- {:.1f}', kl_val, 2 * kl_val)


        ap, ap_std = utils.bootstrap_ap(test_targets[:, 0], test_scores)
        deo, deo_std = utils.bootstrap_deo(test_targets[:, 1], test_targets[:, 0], test_pred)
        ba, ba_std = utils.bootstrap_bias_amp(test_targets[:, 1], test_targets[:, 0], test_pred)
        kl, kl_std = utils.bootstrap_kl(test_targets[:, 1], test_targets[:, 0], test_scores)


        print('Test results: ')
        print('AP : {:.1f} +- {:.1f}', 100 * ap, 200 * ap_std)
        print('DEO : {:.1f} +- {:.1f}', 100 * deo, 200 * deo_std)
        print('BA : {:.1f} +- {:.1f}', 100 * ba, 200 * ba_std)
        print('KL : {:.1f} +- {:.1f}', kl, 2 * kl)


        if ap_val > best['valid_ap']:
            best = dict(
                valid_ap = ap_val,
                valid_deo = deo_val,
                valid_ba = ba_val,
                valid_kl = kl_val,
                test_ap = ap,
                test_deo = deo,
                test_ba = ba,
                test_kl = kl
            )

        if not config.local:
            metrics = {
                'train_acc': train_acc,
                'train_loss': train_loss,
                'train_bce': train_bce,
                'train_abs': train_abs,

                'valid_ap': ap_val,
                'valid_loss': valid_loss,

                'test_ap': ap,
                'test_loss': test_loss,

                'best': best
            }
            wandb.log(metrics)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config}, os.path.join(wandb.run.dir, 'model.pt'))

if __name__ == '__main__':
    if not config.local:
        hyperparameters_defaults = dict(
            lr=config.lr,
            alpha=config.alpha,
            beta=config.beta,
            weight_decay=config.weight_decay,
            batch_size=config.batch_size,
            epochs=config.epochs,
            rho=config.rho
        )
        hyperparameters_defaults.update(vars(config))

        labels_attr = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
                'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        tags = [labels_attr[config.attribute]]
        if config.alpha == 0 and config.beta == 0:
            tags = ['baseline']
        tags.append(str(config.rho))


        wandb.init(
            config=hyperparameters_defaults,
            project='EnD-cvpr21',
            anonymous='allow',
            name=f'Celeba-{tags[0]}-valid',
            tags=tags,
            group=tags[0]
        )

    device = torch.device(config.device)
    main(config)
