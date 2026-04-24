"""Boucles train/test partagees (FedAvg / FedProx / FedNova / FedSGD)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import class_accuracies_from_preds, macro_recall_f1_from_preds


def train(net, loader, epochs, lr, device, mu=0.0, global_params=None):
    """Entrainement local multi-epoch. Si mu > 0 et global_params fourni -> FedProx."""
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    tot_loss, tot_ex, steps = 0.0, 0, 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = net(x)
            loss = crit(out, y)
            if mu > 0 and global_params is not None:
                prox = sum(((p - gp) ** 2).sum() for p, gp in zip(net.parameters(), global_params))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()
            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_ex += bs
            steps += 1
    return tot_loss / max(tot_ex, 1), steps


def fedsgd_update(net, loader, lr, device):
    """FedSGD : UN seul pas de gradient sur UN SEUL mini-batch.

    Le serveur (FedAvg) moyenne ensuite les poids retournes, ce qui revient
    a moyenner les gradients (w_avg = w - lr * mean(g_i)).
    """
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    opt.zero_grad()
    loss = crit(net(x), y)
    loss.backward()
    opt.step()
    return loss.item(), y.size(0)


def test(net, loader, device):
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    net.eval()
    tot_loss, tot_ok, tot_ex = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss = crit(out, y)
            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_ok += (out.argmax(1) == y).sum().item()
            tot_ex += bs
    return tot_loss / max(tot_ex, 1), tot_ok / max(tot_ex, 1)


def test_with_class_accuracies(net, loader, device, num_classes=10):
    """Comme test() mais renvoie aussi la liste des accuracies par classe."""
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    net.eval()
    tot_loss, tot_ex = 0.0, 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss = crit(out, y)
            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_ex += bs
            ys.append(y.cpu().numpy())
            ps.append(out.argmax(1).cpu().numpy())
    if tot_ex == 0:
        return 0.0, 0.0, [0.0] * num_classes, 0.0, 0.0
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    overall_acc = float((y_true == y_pred).sum() / tot_ex)
    class_accs = class_accuracies_from_preds(y_true, y_pred, num_classes=num_classes)
    macro_recall, macro_f1 = macro_recall_f1_from_preds(y_true, y_pred)
    return tot_loss / tot_ex, overall_acc, class_accs, macro_recall, macro_f1
