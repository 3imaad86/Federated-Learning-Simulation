"""Boucles train/test partagees par les 4 algos.

  * train(...)                     : SGD multi-epoch (FedAvg/FedNova) ou + prox (FedProx)
  * fedsgd_update(...)             : UN seul pas SGD sur UN batch (FedSGD)
  * test(...)                      : eval (loss, accuracy)
  * test_with_class_accuracies(...): eval + per-class + macro recall/F1
"""

import numpy as np
import torch
import torch.nn as nn

from .metrics import class_accuracies_from_preds, macro_recall_f1_from_preds


# ============================================================================
# 1) Entrainement local
# ============================================================================

def train(net, loader, epochs, lr, device, mu=0.0, global_params=None):
    """Entrainement multi-epoch SGD + momentum.

    Si mu > 0 et global_params fourni : ajoute le terme proximal FedProx
        loss += (mu / 2) * sum (p - gp)^2
    """
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()

    tot_loss, tot_ex, steps = 0.0, 0, 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(net(x), y)
            if mu > 0 and global_params is not None:
                prox = sum(((p - gp) ** 2).sum()
                           for p, gp in zip(net.parameters(), global_params))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()

            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_ex += bs
            steps += 1
    return tot_loss / max(tot_ex, 1), steps


def fedsgd_update(net, loader, lr, device):
    """FedSGD : UN seul pas SGD sur UN seul mini-batch (sans momentum).

    Le serveur fait FedAvg sur les poids retournes -> equivalent a moyenner
    les gradients : w_avg = w - lr * mean(g_i).
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


# ============================================================================
# 2) Evaluation (forward sans grad)
# ============================================================================

def _forward_eval(net, loader, device, collect_preds=False):
    """Forward pass sans grad. Retourne (tot_loss, tot_ok, tot_ex, ys, ps).

    Si collect_preds=False, ys/ps sont vides (economie memoire).
    """
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    net.eval()

    tot_loss, tot_ok, tot_ex = 0.0, 0, 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            bs = y.size(0)
            tot_loss += crit(out, y).item() * bs
            preds = out.argmax(1)
            tot_ok += (preds == y).sum().item()
            tot_ex += bs
            if collect_preds:
                ys.append(y.cpu().numpy())
                ps.append(preds.cpu().numpy())
    return tot_loss, tot_ok, tot_ex, ys, ps


def test(net, loader, device):
    """Eval simple. Retourne (loss_moyen, accuracy)."""
    tot_loss, tot_ok, tot_ex, _, _ = _forward_eval(net, loader, device)
    return tot_loss / max(tot_ex, 1), tot_ok / max(tot_ex, 1)


def test_with_class_accuracies(net, loader, device, num_classes=10):
    """Eval complete : loss, accuracy globale, accuracies par-classe, macro recall/F1."""
    tot_loss, _, tot_ex, ys, ps = _forward_eval(net, loader, device, collect_preds=True)
    if tot_ex == 0:
        return 0.0, 0.0, [0.0] * num_classes, 0.0, 0.0

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    overall_acc = float((y_true == y_pred).sum() / tot_ex)
    class_accs = class_accuracies_from_preds(y_true, y_pred, num_classes=num_classes)
    macro_recall, macro_f1 = macro_recall_f1_from_preds(y_true, y_pred)
    return tot_loss / tot_ex, overall_acc, class_accs, macro_recall, macro_f1
