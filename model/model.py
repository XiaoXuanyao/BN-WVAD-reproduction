import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from UR_DMU.model import *
from utils.debug import Debug, CostTime
from utils.display import display_results
from sklearn.metrics import roc_auc_score


# Enhancer Module
class Enhancer(nn.Module):
    def __init__(self):
        super(Enhancer, self).__init__()
        self.enc = Temporal(input_size=400, out_size=512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.5)
        self.amem = Memory_Unit(nums=60, dim=512)
        self.nmem = Memory_Unit(nums=60, dim=512)
        self.vaeenc_mu = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.vaeenc_var = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.msconv = nn.ModuleList([
            nn.Conv1d(512, 128, 3, padding=1),
            nn.Conv1d(512, 128, 5, padding=2),
            nn.Conv1d(512, 128, 7, padding=3)
        ])
        self.bn = nn.ModuleList([
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(128)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(512 + 384 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512)
        )
        self.dropout = nn.Dropout(0.1)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x, mode="train"):
        B, T, D = x.size()
        x = self.enc(x)
        x = self.selfatt(x)
        x = self.dropout(x)

        natt, ah = self.amem(x)
        natt, nh = self.nmem(x)
        h = (ah + nh) / 2

        z = h
        if mode == "train":
            fm = h.view(-1, 512)
            mu = self.vaeenc_mu(fm)
            logvar = self.vaeenc_var(fm)
            z = self._reparameterize(mu, logvar)
            z = z.view(B, T, -1)
        
        x2 = x.transpose(1, 2)
        msf = []
        for i, (conv, bn) in enumerate(zip(self.msconv, self.bn)):
            msf.append(F.relu(bn(conv(x2))))
        msf = torch.cat(msf, dim=1)
        msf = msf.transpose(1, 2)

        x = torch.cat([x, msf, z], dim=-1)
        x = self.fusion(x)
        return x


# Backbone Module
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv1d(512, 32, 1, 1, 0)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(512, 16, 1, 1, 0)
        self.bn2 = nn.BatchNorm1d(16)
    
    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B * T, D, 1)
        x1 = F.relu(self.bn1(self.conv1(x))).view(B, T, -1)
        x2 = F.relu(self.bn2(self.conv2(x))).view(B, T, -1)
        return x1, x2


# Classifier Module
class Classifier(nn.Module):
    def __init__(self, input_channels):
        super(Classifier, self).__init__()
        self.conv = nn.Conv1d(200 * input_channels, 1, 1, 1, 0)
    
    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B, T * D, 1)
        return F.sigmoid(self.conv(x)).view(B)


# Calculate Mahalanobis Distance from Feature and mean
def DFM(xh, mean, var):
    d = xh - mean
    inv = 1.0 / (var + 1e-6)
    inv = inv.view(1, 1, -1)

    wd = d * inv
    if d.dim() == 1:
        dfm = torch.sum(wd * d)
    else:
        dfm = torch.sum(wd * d, dim=-1)
    return dfm

# Sample-level Selection (SLS) Strategy
def SLS(x: list[dict], B, T, k):
    res = []
    for i in range(k):
        bi = i % B
        for e in x:
            if e["B"] == bi:
                res.append(e)
                x.remove(e)
                break
    return res

# Batch-level Selection (BLS) Strategy
def BLS(x: list[dict], B, T, k):
    return x[: k]

# Sample-Batch Selection (SBS) Strategy
def SBS(x: list[dict], B, T, ks, kb):
    res1 = SLS(x, B, T, ks)
    res2 = BLS(x, B, T, kb)
    res = res1 + res2
    unique = [e for e in sorted(res, key=lambda x: x["V"], reverse=True)]
    return unique
    
# Calculate MPP Loss
def MPPLoss(xn, xa, mean, var, ps, pb, m=1):
    B, T, D = xn.shape
    dfm_xn = DFM(xn, mean, var)
    dfm_xa = DFM(xa, mean, var)
    I, J = torch.meshgrid(torch.arange(B), torch.arange(T), indexing='ij')
    I = I.flatten()
    J = J.flatten()
    V_xn = dfm_xn.flatten()
    V_xa = dfm_xa.flatten()
    dfm_xn2 = [{"B": int(i), "T": int(j), "V": v.detach().item(), "tensor": v} for i, j, v in zip(I, J, V_xn)]
    dfm_xa2 = [{"B": int(i), "T": int(j), "V": v.detach().item(), "tensor": v} for i, j, v in zip(I, J, V_xa)]
    sorted_dfm_xn = sorted(dfm_xn2, key=lambda x: x["V"], reverse=True)
    sorted_dfm_xa = sorted(dfm_xa2, key=lambda x: x["V"], reverse=True)
    sorted_dfm_xa = SBS(sorted_dfm_xa, B, T, int(B * T * ps), int(B * T * pb))
    sorted_dfm_xn = SLS(sorted_dfm_xn, B, T, len(sorted_dfm_xa))

    sum = 0
    for i in range(len(sorted_dfm_xa)):
        sum += F.relu(m - sorted_dfm_xn[i]["tensor"] + sorted_dfm_xa[i]["tensor"])
    loss = sum / len(sorted_dfm_xa)
    return loss


# Custom Loss Function
class MLoss(nn.Module):
    def __init__(self, w1, w2):
        super(MLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2

    def forward(self, nor, mpp1, mpp2):
        return nor + mpp1 * self.w1 + mpp2 * self.w2


# Training for one epoch
def train_one_epoch(model, enhancer, classifier1, classifier2, dataloader, criterion, optimizer, alpha, ps, pb, w1, w2, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    __mean1 = torch.zeros(32).to(device)
    __var1 = torch.ones(32).to(device)
    __mean2 = torch.zeros(16).to(device)
    __var2 = torch.ones(16).to(device)
    for i, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        labels = labels.to(device)
        B, T, D = data.shape
        data = enhancer(data, mode="train")

        nmask = labels == 0
        amask = labels == 1

        optimizer.zero_grad()

        xh1, xh2 = model(data)

        mean1 = model.bn1.running_mean
        var1 = model.bn1.running_var
        mean2 = model.bn2.running_mean
        var2 = model.bn2.running_var
        __mean1 = (1 - alpha) * __mean1 + alpha * mean1
        __var1 = (1 - alpha) * __var1 + alpha * var1
        __mean2 = (1 - alpha) * __mean2 + alpha * mean2
        __var2 = (1 - alpha) * __var2 + alpha * var2

        nor = classifier1(xh1[nmask]) * w1 + classifier2(xh2[nmask]) * w2
        nor = torch.sum(nor)
        mpp1 = MPPLoss(xh1[nmask], xh1[amask], __mean1, __var1, ps=ps, pb=pb, m=1)
        mpp2 = MPPLoss(xh2[nmask], xh2[amask], __mean2, __var2, ps=ps, pb=pb, m=1)
        with torch.no_grad():
            cls = classifier1(xh1) * w1 + classifier2(xh2) * w2
            dfm1 = DFM(xh1, __mean1, __var1)
            dfm2 = DFM(xh2, __mean2, __var2)

        loss = criterion(nor, mpp1, mpp2).mean()
        with torch.no_grad():
            score = cls * torch.max(dfm1 * w1 + dfm2 * w2, -1)[0]
            all_preds.append(score.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print("labels:", labels.cpu().numpy())
        # print("preds:", pred)
        print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}", end='\r')
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    # display_results(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    return {
        "loss": total_loss / len(dataloader),
        "auc": auc
    }

# Validation for one epoch
def validate(model, enhancer, classifier1, classifier2, dataloader, criterion, alpha, ps, pb, w1, w2, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    __mean1 = torch.zeros(32).to(device)
    __var1 = torch.ones(32).to(device)
    __mean2 = torch.zeros(16).to(device)
    __var2 = torch.ones(16).to(device)
    with torch.no_grad():
        for i, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device)
            B, T, D = data.shape
            data = enhancer(data, mode="test")

            nmask = labels == 0
            amask = labels == 1

            xh1, xh2 = model(data)

            mean1 = model.bn1.running_mean
            var1 = model.bn1.running_var
            mean2 = model.bn2.running_mean
            var2 = model.bn2.running_var
            __mean1 = (1 - alpha) * __mean1 + alpha * mean1
            __var1 = (1 - alpha) * __var1 + alpha * var1
            __mean2 = (1 - alpha) * __mean2 + alpha * mean2
            __var2 = (1 - alpha) * __var2 + alpha * var2

            nor = classifier1(xh1[nmask]) * w1 + classifier2(xh2[nmask]) * w2
            nor = torch.sum(nor)
            mpp1 = MPPLoss(xh1[nmask], xh1[amask], __mean1, __var1, ps=ps, pb=pb, m=1)
            mpp2 = MPPLoss(xh2[nmask], xh2[amask], __mean2, __var2, ps=ps, pb=pb, m=1)
            cls = classifier1(xh1) * w1 + classifier2(xh2) * w2
            dfm1 = DFM(xh1, __mean1, __var1)
            dfm2 = DFM(xh2, __mean2, __var2)

            loss = criterion(nor, mpp1, mpp2)
            score = cls * torch.max(dfm1 * w1 + dfm2 * w2, -1)[0]
            pred = score.cpu().numpy()
            all_preds.append(pred)
            all_labels.append(labels.cpu().numpy())

            total_loss += loss.mean().item()
            print(f"Val Batch {i+1}/{len(dataloader)}, Loss: {loss.mean().item():.4f}", end='\r')
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return {
        "loss": total_loss / len(dataloader),
        "auc": roc_auc_score(all_labels, all_preds)
    }

# Main Training Loop
def train(model, enhancer, classifier1, classifier2, train_loader, val_loader, criterion, optimizer, epochs, alpha, ps, pb, w1, w2, device):
    for epoch in range(epochs):
        train_res = train_one_epoch(model, enhancer, classifier1, classifier2, train_loader, criterion, optimizer, alpha, ps, pb, w1, w2, device)
        val_res = validate(model, enhancer, classifier1, classifier2, val_loader, criterion, alpha, ps, pb, w1, w2, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_res['loss']:.4f}, AUC: {train_res['auc']:.4f}, Val Loss: {val_res['loss']:.4f}, AUC: {val_res['auc']:.4f}")