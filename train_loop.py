from typing import Union
import torch
from torch_geometric.loader import DataLoader


def reset_metrics(metrics:dict):
    # reset metric for evaluation
    for _, metric in metrics.items():
        metric.reset()


def _merge_loss(loss_fn: callable, out: torch.Tensor, target: torch.Tensor, aux_loss: torch.Tensor):
    loss = loss_fn(out, target)
    if aux_loss is not None:
        loss = loss + aux_loss
    return loss


def train_epoch(
        model:torch.nn.Module, 
        loader:DataLoader, 
        optimizer:torch.optim.Optimizer, 
        loss_fn:callable,
        metrics:dict,
        device:torch.cuda.device
) -> float:
    #trainning process

    model.train()
    reset_metrics(metrics=metrics)
         
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        # add axuliary loss
        out, aux_loss = model(data)     
        loss = _merge_loss(loss_fn, out, data.y, aux_loss)
        loss.backward()
        optimizer.step()
        # detach loss for document
        loss = loss.detach()
        metrics["loss"].update(loss)
    loss = metrics["loss"].compute() 
    return float(loss)

@torch.no_grad()
def evaluate_epoch(
    model:torch.nn.Module, 
    loader:DataLoader, 
    loss_fn:callable,
    metrics:dict,
    device:torch.cuda.device
) -> Union[float, float]:
    #testing process
    
    model.eval()

    reset_metrics(metrics=metrics)  

    for data in loader:
        data = data.to(device)
        out, aux_loss = model(data)
        loss = _merge_loss(loss_fn, out, data.y, aux_loss)

        # categorical accuracy if its classification problem
        metrics["acc"].update(out, data.y)
        metrics["loss"].update(loss)
        
    acc = metrics["acc"].compute()
    loss = metrics["loss"].compute()
    return float(acc), float(loss)
