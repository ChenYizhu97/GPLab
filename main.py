import json
import toml
import typer
import torch
import numpy as np
from rich import print as rprint
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import summary
from torcheval.metrics import MulticlassAccuracy, Mean
from model.Classifier import GRAPH_CLASSIFIER
from utils.io import print_expr_info, sep_c
from utils.dataset import load_dataset, split_dataset
from training import train, test

from typing_extensions import Annotated, Optional

TU_DATASET = ["MUTAG", "PROTEINS", "ENZYMES", "FRANKENSTEIN", "Mutagenicity", "AIDS", "DD"]
POOLING = ["nopool",  "topkpool", "sagpool", "diffpool", "mincutpool", "densepool", "sparsepool"]

def main(
        pooling: Annotated[str, typer.Option()] = "nopool",
        pool_ratio: Annotated[float, typer.Option()] = "0.5",
        dataset: Annotated[str, typer.Option()] = "PROTEINS",
        logging: Annotated[Optional[str], typer.Option()] = None,
        config: Annotated[str, typer.Option()] = "config/config.toml",
        comment: Annotated[Optional[str], typer.Option()] = None
):
    #check dataset and pooling
    assert dataset in TU_DATASET
    assert pooling in POOLING

    # load setting from config
    conf = toml.load(config)
    pool_conf = dict(method=pooling, ratio=pool_ratio)
    conf["pool"] = pool_conf
    conf["dataset"] = dataset
    if comment is not None: conf["comment"] = comment

    expr_conf = conf["experiment"]
    
    #set gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_expr_info(conf, device)

    # load dataset
    dataset = load_dataset(dataset)
    avg_node_num = dataset._data.num_nodes//len(dataset)

    #metrics to evaluate model
    metrics = {
        "loss": Mean(device=device),
        "acc": MulticlassAccuracy(
            average="micro", 
            device=device,
            num_classes=dataset.num_classes
        ),
    }

    #generate model from config
    model = GRAPH_CLASSIFIER(
        dataset.num_node_features, 
        dataset.num_classes, 
        pool_method=pool_conf["method"], 
        config=conf["model"],
        avg_node_num=avg_node_num
    ).to(device)
    
    rprint(summary(model, data=dataset[0].to(device), leaf_module=None, max_depth=5))
    
    loss_list = []
    acc_list = []
    epoch_list = []
    
    for r in range(1, expr_conf["runs"] + 1):

        # resplit dataset for each run
        train_loader, val_loader, test_loader = split_dataset(r, expr_conf, dataset)
        # reset model and optimizer
        model.reset_parameters() 
        opt = torch.optim.Adam(model.parameters(), lr=expr_conf["lr"])  
        loss_fn = F.cross_entropy  

        best_val_loss = np.inf
        best_test_acc = 0
        best_epoch = 1

        # run epochs
        loop = tqdm(range(1, expr_conf["epochs"] + 1))
        for _, epoch in enumerate(loop):
            #train and test for each epoch
            train(model, train_loader, opt, loss_fn, metrics, device)
            _, val_loss = test(model, val_loader, loss_fn, metrics, device)
            test_acc, _ = test(model, test_loader, loss_fn, metrics, device)

            #early stoping with patience
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_acc = test_acc
                best_epoch = epoch

            if epoch > best_epoch + expr_conf["patience"]: break

            #progress bar for tqmd
            loop.set_description(f"Run [{r}/{expr_conf['runs']}]-Epoch [{epoch}/{expr_conf['epochs']}]")
            loop.set_postfix(best_epoch= best_epoch, best_test_acc=best_test_acc, best_val_loss = best_val_loss)

        if r != expr_conf["runs"]: rprint(sep_c('-'))
        
        loss_list.append(best_val_loss)
        acc_list.append(best_test_acc)
        epoch_list.append(best_epoch)

    # calculate statistics
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    statistic = dict(mean = acc_mean, std = acc_std)    
    data = dict(val_loss=loss_list, test_acc=acc_list, epochs_stop=epoch_list)
    conf["results"] = dict(statistic=statistic, data=data)
    
    # save experiment information and results
    if logging is not None:
        with open(logging, "a+") as file_to_save:
            json_str = json.dumps(conf)
            print(json_str, file=file_to_save)
        
if __name__ == "__main__":
    typer.run(main)