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
# from model.Classifier import GRAPH_CLASSIFIER
from model.Classifer_Sum import GRAPH_CLASSIFIER_SUM
from utils.io import print_expr_info, sep_c
from utils.dataset import load_dataset, split_dataset
from utils.reproducibility import load_seeds, generate_loader, set_np_and_torch
from training import train, test
from typing_extensions import Annotated, Optional

TU_DATASET = ["MUTAG", "PROTEINS", "ENZYMES", "FRANKENSTEIN", "Mutagenicity", "AIDS", "DD", "NCI1", "COX2"]
POOLING = ["nopool", "lspool", "topkpool", "sagpool", "diffpool", "mincutpool", "densepool", "sparsepool", "asapool"]

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
        pooling: Annotated[str, typer.Option()] = "nopool",
        pool_ratio: Annotated[float, typer.Option()] = "0.5",
        dataset: Annotated[str, typer.Option()] = "PROTEINS",
        logging: Annotated[Optional[str], typer.Option()] = None,
        model_conf: Annotated[str, typer.Option()] = "config/model.toml",
        expr_conf: Annotated[str, typer.Option()] = "config/experiment.toml",
        comment: Annotated[Optional[str], typer.Option()] = None
):
    #check dataset and pooling
    assert dataset in TU_DATASET
    assert pooling in POOLING

    # load setting from config
    mode_conf = toml.load(model_conf)
    expr_conf = toml.load(expr_conf)
    pool_conf = dict(method=pooling, ratio=pool_ratio)
    # merge dictionary
    conf = {**mode_conf, **expr_conf}

    expr_conf = conf["experiment"]
    mode_conf = conf["model"]
    conf["pool"] = pool_conf
    conf["dataset"] = dataset
    
    if comment is not None: conf["comment"] = comment
  
    # set gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print settings to console
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
    model = GRAPH_CLASSIFIER_SUM(
        dataset.num_node_features, 
        dataset.num_classes, 
        pool_method=pooling, 
        config=conf["model"],
        avg_node_num=avg_node_num
    ).to(device)
    
    rprint(summary(model, data=dataset[0].to(device), leaf_module=None, max_depth=5))
    
    # load seeds and replace seeds file in config to true seeds for saving
    seeds = load_seeds(expr_conf["seeds"], expr_conf["runs"])
    conf["experiment"]["seeds"] = seeds
    # print(seeds)

    # experiment start
    loss_list = []
    acc_list = []
    epoch_list = []
    # set_np_and_torch()
    for r in range(1, expr_conf["runs"] + 1):

        # resplit dataset for each run
        set_np_and_torch(seeds[r-1])
        train_dataset, val_dataset, test_dataset = split_dataset(dataset)

        # initialize dataloader
        train_loader = generate_loader(train_dataset, expr_conf["batch_size"], shuffle=True)
        val_loader = generate_loader(val_dataset, expr_conf["batch_size"], shuffle=True)
        test_loader = generate_loader(test_dataset, expr_conf["batch_size"], shuffle=True)

        # reset model, optimizer and loss function
        model.reset_parameters() 
        opt = torch.optim.Adam(model.parameters(), lr=expr_conf["lr"])  
        #loss_fn = F.cross_entropy  
        loss_fn = F.nll_loss

        # data to track
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

    # calculate experiment statistics
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
    app()