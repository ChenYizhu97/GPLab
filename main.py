import json
import toml
import typer
import torch
import numpy as np
import hashlib
from pathlib import Path
from rich import print as rprint
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import summary
from torcheval.metrics import MulticlassAccuracy, Mean
# from model.Classifier import GRAPH_CLASSIFIER
from model.Classifer_Sum import GRAPH_CLASSIFIER_SUM
from layers.resolver import list_builtin_pools
from utils.io import print_expr_info, sep_c, build_runtime_meta
from utils.dataset import load_dataset, split_dataset, build_split_indices
from utils.reproducibility import load_seeds, generate_loader, set_np_and_torch
from training import train, test
from typing_extensions import Annotated, Optional

TU_DATASET = ["MUTAG", "PROTEINS", "ENZYMES", "FRANKENSTEIN", "Mutagenicity", "AIDS", "DD", "NCI1", "COX2"]
POOLING = list_builtin_pools()

app = typer.Typer(pretty_exceptions_enable=False)


def _validate_cli_options(dataset: str, pooling: str, pool_ratio: float) -> bool:
    if dataset not in TU_DATASET:
        raise typer.BadParameter(
            f"Unsupported dataset '{dataset}'. Supported datasets: {', '.join(TU_DATASET)}",
            param_hint="--dataset",
        )
    if pool_ratio <= 0.0 or pool_ratio > 1.0:
        raise typer.BadParameter("pool_ratio must be in (0, 1].", param_hint="--pool-ratio")

    is_custom_pool = ":" in pooling
    if not is_custom_pool and pooling not in POOLING:
        raise typer.BadParameter(
            f"Unknown pooling method '{pooling}'. Built-ins: {', '.join(POOLING)}",
            param_hint="--pooling",
        )
    return is_custom_pool


@app.command()
def main(
        pooling: Annotated[str, typer.Option()] = "nopool",
        pool_ratio: Annotated[float, typer.Option(help="Pooling ratio for built-in or custom pooling methods.")] = 0.5,
        dataset: Annotated[str, typer.Option()] = "PROTEINS",
        logging: Annotated[Optional[str], typer.Option()] = None,
        model_conf: Annotated[str, typer.Option()] = "config/model.toml",
        expr_conf: Annotated[str, typer.Option()] = "config/experiment.toml",
        comment: Annotated[Optional[str], typer.Option()] = None
):
    dataset_name = dataset
    # check dataset/pooling arguments with explicit error messages.
    is_custom_pool = _validate_cli_options(dataset_name, pooling, pool_ratio)

    # load setting from config
    mode_conf = toml.load(model_conf)
    expr_conf = toml.load(expr_conf)
    if "model" not in mode_conf:
        raise ValueError(f"Missing [model] section in {model_conf}")
    if "experiment" not in expr_conf:
        raise ValueError(f"Missing [experiment] section in {expr_conf}")

    pool_conf = dict(
        method=pooling,
        ratio=pool_ratio,
        source="custom_factory" if is_custom_pool else "builtin",
        protocol="unified_adapter",
        protocol_note=(
            "DiffPool/Mincut/DensePool currently use unified dense-to-sparse adapter in this benchmark; "
            "paper-style dense backbone path is intentionally ignored for now."
        ),
    )
    # merge dictionary
    conf = {**mode_conf, **expr_conf}

    expr_conf = conf["experiment"]
    train_ratio = float(expr_conf.get("train_ratio", 0.8))
    val_ratio = float(expr_conf.get("val_ratio", 0.1))
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratio. Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.")

    # Persist explicit ratios in record even when defaults are used.
    conf["experiment"]["train_ratio"] = train_ratio
    conf["experiment"]["val_ratio"] = val_ratio
    conf["pool"] = pool_conf
    conf["dataset"] = dataset_name
    
    if comment is not None: conf["comment"] = comment
  
    # set gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["meta"] = build_runtime_meta(device)
    # print settings to console
    print_expr_info(conf, device)
    # load dataset
    dataset = load_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Failed to load dataset '{dataset_name}'.")
    if len(dataset) == 0:
        raise ValueError("Loaded dataset is empty.")
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
        ratio=pool_ratio,
        config=conf["model"],
        avg_node_num=avg_node_num
    ).to(device)
    
    rprint(summary(model, data=dataset[0].to(device), leaf_module=None, max_depth=5))
    
    # load seeds and replace seeds file in config to true seeds for saving
    seeds = load_seeds(expr_conf["seeds"], expr_conf["runs"])
    conf["experiment"]["seeds"] = seeds
    split_indices_all = [
        build_split_indices(len(dataset), seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
        for seed in seeds
    ]
    split_digest = hashlib.sha1(
        json.dumps(split_indices_all, sort_keys=True).encode("utf-8")
    ).hexdigest()
    conf["experiment"]["split_digest"] = split_digest
    conf["experiment"]["split_ratio"] = {
        "train": train_ratio,
        "val": val_ratio,
        "test": 1 - train_ratio - val_ratio,
    }

    # experiment start
    loss_list = []
    acc_list = []
    epoch_list = []
    run_records = []
    # set_np_and_torch()
    for r in range(1, expr_conf["runs"] + 1):
        run_seed = seeds[r - 1]
        run_split = split_indices_all[r - 1]

        # resplit dataset for each run
        set_np_and_torch(run_seed)
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, split_indices=run_split)

        # initialize dataloader
        train_loader = generate_loader(train_dataset, expr_conf["batch_size"], shuffle=True, seed=run_seed)
        val_loader = generate_loader(val_dataset, expr_conf["batch_size"], shuffle=False, seed=run_seed)
        test_loader = generate_loader(test_dataset, expr_conf["batch_size"], shuffle=False, seed=run_seed)

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
        run_records.append(
            {
                "run": r,
                "seed": run_seed,
                "split_sizes": {
                    "train": len(train_dataset),
                    "val": len(val_dataset),
                    "test": len(test_dataset),
                },
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_test_acc": best_test_acc,
            }
        )

    # calculate experiment statistics
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    statistic = dict(mean = acc_mean, std = acc_std)    
    data = dict(val_loss=loss_list, test_acc=acc_list, epochs_stop=epoch_list, runs=run_records)
    conf["results"] = dict(statistic=statistic, data=data)
    
    # save experiment information and results
    if logging is not None:
        log_path = Path(logging)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a+", encoding="utf-8") as file_to_save:
            json_str = json.dumps(conf)
            print(json_str, file=file_to_save)
        
if __name__ == "__main__":
    app()
