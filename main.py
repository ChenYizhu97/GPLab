import json
import toml
import typer
import torch
import numpy as np
import hashlib
from decimal import Decimal
from pathlib import Path
from rich import print as rprint
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import summary
from torcheval.metrics import MulticlassAccuracy, Mean
from model.Classifer_Sum import GRAPH_CLASSIFIER_SUM
from layers.resolver import list_builtin_pools
from utils.io import print_expr_info, sep_c, build_runtime_meta
from utils.dataset import load_dataset, split_dataset, build_split_indices, build_dataset_id
from utils.reproducibility import (
    resolve_seeds,
    generate_loader,
    set_np_and_torch,
    configure_runtime_threads,
)
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


def _build_repro_block(conf: dict, seed_mode: str, seed_base: Optional[int], dataset_id: dict) -> dict:
    # Minimal reproducibility payload without schema/version branching.
    return {
        "seed_mode": seed_mode,
        "seed_base": seed_base,
        "seeds": conf["experiment"]["seeds"],
        "split_digest": conf["experiment"]["split_digest"],
        "split_ratio": conf["experiment"]["split_ratio"],
        "dataset_id": dataset_id,
        "env": conf["meta"],
    }


def _run_experiment(
        mode_conf: dict,
        expr_conf_dict: dict,
        pooling: str,
        pool_ratio: float,
        dataset_name: str,
        logging: Optional[str],
        comment: Optional[str],
        seed_mode: str,
        seed_base: Optional[int],
        allow_duplicate_seeds: bool,
) -> dict:
    is_custom_pool = _validate_cli_options(dataset_name, pooling, pool_ratio)

    if "model" not in mode_conf:
        raise ValueError("Missing [model] section in model config")
    if "experiment" not in expr_conf_dict:
        raise ValueError("Missing [experiment] section in experiment config")

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
    conf = {**mode_conf, **expr_conf_dict}

    expr_conf = conf["experiment"]
    train_ratio = float(expr_conf.get("train_ratio", 0.8))
    val_ratio = float(expr_conf.get("val_ratio", 0.1))
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratio. Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.")

    expr_conf["train_ratio"] = train_ratio
    expr_conf["val_ratio"] = val_ratio
    expr_conf["seed_mode"] = seed_mode
    expr_conf["seed_base"] = seed_base
    expr_conf["allow_duplicate_seeds"] = allow_duplicate_seeds

    conf["pool"] = pool_conf
    conf["dataset"] = dataset_name

    if comment is not None:
        conf["comment"] = comment

    configure_runtime_threads()
    set_np_and_torch(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["meta"] = build_runtime_meta(device)
    print_expr_info(conf, device)

    dataset = load_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Failed to load dataset '{dataset_name}'.")
    if len(dataset) == 0:
        raise ValueError("Loaded dataset is empty.")
    avg_node_num = dataset._data.num_nodes // len(dataset)

    metrics = {
        "loss": Mean(device=device),
        "acc": MulticlassAccuracy(average="micro", device=device, num_classes=dataset.num_classes),
    }

    model = GRAPH_CLASSIFIER_SUM(
        dataset.num_node_features,
        dataset.num_classes,
        pool_method=pooling,
        ratio=pool_ratio,
        config=conf["model"],
        avg_node_num=avg_node_num,
    ).to(device)

    rprint(summary(model, data=dataset[0].to(device), leaf_module=None, max_depth=5))

    seeds = resolve_seeds(
        runs=expr_conf["runs"],
        seed_mode=seed_mode,
        seeds_path=expr_conf.get("seeds"),
        seed_base=seed_base if seed_base is not None else 20260320,
        allow_duplicate_seeds=allow_duplicate_seeds,
    )
    conf["experiment"]["seeds"] = seeds

    split_indices_all = [
        build_split_indices(len(dataset), seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
        for seed in seeds
    ]
    split_digest = hashlib.sha1(json.dumps(split_indices_all, sort_keys=True).encode("utf-8")).hexdigest()
    train_ratio_dec = Decimal(str(train_ratio))
    val_ratio_dec = Decimal(str(val_ratio))
    test_ratio_dec = Decimal("1") - train_ratio_dec - val_ratio_dec
    conf["experiment"]["split_digest"] = split_digest
    conf["experiment"]["split_ratio"] = {
        "train": float(train_ratio_dec),
        "val": float(val_ratio_dec),
        "test": float(test_ratio_dec),
    }

    loss_list = []
    acc_list = []
    epoch_list = []
    run_records = []

    for r in range(1, expr_conf["runs"] + 1):
        run_seed = seeds[r - 1]
        run_split = split_indices_all[r - 1]

        set_np_and_torch(run_seed)
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, split_indices=run_split)

        train_loader = generate_loader(train_dataset, expr_conf["batch_size"], shuffle=True, seed=run_seed)
        val_loader = generate_loader(val_dataset, expr_conf["batch_size"], shuffle=False, seed=run_seed)
        test_loader = generate_loader(test_dataset, expr_conf["batch_size"], shuffle=False, seed=run_seed)

        model.reset_parameters()
        opt = torch.optim.Adam(model.parameters(), lr=expr_conf["lr"])
        loss_fn = F.nll_loss

        best_val_loss = np.inf
        best_test_acc = 0
        best_epoch = 1

        loop = tqdm(range(1, expr_conf["epochs"] + 1))
        for _, epoch in enumerate(loop):
            train(model, train_loader, opt, loss_fn, metrics, device)
            _, val_loss = test(model, val_loader, loss_fn, metrics, device)
            test_acc, _ = test(model, test_loader, loss_fn, metrics, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_acc = test_acc
                best_epoch = epoch

            if epoch > best_epoch + expr_conf["patience"]:
                break

            loop.set_description(f"Run [{r}/{expr_conf['runs']}]-Epoch [{epoch}/{expr_conf['epochs']}]")
            loop.set_postfix(best_epoch=best_epoch, best_test_acc=best_test_acc, best_val_loss=best_val_loss)

        if r != expr_conf["runs"]:
            rprint(sep_c('-'))

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

    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    conf["results"] = {
        "statistic": {"mean": acc_mean, "std": acc_std},
        "data": {"val_loss": loss_list, "test_acc": acc_list, "epochs_stop": epoch_list, "runs": run_records},
    }
    conf["repro"] = _build_repro_block(
        conf,
        seed_mode=seed_mode,
        seed_base=seed_base,
        dataset_id=build_dataset_id(dataset_name, dataset),
    )

    if logging is not None:
        log_path = Path(logging)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a+", encoding="utf-8") as file_to_save:
            print(json.dumps(conf), file=file_to_save)

    return conf


@app.command()
def main(
        pooling: Annotated[str, typer.Option()] = "nopool",
        pool_ratio: Annotated[float, typer.Option(help="Pooling ratio for built-in or custom pooling methods.")] = 0.5,
        dataset: Annotated[str, typer.Option()] = "PROTEINS",
        logging: Annotated[Optional[str], typer.Option()] = None,
        model_conf: Annotated[str, typer.Option()] = "config/model.toml",
        expr_conf: Annotated[str, typer.Option()] = "config/experiment.toml",
        comment: Annotated[Optional[str], typer.Option()] = None,
        seed_mode: Annotated[Optional[str], typer.Option(help="Seed source mode: auto or file.")] = None,
        seed_base: Annotated[Optional[int], typer.Option(help="Base integer for deterministic seed generation in auto mode.")] = None,
        allow_duplicate_seeds: Annotated[Optional[bool], typer.Option(help="Allow duplicate seeds in file mode.")] = None,
):
    mode_conf = toml.load(model_conf)
    expr_conf_data = toml.load(expr_conf)

    exp = expr_conf_data.get("experiment", {})
    final_seed_mode = seed_mode if seed_mode is not None else exp.get("seed_mode", "auto")
    final_seed_base = seed_base if seed_base is not None else exp.get("seed_base", 20260320)
    final_allow_dup = allow_duplicate_seeds if allow_duplicate_seeds is not None else exp.get("allow_duplicate_seeds", False)

    _run_experiment(
        mode_conf=mode_conf,
        expr_conf_dict=expr_conf_data,
        pooling=pooling,
        pool_ratio=pool_ratio,
        dataset_name=dataset,
        logging=logging,
        comment=comment,
        seed_mode=final_seed_mode,
        seed_base=final_seed_base,
        allow_duplicate_seeds=final_allow_dup,
    )


if __name__ == "__main__":
    app()
