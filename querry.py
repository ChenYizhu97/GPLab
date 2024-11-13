import json
import typer
from typing_extensions import Optional, Annotated, Tuple
import numpy as np

app = typer.Typer(pretty_exceptions_enable=False)


def _load_jsonstream(file:str):
    with open(file, mode="r") as f:
        json_strs = f.readlines()
        json_objs = [json.loads(json_str) for json_str in json_strs]
    return json_objs


def _filter_pool(pool:str, data:dict):
    data = filter(lambda x: x["pool"]["method"] == pool, data)
    return data

def _filter_dataset(dataset:str, data:dict):
    data = filter(lambda x: x["dataset"].lower() == dataset.lower(), data)
    return data 

def _filter_comment(comment:str, data:dict):
    # todo: replace equal match with re match
    data = filter(lambda x: x.get("comment", "no") == comment, data)
    return data

def _read_epochs(record:dict):
    avg_epoch = np.mean(record["results"]["data"]["epochs_stop"])
    avg_epoch = dict(avg_epoch=avg_epoch)
    return avg_epoch

def _read_statistic(record: dict):
    statistic = record["results"]["statistic"]
    return statistic

def _read_default(record: dict):
    pool = record["pool"]
    dataset = record["dataset"]
    default = {**pool, "dataset":dataset}
    comment = record.get("comment", None)
    if comment is not None:
        default["comment"] = comment
    return default


def _read(record:dict, epoch:bool=False):
    pool_n_dataset = _read_default(record)
    statistic = _read_statistic(record)
    read = {**pool_n_dataset,  **statistic}
    if epoch:
        avg_epoch = _read_epochs(record)
        read = {**read, **avg_epoch}
    return read

@app.command()
def main(
        file:Annotated[str, typer.Argument()],
        pool:Annotated[Optional[str], typer.Option()] = None, 
        dataset:Annotated[Optional[str], typer.Option()] = None,
        comment:Annotated[Optional[str], typer.Option()] = None,
        epoch:Annotated[Optional[bool], typer.Option()] = False,
    ):
    # show the experiments results for given conditions

    records = _load_jsonstream(file)
    if dataset is not None:
        records = _filter_dataset(dataset, records)
    if pool is not None:
        records = _filter_pool(pool, records)


    if comment is not None:
        records = _filter_comment(comment, records)


    results = [_read(record, epoch) for record in records]
    [print(result) for result in results]

if __name__=="__main__":
    app()