import json
import typer
from typing_extensions import Optional, Annotated


def _read_jsonstream(file:str):
    with open(file, mode="r") as f:
        json_strs = f.readlines()
        json_objs = [json.loads(json_str) for json_str in json_strs]
    return json_objs

# replace x["pool"]["pool"] to x["method"]

def _filter_pool(pool:str, data:dict):
    data = filter(lambda x: x["pool"]["method"] == pool, data)
    return data

def _filter_dataset(dataset:str, data:dict):
    data = filter(lambda x: x["dataset"].lower() == dataset.lower(), data)
    return data 

def _filter_comment(comment:str, data:dict):
    data = filter(lambda x: x.get("comment", "no") == comment, data)
    return data

def _read_statistic(data: dict):
    data = [
        (x["pool"]["method"],x["dataset"],x["results"]["statistic"], x["comment"]) if "comment" in x 
        else (x["pool"]["method"],x["dataset"],x["results"]["statistic"])
        for x in data
    ]
    return data


def main(
        file:Annotated[str, typer.Argument()],
        pool:Annotated[Optional[str], typer.Option()] = None, 
        dataset:Annotated[Optional[str], typer.Option()] = None,
        comment:Annotated[Optional[str], typer.Option()] = None,
    ):
    # show the experiments results for given pooling method 
    records = _read_jsonstream(file)
    if dataset is not None:
        records = _filter_dataset(dataset, records)
    if pool is not None:
        records = _filter_pool(pool, records)
    if comment is not None:
        records = _filter_comment(comment, records)
    
    results = _read_statistic(records)
    [print(result) for result in results]

if __name__=="__main__":
    typer.run(main)