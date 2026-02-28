import os
import sys
import platform
from datetime import datetime, timezone
import torch
from rich import print as rprint

def print_expr_info(
        conf: dict, 
        device: torch.device, 
        file=sys.stderr
):
    #print the information of experiments.
    if device.type == "cuda" and torch.cuda.is_available():
        device_property = torch.cuda.get_device_properties(device)
    else:
        device_property = f"CPU({platform.processor() or 'unknown'})"
    
    info_str = f"{sep_c('=')}\nExperiments setting:\n{conf['experiment']}\n{sep_c('-')}\n"\
    + f"Device properties:\n{device_property}\n{sep_c('-')}\n"\
    + f"Pooling setting:\n{conf['pool']}\n{sep_c('-')}\n"\
    + f"Dataset:\n[green]{conf['dataset']}[/green]\n{sep_c('-')}\n"\
    + f"Model configuration:\n{conf['model']}\n{sep_c('=')}"

    rprint(info_str, file=file)


def build_runtime_meta(device: torch.device) -> dict:
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device": str(device),
    }
    return meta


def sep_c(
        sep:chr, 
        ratio:float=0.8
) -> int:
    # generate separetor which fits the console width

    try:
        columns = os.get_terminal_size().columns
    except OSError:
        columns = 120
    w = int(ratio * columns)
    return w*sep
