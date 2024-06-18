import os
import sys
import torch
from rich import print as rprint

def print_expr_info(
        conf: dict, 
        device: torch.device, 
        file=sys.stderr
):
    #print the information of experiments.
    
    device_property = torch.cuda.get_device_properties(device)
    
    info_str = f"{sep_c('=')}\nExperiments setting:\n{conf['experiment']}\n{sep_c('-')}\n"\
    + f"Device properties:\n{device_property}\n{sep_c('-')}\n"\
    + f"Pooling setting:\n{conf['pool']}\n{sep_c('-')}\n"\
    + f"Dataset:\n[green]{conf['dataset']}[/green]\n{sep_c('-')}\n"\
    + f"Model configuration:\n{conf['model']}\n{sep_c('=')}"

    rprint(info_str, file=file)


def sep_c(
        sep:chr, 
        ratio:float=0.8
) -> int:
    # generate separetor which fits the console width
    
    w = int(ratio * os.get_terminal_size().columns)
    return w*sep