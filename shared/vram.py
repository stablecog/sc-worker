import torch
import logging
from tabulate import tabulate

from shared.constants import TabulateLevels


def get_device_vram_gb(device_id=0):
    if torch.cuda.device_count() > device_id:
        device_properties = torch.cuda.get_device_properties(device_id)
        total_memory = device_properties.total_memory
        total_memory_gb = total_memory / (1024**3)
        total_memory_gb_str = f"{total_memory_gb:.1f} GB"
        logging.info(
            tabulate(
                [["Total GPU Memory", total_memory_gb_str]],
                tablefmt=TabulateLevels.PRIMARY.value,
            )
        )
        return total_memory_gb
    else:
        return 0


device_vram_gb = get_device_vram_gb(0)
