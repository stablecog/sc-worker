import torch
from shared.logger import logger
from tabulate import tabulate


def get_device_vram_gb(device_id=0):
    if torch.cuda.device_count() > device_id:
        device_properties = torch.cuda.get_device_properties(device_id)
        total_memory = device_properties.total_memory
        total_memory_gb = total_memory / (1024**3)
        total_memory_gb_str = f"{total_memory_gb:.1f} GB"
        logger.info(
            tabulate(
                [["Total GPU Memory", total_memory_gb_str]], tablefmt="double_grid"
            )
        )
        return total_memory_gb
    else:
        return 0


device_vram_gb = get_device_vram_gb(0)
