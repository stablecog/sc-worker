import torch


def get_device_vram_gb(device_id=0):
    if torch.cuda.device_count() > device_id:
        device_properties = torch.cuda.get_device_properties(device_id)
        total_memory = device_properties.total_memory
        total_memory_gb = total_memory / (1024**3)

        return total_memory_gb
    else:
        return 0


device_vram_gb = get_device_vram_gb(0)
