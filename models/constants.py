DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"


def is_not_cuda(device: str) -> bool:
    return device.startswith(DEVICE_CUDA) is False
