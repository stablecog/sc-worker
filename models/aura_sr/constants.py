from shared.vram import device_vram_gb

AURA_SR_MODEL_ID = "fal/AuraSR-v2"
AURA_SR_MODEL_NAME = "AuraSR"
AURA_SR_KEEP_IN_CPU_WHEN_IDLE = device_vram_gb < 40
