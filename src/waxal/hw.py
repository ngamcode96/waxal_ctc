"""GPU capability checks.

torch.cuda.is_bf16_supported() returns True on a Tesla T4 (sm_75), which has no
native bf16 -- recent PyTorch counts *emulated* bf16 as supported. Training in
emulated bf16 is far slower than the fp16 tensor cores the card actually has, so
selecting precision from that check silently cripples throughput on pre-Ampere
hardware. Gate on compute capability instead: bf16 is native from sm_80 (Ampere).
"""

import torch


def supports_bf16() -> bool:
    """True only where bf16 runs on hardware, not through emulation."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def describe() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability()
    native = supports_bf16()
    emulated = torch.cuda.is_bf16_supported() and not native
    note = " (bf16 emulated -- using fp16 instead)" if emulated else ""
    return f"{name} sm_{major}{minor} bf16_native={native}{note}"
