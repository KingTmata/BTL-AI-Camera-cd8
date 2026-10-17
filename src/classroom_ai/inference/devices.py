"""Thiết bị cho checkpoint PyTorch .pt: CPU hoặc GPU NVIDIA/CUDA."""

import re

import torch


def cuda_devices() -> tuple[dict[str, str], str]:
    """Kiểm cả runtime và một phép tính CUDA; lỗi GPU không chặn CPU."""
    if torch.version.cuda is None:
        return (
            {},
            "PyTorch hiện là bản CPU hoặc không có CUDA. Cài bản CUDA theo docs/guides/gpu_setup.md.",
        )
    devices, failures = {}, []
    try:
        if not torch.cuda.is_available():
            return {}, "CUDA không khả dụng: kiểm tra GPU NVIDIA và driver. Ứng dụng vẫn dùng CPU."
        for index in range(torch.cuda.device_count()):
            device = f"cuda:{index}"
            try:
                name = torch.cuda.get_device_name(index)
                # .item() đợi kernel hoàn tất, phát hiện driver/kiến trúc không tương thích.
                torch.ones(1, device=device).sum().item()
                devices[device] = f"GPU {index} · {name} (CUDA)"
            except Exception as exc:
                failures.append(f"{device}: {exc}")
    except Exception as exc:
        failures.append(str(exc))
    reason = "; ".join(failures)
    if not devices and not reason:
        reason = "Không tìm thấy GPU CUDA. Ứng dụng vẫn dùng CPU."
    return devices, reason


def validate_device(device: str) -> str:
    if device == "cpu":
        return device
    if not isinstance(device, str) or re.fullmatch(r"cuda:\d+", device) is None:
        raise ValueError("Thiết bị phải là cpu hoặc cuda:<chỉ số GPU>.")
    devices, reason = cuda_devices()
    if device not in devices:
        raise ValueError(
            f"Không dùng được {device}: {reason or 'GPU không tồn tại'}. Hãy chọn CPU."
        )
    return device
