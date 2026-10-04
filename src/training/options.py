"""Validated advanced options for new local training runs."""

import math

ADVANCED_DEFAULTS = {
    "mosaic": 1.0, "mixup": 0.0, "cutmix": 0.0, "degrees": 0.0,
    "fliplr": 0.5, "flipud": 0.0, "close_mosaic": 10,
    "cls_pw": 0.0, "box": 7.5, "cls": 0.5, "dfl": 1.5,
}


def validate_advanced(options):
    if not isinstance(options, dict) or options.keys() - ADVANCED_DEFAULTS.keys():
        raise ValueError("Tham số Nâng cao không được hỗ trợ.")
    values = {**ADVANCED_DEFAULTS, **options}
    for name, value in values.items():
        if name == "close_mosaic":
            if type(value) is not int or not 0 <= value <= 1000:
                raise ValueError("close_mosaic phải là số nguyên từ 0 đến 1000.")
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} phải là số hữu hạn không âm.")
        if name in {"mosaic", "mixup", "cutmix", "fliplr", "flipud", "cls_pw"} and value > 1:
            raise ValueError(f"{name} chỉ nhận giá trị từ 0.0 đến 1.0.")
        if name == "degrees" and value > 180:
            raise ValueError("degrees chỉ nhận giá trị từ 0 đến 180 độ.")
        values[name] = float(value)
    return values


def mosaic_active_epochs(epochs, close_mosaic):
    """Count epochs before Ultralytics closes all four mixing augmentations."""
    return epochs if close_mosaic == 0 else max(0, epochs - close_mosaic)
