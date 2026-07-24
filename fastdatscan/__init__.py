# -*- coding: utf-8 -*-
"""
fastdatscan -- deep-learning enhancement of fast-acquisition brain DaTscan SPECT.

Converts a short I123-ioflupane acquisition (e.g. 3 minutes) into an estimate of
the standard-duration scan (15 minutes), using the trained models published with
the Fast-SPECT-Imaging project.

Quick start
-----------
>>> from fastdatscan import predict_image
>>> predict_image("patient01.nii.gz", "/data/models/.../3min/unet--light", "/data/out")

Command line
------------
$ fastdatscan --input patient01.nii.gz --model-dir /data/models/... --output-dir out

The preprocessing the models expect (99th-percentile normalization and cropping
to the body region) is applied automatically; see `predict_image` to override it.
"""

__version__ = "0.1.0"
__author__ = "Yazdan Salimi"
__email__ = "salimiyazdan@gmail.com"
__url__ = "https://github.com/YazdanSalimi/Fast-SPECT-Imaging"

from .api import (  # noqa: F401
    predict_image,
    predict_batch,
    list_models,
    list_folds,
    normalize_to_percentile,
    crop_to_body,
    restore_geometry,
    DEFAULT_PERCENTILE,
    DEFAULT_CROP_THRESHOLD,
    DEFAULT_CROP_MARGIN_MM,
    DEFAULT_CRITERIA,
)

__all__ = [
    "predict_image",
    "predict_batch",
    "list_models",
    "list_folds",
    "normalize_to_percentile",
    "crop_to_body",
    "restore_geometry",
    "DEFAULT_PERCENTILE",
    "DEFAULT_CROP_THRESHOLD",
    "DEFAULT_CROP_MARGIN_MM",
    "DEFAULT_CRITERIA",
    "__version__",
]
