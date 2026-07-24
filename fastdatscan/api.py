# -*- coding: utf-8 -*-
"""
fastdatscan.api -- the public interface.

Three levels, from highest to lowest:

    predict_image(...)      one image  -> one enhanced image
    predict_batch(...)      many images -> many enhanced images
    the building blocks     normalize / crop / restore, if you want to compose
                            your own pipeline

Everything else in the package is an implementation detail.
"""

import os
import glob as _glob

from . import _core


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
]

DEFAULT_PERCENTILE = 99.0
DEFAULT_CROP_THRESHOLD = 0.09
DEFAULT_CROP_MARGIN_MM = 10.0
DEFAULT_CRITERIA = "BestTrainMetricModel-inference.pth"


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def list_models(models_root):
    """Find every model directory under `models_root`.

    A model directory is any folder that directly contains `fold*` subfolders.
    Returns a list of dicts with keys: name, task, level, config, path.

    >>> list_models("/data/models")
    [{'name': 'BrainDaTscan-I123/3min/unet--light', 'task': ..., 'path': ...}]
    """
    return _core.build_catalog_from_models_dir(models_root)


def list_folds(model_directory):
    """Return the fold numbers available in a model directory, e.g. [0, 1, 2, 3, 4]."""
    return [_core._fold_number(d) for d in _core.list_fold_dirs(model_directory)]


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def normalize_to_percentile(input_url, output_url,
                            percentile=DEFAULT_PERCENTILE, clip=False):
    """Rescale to [0, 1] using min=0 and max=P`percentile` of the NON-ZERO voxels.

    Writes the unclipped image by default, which is what training used.
    """
    return _core.normalize_percentile(input_url, output_url,
                                      percentile=percentile, clip=clip,
                                      log=lambda *a: None)


def crop_to_body(input_url, output_url,
                 threshold=DEFAULT_CROP_THRESHOLD,
                 margin_mm=DEFAULT_CROP_MARGIN_MM,
                 n_largest=1):
    """Crop a NORMALIZED image to its body region.

    Thresholds the image, keeps the `n_largest` connected component(s), and crops
    to the bounding box with `margin_mm` of padding.

    Returns (output_url, crop_box).
    """
    return _core.crop_to_body(input_url, output_url, threshold=threshold,
                              margin_mm=margin_mm, n_largest=n_largest,
                              log=lambda *a: None)


def restore_geometry(prediction_url, reference_url, output_url,
                     default_value=0.0):
    """Resample a prediction made on a cropped volume back onto `reference_url`'s
    grid, so it overlays the original scan."""
    return _core.paste_into_reference(prediction_url, reference_url, output_url,
                                      default_value=default_value,
                                      log=lambda *a: None)


# --------------------------------------------------------------------------- #
# main entry points
# --------------------------------------------------------------------------- #
def predict_image(input_url,
                  model_directory,
                  output_dir,
                  folds="all",
                  device="auto",
                  normalize=True,
                  percentile=DEFAULT_PERCENTILE,
                  crop=True,
                  crop_threshold=DEFAULT_CROP_THRESHOLD,
                  crop_margin_mm=DEFAULT_CROP_MARGIN_MM,
                  restore_original_geometry=True,
                  use_autocast=True,
                  autocast_dtype="bfloat16",
                  sliding_overlap="from-model",
                  sw_batch_size=1,
                  decimal_places=4,
                  model_criteria=DEFAULT_CRITERIA,
                  progress_cb=None,
                  verbose=True):
    """Enhance ONE fast-acquisition image.

    Parameters
    ----------
    input_url : str
        Path to the input NIfTI (e.g. a 3-minute DaTscan).
    model_directory : str
        A model directory containing `fold*` subfolders, or a single .pth file.
    output_dir : str
        Where intermediate and final images are written.
    folds : "all" | int | list[int] | str
        Which folds to ensemble. `"all"`, `3` (first three), `[0, 2, 4]`,
        or `"0,2,4"`.
    device : "auto" | "cuda" | "cpu"
    normalize, percentile
        Percentile normalization (ON by default -- the models expect it).
    crop, crop_threshold, crop_margin_mm
        Body cropping (ON by default -- the models were trained on cropped data).
    restore_original_geometry : bool
        Resample the prediction back onto the input grid so it overlays the scan.
    use_autocast, autocast_dtype
        bfloat16 autocast on CUDA. Much faster, negligible numerical difference.
    progress_cb : callable(fold_index, fold_total, fraction, message) or None

    Returns
    -------
    str : path to the enhanced image.

    Examples
    --------
    >>> from fastdatscan import predict_image
    >>> predict_image("patient01.nii.gz",
    ...               "/data/models/BrainDaTscan-I123/3min/unet--light",
    ...               "/data/out")
    '/data/out/patient01--normalized--cropped_Ensemble--fullfov.nii.gz'
    """
    log = print if verbose else (lambda *a: None)
    mode = "file" if os.path.isfile(model_directory) else "dir"
    return _core.run(
        input_url=input_url,
        model_path=model_directory,
        output_dir=output_dir,
        mode=mode,
        device=device,
        sliding_overlap=sliding_overlap,
        model_criteria=model_criteria,
        folds=folds,
        sw_batch_size=sw_batch_size,
        decimal_places=decimal_places,
        normalize=normalize,
        percentile=percentile,
        crop_to_body_region=crop,
        crop_threshold=crop_threshold,
        crop_margin_mm=crop_margin_mm,
        restore_original_geometry=restore_original_geometry,
        use_autocast_inference=use_autocast,
        autocast_dtype=autocast_dtype,
        progress_cb=progress_cb,
        cancel_cb=None,
        log=log,
    )


def predict_batch(list_images,
                  model_directory,
                  output_dir,
                  skip_existing=True,
                  verbose=True,
                  **kwargs):
    """Enhance MANY images with the same model.

    `list_images` may be a list of paths or a glob pattern such as
    "/data/in/*.nii.gz".

    Returns a list of output paths (None for any image that failed, so one bad
    file does not abort the batch).
    """
    if isinstance(list_images, str):
        list_images = sorted(_glob.glob(list_images))
    log = print if verbose else (lambda *a: None)

    outputs = []
    total = len(list_images)
    for i, url in enumerate(list_images, 1):
        name = os.path.basename(url)
        log(f"[{i}/{total}] {name}")
        try:
            out = predict_image(url, model_directory, output_dir,
                                verbose=verbose, **kwargs)
            outputs.append(out)
        except Exception as exc:  # noqa: BLE001
            log(f"    FAILED: {type(exc).__name__}: {exc}")
            outputs.append(None)
    ok = sum(1 for o in outputs if o)
    log(f"\ndone: {ok}/{total} succeeded")
    return outputs
