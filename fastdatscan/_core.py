# -*- coding: utf-8 -*-
"""
fastdatscan._core -- preprocessing + inference primitives.

This module is deliberately free of any 3D Slicer dependency: it only needs
torch, monai, nibabel and SimpleITK, so the exact same code runs from a script,
a notebook, a batch job, or inside Slicer.

The public API lives in `fastdatscan.api`; this module holds the implementation.

Pipeline (matching how the models were trained):

    normalize to the 99th percentile  ->  crop to the body region
    ->  sliding-window inference (optionally ensembled over folds)
    ->  paste the prediction back into the original image geometry
"""

import os
import re
import glob
import json


# --------------------------------------------------------------------------- #
# small filesystem helpers
# --------------------------------------------------------------------------- #
def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "model"


def _fs_name(x):
    """Folder-name-safe but human-readable (keeps % and spaces)."""
    return re.sub(r'[<>:"/\\|?*\r\n\t]+', "_", str(x)).strip() or "_"


META_NAME = "_fastspect.json"
DEFAULT_CRITERIA = "BestTrainMetricModel-inference.pth"
_ROOT_ALIASES = ("model-directory", "models", "model_directory", "data")


class InferenceCancelled(Exception):
    """Raised inside the worker when the user cancels."""


# --------------------------------------------------------------------------- #
# fold-directory helpers
# --------------------------------------------------------------------------- #
def _fold_number(fold_dir):
    """Extract the integer after 'fold--' (or trailing digits) from a path."""
    base = os.path.basename(fold_dir.rstrip("/\\"))
    m = re.search(r"fold\D*?(\d+)", base, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", base)
    return int(m.group(1)) if m else None


def list_fold_dirs(model_directory):
    """Return all fold* subdirectories, naturally sorted by fold number."""
    if not model_directory or not os.path.isdir(model_directory):
        return []
    fold_dirs = [d for d in glob.glob(os.path.join(model_directory, "fold*"))
                 if os.path.isdir(d)]

    def key(d):
        n = _fold_number(d)
        return (0, n) if n is not None else (1, os.path.basename(d))

    return sorted(fold_dirs, key=key)


def resolve_folds(fold_dirs, folds):
    """
    folds may be:
      * 'all'                          -> every fold dir
      * an int N                       -> the first N fold dirs (natural order)
      * a list/tuple/set of ints       -> fold dirs whose number is in the set
      * a string 'all' / '5' / '1,2,3' -> parsed as above
    Returns the filtered, ordered list of fold dirs.
    """
    if folds is None:
        return fold_dirs

    if isinstance(folds, str):
        s = folds.strip().lower()
        if s in ("", "all"):
            return fold_dirs
        if "," in s:
            folds = [int(x) for x in s.split(",") if x.strip() != ""]
        else:
            folds = int(s)  # bare number -> "first N"

    if folds == "all":
        return fold_dirs
    if isinstance(folds, int):
        return fold_dirs[:max(0, folds)]
    if isinstance(folds, (list, tuple, set)):
        wanted = set(int(x) for x in folds)
        return [fd for fd in fold_dirs if _fold_number(fd) in wanted]
    return fold_dirs


# --------------------------------------------------------------------------- #
# model directory discovery (local only, no download)
# structure expected:  <models>/[wrapper/]<task>/<level>/<config>/fold*/…*.pth
# a ModelDirectory is any folder that directly contains at least one fold* dir.
# --------------------------------------------------------------------------- #
def _bagit_payload_root(root):
    """If `root` is a BagIt bag (has data/ + bagit.txt), return root/data."""
    data_dir = os.path.join(root, "data")
    if os.path.isdir(data_dir) and (
            os.path.isfile(os.path.join(root, "bagit.txt")) or os.listdir(data_dir)):
        return data_dir
    return root


def find_model_dirs(root):
    """Return [(name, path), ...] for every directory under `root` that is a
    ModelDirectory (directly contains at least one fold* subfolder).
    If none is found but `root` itself has fold*, returns [(basename, root)]."""
    found = []
    if not root or not os.path.isdir(root):
        return found
    root = _bagit_payload_root(root)
    if list_fold_dirs(root):
        found.append((os.path.basename(root.rstrip("/\\")) or "model", root))
    for dp, dns, _ in os.walk(root):
        for d in dns:
            full = os.path.join(dp, d)
            if list_fold_dirs(full):
                name = os.path.relpath(full, root).replace(os.sep, "/")
                found.append((name, full))
    seen, uniq = set(), []
    for name, path in found:
        ap = os.path.abspath(path)
        if ap not in seen:
            seen.add(ap)
            uniq.append((name, path))
    return uniq


def _read_meta_upwards(model_dir, stop_root):
    """Look for _fastspect.json in model_dir and its ancestors up to stop_root."""
    cur = os.path.abspath(model_dir)
    stop = os.path.abspath(stop_root) if stop_root else None
    for _ in range(8):
        p = os.path.join(cur, META_NAME)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return None
        parent = os.path.dirname(cur)
        if parent == cur or (stop and cur == stop):
            break
        cur = parent
    return None


def build_catalog_from_tree(root, criteria=DEFAULT_CRITERIA, meta_root=None):
    """Scan for local models. Uses each model's _fastspect.json when present
    (exact task/level/config); otherwise infers from a <task>/<level>/<config>
    folder tree. A model with fewer than three classifying path parts and no
    metadata is still listed with best-effort labels so nothing is silently
    hidden from the user's own Models folder."""
    meta_root = meta_root or root
    out = []
    for name, path in find_model_dirs(root):
        meta = _read_meta_upwards(path, meta_root)
        if meta:
            task = meta.get("task", meta.get("training_data", ""))
            level = meta.get("level", meta.get("dose", ""))
            config = meta.get("config", "")
            crit = meta.get("criteria", criteria)
        else:
            parts = [p for p in name.replace("\\", "/").split("/") if p]
            while parts and parts[0].lower() in _ROOT_ALIASES:
                parts = parts[1:]
            if len(parts) >= 3:
                task, level, config = parts[-3], parts[-2], parts[-1]
            elif len(parts) == 2:
                task, level, config = parts[-2], parts[-1], "default"
            elif len(parts) == 1:
                task, level, config = parts[-1], "default", "default"
            else:
                task, level, config = "model", "default", "default"
            crit = criteria
        out.append({"task": task, "level": level, "config": config,
                    "path": path, "criteria": crit, "type": "dir",
                    "name": f"{task}/{level}/{config}"})
    return out


def build_catalog_from_models_dir(models_dir, criteria=DEFAULT_CRITERIA):
    """Scan the whole (local) Models folder for models the user downloaded."""
    if not models_dir or not os.path.isdir(models_dir):
        return []
    return build_catalog_from_tree(models_dir, criteria=criteria, meta_root=models_dir)


def catalog_values(catalog, field, task=None, level=None):
    seen = []
    for e in catalog:
        if task is not None and e.get("task") != task:
            continue
        if level is not None and e.get("level") != level:
            continue
        v = e.get(field)
        if v is not None and v not in seen:
            seen.append(v)
    return seen


def find_catalog_entry(catalog, task, level, config):
    for e in catalog:
        if (e.get("task") == task and e.get("level") == level
                and e.get("config") == config):
            return e
    return None


# --------------------------------------------------------------------------- #
# device helper: resolve 'auto'/'cuda'/'cpu' against what torch actually has
# --------------------------------------------------------------------------- #
def resolve_device(device, log=print):
    import torch
    dev = str(device).strip().lower()
    cuda_ok = torch.cuda.is_available()
    if dev == "auto":
        chosen = "cuda" if cuda_ok else "cpu"
        log(f"device=auto -> {chosen}" + ("" if cuda_ok else " (no CUDA in this torch build)"))
        return chosen
    if dev.startswith("cuda") and not cuda_ok:
        log("CUDA requested but this Slicer's torch has no CUDA support; falling back to CPU.")
        return "cpu"
    return dev


# --------------------------------------------------------------------------- #
# copied: Fast_SPECT image helpers (CopyInfo / percentile / rescale / match)
# --------------------------------------------------------------------------- #
def copy_info(reference_image, updating_image, origin=True, spacing=True, direction=True):
    import SimpleITK as sitk
    if isinstance(reference_image, str):
        reference_image = sitk.ReadImage(reference_image)
    if isinstance(updating_image, str):
        updating_image = sitk.ReadImage(updating_image)
    if origin:
        updating_image.SetOrigin(reference_image.GetOrigin())
    if spacing:
        updating_image.SetSpacing(reference_image.GetSpacing())
    if direction:
        updating_image.SetDirection(reference_image.GetDirection())
    return updating_image


def sitk_percentile(image, percentile=99.0, segment=None):
    """(non_zero_percentile, with_zero_percentile) of an image, optionally
    restricted to a segmentation mask. Faithful port of the repo helper."""
    import SimpleITK as sitk
    import numpy as np
    if isinstance(image, str):
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(image))
    elif isinstance(image, sitk.Image):
        image_array = sitk.GetArrayFromImage(image)
    elif isinstance(image, np.ndarray):
        image_array = image
    else:
        raise TypeError("image must be a path (str), sitk.Image, or numpy array.")

    if segment is not None and not (isinstance(segment, str) and segment == "none"):
        if isinstance(segment, str):
            segment_array = sitk.GetArrayFromImage(sitk.ReadImage(segment))
        elif isinstance(segment, sitk.Image):
            segment_array = sitk.GetArrayFromImage(segment)
        elif isinstance(segment, np.ndarray):
            segment_array = segment
        else:
            raise TypeError("segment must be a path (str), sitk.Image, or numpy array.")
        image_array = image_array[segment_array != 0]

    non_zero = image_array[image_array != 0]
    if non_zero.size == 0:
        non_zero_percentile = np.percentile(image_array, percentile)
    else:
        non_zero_percentile = np.percentile(non_zero, percentile)
    with_zero_percentile = np.percentile(image_array, percentile)
    return float(non_zero_percentile), float(with_zero_percentile)


def sitk_rescale(image, input_min="image-min", input_max="image-max",
                 output_min=0.0, output_max=1.0):
    """Rescale intensities to [output_min, output_max]. Returns (clipped,
    unclipped). Faithful port of the repo helper."""
    import SimpleITK as sitk
    import numpy as np
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)

    if input_min == "image-min":
        input_min = float(np.min(array))
    if input_max == "image-max":
        input_max = float(np.max(array))

    denom = (input_max - input_min)
    if denom == 0:
        denom = 1.0
    scale = (output_max - output_min) / denom
    scaled = (array - input_min) * scale + output_min

    image_no_clip = sitk.GetImageFromArray(scaled)
    image_no_clip = copy_info(image, image_no_clip)

    image_clip = sitk.GetImageFromArray(np.clip(scaled, output_min, output_max))
    image_clip = copy_info(image, image_clip)
    return image_clip, image_no_clip


def normalize_percentile(input_url, output_url, percentile=99.0, segment=None,
                         clip=False, log=print):
    """Normalize an image to [0, 1] with min = 0 and max = the (non-zero)
    `percentile` of the image, and write it to `output_url`.

    This mirrors Fast-SPECT-Imaging/preprocessing.py, which uses
    sitk_rescale(input_min=0, input_max=sitk_percentile(...)[0])[unclipped].
    By default the UNCLIPPED image is written (as in the repo); set clip=True
    to write the clipped [0, 1] version instead.
    Returns output_url.
    """
    import SimpleITK as sitk
    p_nonzero, _ = sitk_percentile(input_url, percentile=percentile, segment=segment)
    log(f"    normalizing: min=0  max=P{percentile:g}(non-zero)={p_nonzero:.6g}")
    clipped, unclipped = sitk_rescale(input_url, input_min=0, input_max=p_nonzero,
                                      output_min=0.0, output_max=1.0)
    out_image = clipped if clip else unclipped
    os.makedirs(os.path.dirname(os.path.abspath(output_url)), exist_ok=True)
    sitk.WriteImage(out_image, output_url)
    return output_url


def match_space(input_image, reference_image, interpolate="linear", default_pixel_value=0):
    import SimpleITK as sitk
    if isinstance(input_image, str):
        input_image = sitk.ReadImage(input_image)
    if isinstance(reference_image, str):
        reference_image = sitk.ReadImage(reference_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetDefaultPixelValue(default_pixel_value)
    if interpolate == "nearest":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolate == "linear":
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolate.lower() == "bspline":
        resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(input_image)


def compress_image(url, target_url="none", decimal_places=4, cast_to_int16=True):
    import SimpleITK as sitk
    if isinstance(url, str):
        image = sitk.ReadImage(url, sitk.sitkFloat32)
    else:
        image = sitk.Cast(url, sitk.sitkFloat32)
    image_compressed = sitk.Round(image * 10 ** decimal_places) / 10 ** decimal_places
    if decimal_places == 0 and cast_to_int16:
        image_compressed = sitk.Cast(image_compressed, sitk.sitkInt16)
    if target_url != "none":
        sitk.WriteImage(image_compressed, target_url)
    return image_compressed


def ensemble_image_regression(list_inference_models, target_url="none"):
    import SimpleITK as sitk
    image_ensemble = sitk.ReadImage(list_inference_models[0])
    for url in list_inference_models[1:]:
        try:
            image_ensemble += sitk.ReadImage(url)
        except Exception:
            matched = match_space(input_image=url, reference_image=image_ensemble)
            image_ensemble += sitk.Cast(matched, image_ensemble.GetPixelID())
    image_ensemble = image_ensemble / len(list_inference_models)
    if target_url != "none":
        sitk.WriteImage(image_ensemble, target_url)
    return image_ensemble


# --------------------------------------------------------------------------- #
# sliding-window count estimate (for progress)
# --------------------------------------------------------------------------- #
def crop_image_to_segment(image, segment, crop_dims="all", margin_mm=0,
                          lowerThreshold=0.1, upperThreshold=0.9,
                          insideValue=0, outsideValue=1,
                          force_match=False, orientation=None):
    """Faithful port of yazdan.image.crop_image_to_segment.

    Returns (image_cropped, segment_cropped, segment_non_binary_cropped, crop_box).
    `crop_box` carries the physical points / indices / sizes needed to place the
    prediction back into the original volume.

    NOTE the threshold defaults: insideValue=0 / outsideValue=1 means the
    BinaryThreshold marks voxels OUTSIDE [lowerThreshold, upperThreshold] as 1,
    and the bounding box is taken from that label. This mirrors the training
    code exactly -- do not "fix" it.
    """
    import SimpleITK as sitk

    if isinstance(image, str):
        image = sitk.ReadImage(image)
    if orientation is not None:
        image = sitk.DICOMOrient(image, orientation)
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
    if orientation is not None:
        segment = sitk.DICOMOrient(segment, orientation)
    if force_match:
        segment = match_space(input_image=segment, reference_image=image,
                              interpolate="nearest")

    segment = sitk.Cast(segment, sitk.sitkUInt8)
    segment_non_binary = segment
    segment = sitk.BinaryThreshold(segment, lowerThreshold=lowerThreshold,
                                   upperThreshold=upperThreshold,
                                   insideValue=insideValue,
                                   outsideValue=outsideValue)

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(segment)
    bounding_box = label_shape_filter.GetBoundingBox(1)

    half = int(len(bounding_box) / 2)
    start_physical_point = segment.TransformIndexToPhysicalPoint(bounding_box[0:half])
    end_physical_point = segment.TransformIndexToPhysicalPoint(
        [x + sz for x, sz in zip(bounding_box[0:half], bounding_box[half:])])

    start_physical_point = [x - margin_mm for x in start_physical_point]
    end_physical_point = [x + margin_mm for x in end_physical_point]

    image_crop_start_indices = image.TransformPhysicalPointToIndex(start_physical_point)
    image_crop_end_indices = image.TransformPhysicalPointToIndex(end_physical_point)
    segment_crop_start_indices = segment.TransformPhysicalPointToIndex(start_physical_point)
    segment_crop_end_indices = segment.TransformPhysicalPointToIndex(end_physical_point)

    image_crop_sizes = [a - b for a, b in zip(image_crop_end_indices,
                                              image_crop_start_indices)]
    segment_crop_sizes = [a - b for a, b in zip(segment_crop_end_indices,
                                                segment_crop_start_indices)]

    image_crop_start_indices = list(image_crop_start_indices)
    for d, v in enumerate(image_crop_start_indices):
        if v < 0:
            image_crop_start_indices[d] = 0
    image_crop_sizes = list(image_crop_sizes)
    for d, v in enumerate(image_crop_sizes):
        if v + image_crop_start_indices[d] > image.GetSize()[d]:
            image_crop_sizes[d] = image.GetSize()[d] - image_crop_start_indices[d] - 1

    segment_crop_start_indices = list(segment_crop_start_indices)
    for d, v in enumerate(segment_crop_start_indices):
        if v < 0:
            segment_crop_start_indices[d] = 0
    segment_crop_sizes = list(segment_crop_sizes)
    for d, v in enumerate(segment_crop_sizes):
        if v + segment_crop_start_indices[d] > segment.GetSize()[d]:
            segment_crop_sizes[d] = segment.GetSize()[d] - segment_crop_start_indices[d] - 1

    if crop_dims != "all":
        no_crop_dims = [x for x in [0, 1, 2] if x not in crop_dims]
        for d in no_crop_dims:
            image_crop_start_indices[d] = 0
            image_crop_sizes[d] = image.GetSize()[d]

    image_cropped = sitk.RegionOfInterest(image, image_crop_sizes,
                                          image_crop_start_indices)
    segment_cropped = sitk.RegionOfInterest(segment, segment_crop_sizes,
                                            segment_crop_start_indices)
    segment_non_binary_cropped = sitk.RegionOfInterest(
        segment_non_binary, segment_crop_sizes, segment_crop_start_indices)

    crop_box_out = {
        "start_physical_point": start_physical_point,
        "end_physical_point": end_physical_point,
        "crop_start_indices": image_crop_start_indices,
        "crop_end_indices": image_crop_end_indices,
        "crop_sizes": image_crop_sizes,
    }
    return image_cropped, segment_cropped, segment_non_binary_cropped, crop_box_out


def crop_to_body(input_url, output_url, threshold=0.09, margin_mm=10,
                 n_largest=1, log=print):
    """Crop a NORMALIZED image to its body region, as the training pipeline does:

        body      = image > threshold
        components= ConnectedComponent -> RelabelComponent(sortByObjectSize)
        largest   = sum of the n largest components
        cropped   = crop_image_to_segment(image, largest, margin_mm=margin_mm)

    Returns (output_url, crop_box). The crop box lets us paste the prediction
    back into the original geometry afterwards.
    """
    import SimpleITK as sitk

    image = sitk.ReadImage(input_url)
    body_segment = sitk.Cast(image > threshold, sitk.sitkUInt8)

    component_image = sitk.ConnectedComponent(body_segment)
    sorted_component_image = sitk.RelabelComponent(component_image,
                                                   sortByObjectSize=True)
    largest = sum([sorted_component_image == label
                   for label in range(1, n_largest + 1)])

    image_cropped, _seg, _segnb, crop_box = crop_image_to_segment(
        image=image, segment=largest, margin_mm=margin_mm)

    log(f"    cropping to body: {image.GetSize()} -> {image_cropped.GetSize()} "
        f"(threshold={threshold}, margin={margin_mm}mm)")
    os.makedirs(os.path.dirname(os.path.abspath(output_url)), exist_ok=True)
    sitk.WriteImage(image_cropped, output_url)
    return output_url, crop_box


def paste_into_reference(cropped_url, reference_url, output_url,
                         default_value=0.0, log=print):
    """Put a prediction made on a cropped volume back into the reference
    (uncropped) geometry, so the result overlays the original scan in Slicer.

    Uses physical coordinates via Resample, so it is correct regardless of how
    the crop indices were clipped at the image borders.
    """
    import SimpleITK as sitk

    pred = sitk.ReadImage(cropped_url)
    ref = sitk.ReadImage(reference_url)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetTransform(sitk.Transform())      # identity: physical alignment
    out = resampler.Execute(sitk.Cast(pred, sitk.sitkFloat32))

    os.makedirs(os.path.dirname(os.path.abspath(output_url)), exist_ok=True)
    sitk.WriteImage(out, output_url)
    log(f"    pasted prediction back into original geometry {ref.GetSize()}")
    return output_url


def compute_num_windows(image_size, roi_size, overlap):
    import math
    if isinstance(overlap, (int, float)):
        overlap = [overlap] * len(roi_size)
    n = 1
    for isz, rsz, ov in zip(image_size, roi_size, overlap):
        rsz = min(rsz, isz)
        interval = max(1, int(round(rsz * (1 - ov))))
        steps = 1 if isz <= rsz else int(math.ceil((isz - rsz) / interval)) + 1
        n *= max(1, steps)
    return n


# --------------------------------------------------------------------------- #
# numpy RNG pickle patch (checkpoints may pickle a numpy Generator)
# --------------------------------------------------------------------------- #
def _patch_numpy_rng():
    try:
        import numpy as np
        import numpy.random._pickle
        numpy.random._pickle.BitGenerators['MT19937'] = np.random.MT19937
        numpy.random._pickle.BitGenerators[np.random.MT19937] = np.random.MT19937
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# tolerant unpickler: survives stray imports pickled into the checkpoint
# --------------------------------------------------------------------------- #
def _make_dummy():
    class _Meta(type):
        def __getattr__(cls, _n): return cls
        def __call__(cls, *a, **k): return cls.__new__(cls)
        def __float__(cls): return 0.0
        def __int__(cls): return 0
        def __str__(cls): return ""
        def __iter__(cls): return iter(())
        def __getitem__(cls, _k): return cls

    class _Tolerant(metaclass=_Meta):
        def __init__(self, *a, **k): pass
        def __getattr__(self, _n): return _Tolerant
        def __call__(self, *a, **k): return _Tolerant()
        def __getitem__(self, _k): return _Tolerant
        def __iter__(self): return iter(())
        def __float__(self): return 0.0
        def __int__(self): return 0
        def __str__(self): return ""
        def __bool__(self): return False
        def __setstate__(self, _s): pass
    return _Tolerant


_RNG_PATCH_LOCK = __import__("threading").Lock()


_IMPORT_CHECK_CACHE = {}


def _find_missing_modules_from_path(pth_path):
    """Same as _find_missing_modules but reads the pickle streams straight from
    the file, so we never hold the whole checkpoint in memory just to scan it."""
    with open(pth_path, "rb") as fh:
        head = fh.read(4)
        fh.seek(0)
        # torch .pth are ZIPs ('PK\x03\x04'); read only the small data.pkl member
        if head[:2] == b"PK":
            import zipfile
            with zipfile.ZipFile(fh) as zf:
                names = [n for n in zf.namelist()
                         if n.endswith(".pkl") or os.path.basename(n) == "data.pkl"]
                blobs = [zf.read(n) for n in names]
            return _find_missing_modules(b"", _streams=blobs)
        return _find_missing_modules(fh.read())


def _find_missing_modules(raw, _streams=None):
    """Statically read the pickle streams inside a .pth (torch files are ZIPs
    holding data.pkl; legacy ones are one raw stream) and return the set of
    referenced top-level-resolvable module names that are NOT importable.

    No code is executed -- we only walk pickle opcodes. This lets us stub every
    missing module in one go, so torch.load does not have to be retried once per
    missing module (each retry re-parses the entire stream).
    """
    import io as _io
    import sys as _sys
    import zipfile
    import pickletools
    import importlib

    def _streams_from(buf):
        bio = _io.BytesIO(buf)
        if zipfile.is_zipfile(bio):
            bio.seek(0)
            with zipfile.ZipFile(bio) as zf:
                for nm in zf.namelist():
                    if nm.endswith(".pkl") or os.path.basename(nm) == "data.pkl":
                        yield zf.read(nm)
        else:
            yield buf

    referenced = set()
    stream_iter = _streams if _streams is not None else _streams_from(raw)
    for stream in stream_iter:
        recent = []
        try:
            for op, arg, _pos in pickletools.genops(_io.BytesIO(stream)):
                nm = op.name
                if nm in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8",
                          "SHORT_BINSTRING", "BINSTRING", "UNICODE", "STRING"):
                    if isinstance(arg, (bytes, bytearray)):
                        try:
                            arg = arg.decode("utf-8", "replace")
                        except Exception:
                            continue
                    recent.append(arg)
                    if len(recent) > 4:
                        recent.pop(0)
                elif nm == "GLOBAL":
                    referenced.add(str(arg).split(" ", 1)[0])
                elif nm == "STACK_GLOBAL" and len(recent) >= 2:
                    referenced.add(recent[-2])
        except Exception:
            continue  # truncated/odd stream: use whatever we collected

    # Modules the unpickler resolves itself (py2 aliases etc.) -- stubbing these
    # would shadow torch's own compatibility handling, so never touch them.
    _NEVER_STUB = {"__builtin__", "__builtins__", "copy_reg", "copyreg",
                   "exceptions", "StringIO", "cStringIO"}

    # Deciding "is this importable?" must NOT touch the filesystem.
    #
    # importlib.util.find_spec() walks every sys.path entry for a miss. On a
    # Windows install with ~30 path entries (including a .zip) and a checkpoint
    # referencing a dozen private submodules, that first call cost 60s+ of
    # completely silent stalling -- which looked exactly like a hang with no CPU
    # and no GPU. We only need TOP-LEVEL packages: importing a parent is enough
    # to know whether the whole dotted path can resolve, and the result is
    # cached in sys.modules by the interpreter.
    tops = {}
    for mod in referenced:
        if not mod or not all(part.isidentifier() for part in mod.split(".")):
            continue
        if mod in _NEVER_STUB or mod.split(".")[0] in _NEVER_STUB:
            continue
        tops.setdefault(mod.split(".")[0], []).append(mod)

    missing = set()
    for top, dotted in tops.items():
        if top in _sys.modules:
            continue
        ok = _IMPORT_CHECK_CACHE.get(top)
        if ok is None:
            try:
                importlib.import_module(top)
                ok = True
            except Exception:
                ok = False
            _IMPORT_CHECK_CACHE[top] = ok
        if not ok:
            # the top-level package is unavailable -> every name under it is too
            missing.update(dotted)
            missing.add(top)
    return sorted(missing)


def _safe_torch_load(load_url, log=print, map_location="cpu", _tries=80):
    """Load a checkpoint.

    FAST PATH (the normal case): if every module the pickle references is
    importable, we call torch.load directly -- no stubbing, no numpy patching,
    no buffering. This must happen BEFORE any global state is touched: the
    compatibility machinery below swaps out numpy's RNG constructors process
    wide, which measurably slows and perturbs an otherwise ordinary load.

    SLOW PATH: only when modules are genuinely missing, so the checkpoint cannot
    be unpickled without shims.
    """
    import sys
    import types
    import io
    import time
    import torch

    # ---------------- FAST PATH: nothing missing -> plain torch.load ----------
    t0 = time.perf_counter()
    log("    checking module availability…")
    try:
        missing = _find_missing_modules_from_path(load_url)
    except Exception as _exc:
        log(f"    (module check skipped: {_exc})")
        missing = []
    log(f"    [time] module check: {time.perf_counter() - t0:.2f}s")

    if not missing:
        t0 = time.perf_counter()
        out = torch.load(load_url, map_location=map_location, weights_only=False)
        log(f"    [time] torch.load: {time.perf_counter() - t0:.2f}s "
            f"(direct, no patching needed)")
        return out

    # ---------------- SLOW PATH: shims required -------------------------------
    log(f"    {len(missing)} module(s) not importable "
        f"({', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}); "
        f"loading with compatibility shims.")

    Dummy = _make_dummy()
    touched = {}

    def stub(name):
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            n = ".".join(parts[:i])
            if n not in touched:
                touched[n] = sys.modules.get(n)
            m = types.ModuleType(n)
            m.__path__ = []
            m.__getattr__ = lambda _n: Dummy
            sys.modules[n] = m

    for name in missing:
        stub(name)

    # Neutralize numpy's RNG constructors so a pickled Generator / BitGenerator /
    # RandomState does not force a retry. Serialized, because this mutates global
    # state and may run on a worker thread.
    _RNG_PATCH_LOCK.acquire()
    _rng_saved = {}
    try:
        import numpy.random._pickle as _nrp

        class _IgnoredRNG:
            def __init__(self, label="rng"):
                self.label = label

            def __reduce__(self):
                return (_IgnoredRNG, (getattr(self, "label", "rng"),))

            def __setstate__(self, _s):
                pass

        for _n in ("__bit_generator_ctor", "__generator_ctor", "__randomstate_ctor"):
            _rng_saved[_n] = getattr(_nrp, _n, None)
        if hasattr(_nrp, "__bit_generator_ctor"):
            setattr(_nrp, "__bit_generator_ctor",
                    lambda bit_generator_name="MT19937": _IgnoredRNG("BitGenerator"))
        if hasattr(_nrp, "__generator_ctor"):
            setattr(_nrp, "__generator_ctor",
                    lambda bit_generator_name="MT19937", bit_generator_ctor=None:
                    _IgnoredRNG("Generator"))
        if hasattr(_nrp, "__randomstate_ctor"):
            setattr(_nrp, "__randomstate_ctor",
                    lambda bit_generator_name="MT19937", bit_generator_ctor=None:
                    _IgnoredRNG("RandomState"))
    except Exception:
        _nrp = None

    t0 = time.perf_counter()
    with open(load_url, "rb") as fh:
        raw = fh.read()
    log(f"    [time] read file: {time.perf_counter() - t0:.2f}s "
        f"({len(raw)/1e6:.0f} MB)")


    retries = 0
    try:
        for _ in range(_tries):
            try:
                return torch.load(io.BytesIO(raw), map_location=map_location,
                                  weights_only=False)
            except (ModuleNotFoundError, ImportError) as exc:
                name = getattr(exc, "name", None)
                if not name:
                    raise
                retries += 1
                log(f"    stubbing '{name}' (retry {retries})")
                stub(name)
            except AttributeError as exc:
                # a previously stubbed module was asked for an attribute that our
                # placeholder did not provide; make the stub tolerant and retry.
                msg = str(exc)
                m = re.search(r"module '([\w\.]+)' has no attribute", msg)
                if not m:
                    raise
                retries += 1
                log(f"    re-stubbing '{m.group(1)}' (retry {retries})")
                stub(m.group(1))
        return torch.load(io.BytesIO(raw), map_location=map_location,
                          weights_only=False)
    finally:
        if _nrp is not None:
            for _n, _orig in _rng_saved.items():
                if _orig is not None:
                    setattr(_nrp, _n, _orig)
        try:
            _RNG_PATCH_LOCK.release()
        except Exception:
            pass
        for n, orig in touched.items():
            if orig is not None:
                sys.modules[n] = orig
            else:
                sys.modules.pop(n, None)


# --------------------------------------------------------------------------- #
# core: single-model, single-image inference
# faithful port of Fast_SPECT.DL_inference.model_inference_regression (single)
# --------------------------------------------------------------------------- #
def infer_single_model(model_url,
                       input_url,
                       output_dir,
                       device="cuda",
                       suffix="",
                       sliding_overlap="from-model",
                       sw_batch_size=1,
                       decimal_places=4,
                       slice_inferer_spatial_dim=0,
                       use_autocast_inference=True,
                       autocast_dtype="bfloat16",
                       progress_cb=None,
                       cancel_cb=None,
                       fold_index=0,
                       fold_total=1,
                       fold_tag="model",
                       log=print):
    import torch
    import monai
    import nibabel as nib
    import time

    _patch_numpy_rng()
    os.makedirs(output_dir, exist_ok=True)
    device = resolve_device(device, log=log)
    timings = {}

    def _check_cancel():
        if cancel_cb and cancel_cb():
            raise InferenceCancelled()

    def _report(frac, extra=""):
        if progress_cb:
            progress_cb(fold_index, fold_total, frac, f"{fold_tag}{(' ' + extra) if extra else ''}")

    _check_cancel()

    # prefer a compressed -inference.pth twin if the criteria pointed at -Full
    load_url = model_url
    if model_url.endswith("-Full.pth") or model_url.lower().endswith("full.pth"):
        for twin_suffix in ("-inference.pth", "-Inference.pth"):
            twin = re.sub(r"[-]?[Ff]ull\.pth$", twin_suffix, model_url)
            if twin != model_url and os.path.exists(twin):
                log("using the compressed inference checkpoint twin")
                load_url = twin
                break

    _check_cancel()

    log(f"loading checkpoint: {os.path.basename(load_url)}")
    _report(None, "loading checkpoint…")

    torch_device = torch.device(device)
    # Thread count: setting a high torch/OpenMP thread count from a BACKGROUND
    # thread (as the Slicer module does) makes the OpenMP pool contend with the
    # host application's main thread. On a 32-core box that showed up as ~180x
    # slowdown of everything on the worker -- including torch.load -- with the
    # machine looking idle (spin-waiting, not working).
    #
    # So: only raise the thread count when we are on the main thread, and never
    # request more than a modest cap. GPU runs need almost no CPU threads.
    try:
        import threading as _threading
        _on_main = _threading.current_thread() is _threading.main_thread()
        n_cpu = os.cpu_count() or 1
        if torch_device.type == "cuda":
            want = min(8, n_cpu)          # GPU path: CPU only feeds/collects
        else:
            # CPU path: use everything, but only from the main thread. Raising
            # the OpenMP pool size from a background thread made it contend with
            # the host's Qt main thread and slowed everything down badly.
            want = n_cpu if _on_main else torch.get_num_threads()
        if _on_main:
            if torch.get_num_threads() != want:
                torch.set_num_threads(want)
            log(f"torch threads: {torch.get_num_threads()} "
                f"(machine has {n_cpu} cores)")
        else:
            # worker thread: leave the process-wide setting alone
            log(f"torch threads: {torch.get_num_threads()} "
                f"(machine has {n_cpu} cores; not changed from worker thread)")
    except Exception:
        pass

    t0 = time.perf_counter()
    model_dictionary = _safe_torch_load(load_url, log=log, map_location=str(torch_device))
    timings["load_checkpoint"] = time.perf_counter() - t0
    log(f"    [time] torch.load (unpickle): {timings['load_checkpoint']:.2f}s")

    model = model_dictionary["model"]
    test_transform = model_dictionary["test_transform"]
    sliding_window_shape = model_dictionary["sliding_window_shape"]
    if sliding_overlap == "from-model":
        sliding_overlap = model_dictionary["sliding_windows_overlap"]
    post_transforms_test = model_dictionary["post_transforms_test"]

    t0 = time.perf_counter()
    model.to(torch_device)
    model.eval()
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    timings["model_to_device"] = time.perf_counter() - t0
    log(f"    [time] model.to({device}) + eval: {timings['model_to_device']:.2f}s")

    data = [{"input_image": input_url, "output_image": input_url}]
    dataset = monai.data.Dataset(data=data, transform=test_transform)
    data_loader = monai.data.DataLoader(dataset, batch_size=1, num_workers=0)

    input_name = os.path.basename(input_url).replace(".nii.gz", "").replace(".nii", "")
    out_path = os.path.join(output_dir, f"{input_name}{suffix}.nii.gz")

    log(f"sliding window shape={sliding_window_shape}  overlap={sliding_overlap}  device={device}")

    for batch in data_loader:
        with torch.no_grad():
            t0 = time.perf_counter()
            # match yazdan.DL: non_blocking transfer, no synchronize
            input_image = torch.squeeze(batch["input_image"], dim=-1).to(
                torch_device, non_blocking=True)
            timings["preprocess"] = time.perf_counter() - t0
            log(f"    [time] load+preprocess input: {timings['preprocess']:.2f}s "
                f"(shape {tuple(input_image.shape)})")

            is3d = len(sliding_window_shape) == 3
            total_windows = (compute_num_windows(list(input_image.shape[2:]),
                                                 sliding_window_shape, sliding_overlap)
                             if is3d else None)
            # Verify what the model and input ACTUALLY sit on. A silent CPU
            # fallback here (model or tensor not really on the GPU) is the
            # classic cause of "device=cuda" in the log but ~100x slow windows.
            try:
                _mp = next(model.parameters())
                log(f"    model device: {_mp.device} | dtype: {_mp.dtype}")
            except Exception:
                pass
            log(f"    input device: {input_image.device} | dtype: {input_image.dtype}")

            t0 = time.perf_counter()

            # Match yazdan.DL.model_inference_regression_multi_workers: run the
            # sliding window under bf16 autocast on CUDA. This is THE speed
            # difference -- fp32 Swin inference is many times slower.
            _use_amp = bool(use_autocast_inference) and torch_device.type == "cuda"
            _amp_dtype = {"bfloat16": torch.bfloat16,
                          "float16": torch.float16}.get(str(autocast_dtype).lower(),
                                                        torch.bfloat16)
            if _use_amp:
                log(f"    autocast: cuda/{str(_amp_dtype).replace('torch.', '')}")

            # NOTE: pass the RAW model as the predictor, exactly as
            # yazdan.DL.model_inference_regression_multi_workers does.
            #
            # A Python wrapper here runs on EVERY window, and anything it does
            # (cancel checks, progress reporting, and above all
            # torch.cuda.synchronize() for "accurate" timing) serialises the GPU
            # pipeline and destroys async execution. That was measured at 100s+
            # per window versus 0.05s for the same model called directly.
            #
            # Progress is therefore reported per FOLD rather than per window.
            _report(0.0, "running sliding window…")

            def _run_inference():
                if not is3d:
                    inferer = monai.inferers.SliceInferer(
                        roi_size=sliding_window_shape,
                        overlap=sliding_overlap,
                        sw_batch_size=sw_batch_size,
                        cval=-1,
                        progress=False,
                        spatial_dim=slice_inferer_spatial_dim,
                    )
                    return inferer(input_image, model)
                return monai.inferers.sliding_window_inference(
                    inputs=input_image,
                    roi_size=sliding_window_shape,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=sliding_overlap,
                    progress=False,
                )

            if _use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype):
                    predicted_image = _run_inference()
                # bring the prediction back to fp32 before the inverse transform,
                # so post-processing / saving behaves exactly as in fp32 mode
                predicted_image = predicted_image.float()
            else:
                predicted_image = _run_inference()

            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            timings["sliding_window"] = time.perf_counter() - t0
            log(f"    [time] sliding-window inference: {timings['sliding_window']:.2f}s "
                f"({total_windows or '?'} windows)")

            _check_cancel()
            _report(1.0, "finalizing…")
            t0 = time.perf_counter()
            batch["predict_image"] = predicted_image
            prepared = [post_transforms_test(i) for i in monai.data.decollate_batch(batch)]
            predicted = monai.handlers.utils.from_engine(["predict_image"])(prepared)
            predicted_array = torch.squeeze(predicted[0]).cpu().detach().numpy()

            src = nib.load(input_url)
            predicted_nifti = nib.Nifti1Image(predicted_array, affine=src.affine, header=src.header)
            nib.save(predicted_nifti, out_path)
            compress_image(url=out_path, target_url=out_path, decimal_places=decimal_places)
            timings["postprocess_save"] = time.perf_counter() - t0
            log(f"    [time] invert + save: {timings['postprocess_save']:.2f}s")

    total = sum(timings.values())
    log("    [time] ── fold summary ──  "
        + "  ".join(f"{k}={v:.2f}s" for k, v in timings.items())
        + f"  | TOTAL={total:.2f}s")

    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if not os.path.exists(out_path):
        raise RuntimeError("inference finished but no output file was written.")
    log(f"wrote: {os.path.basename(out_path)}")
    return out_path


# --------------------------------------------------------------------------- #
# high level: single file  OR  fold-directory ensemble, with normalization
# --------------------------------------------------------------------------- #
def run(input_url,
        model_path,
        output_dir,
        mode="auto",
        device="auto",
        sliding_overlap="from-model",
        model_criteria=DEFAULT_CRITERIA,
        folds="all",
        sw_batch_size=1,
        decimal_places=4,
        normalize=True,
        percentile=99.0,
        crop_to_body_region=True,
        crop_threshold=0.09,
        crop_margin_mm=10.0,
        restore_original_geometry=True,
        use_autocast_inference=True,
        autocast_dtype="bfloat16",
        progress_cb=None,
        cancel_cb=None,
        log=print):
    """
    mode == 'file' : `model_path` is a single .pth
    mode == 'dir'  : `model_path` is a ModelDirectory containing fold*/ ;
                     runs the selected folds and averages the results.
    mode == 'auto' : dir if `model_path` is a directory else file.

    normalize : if True (default) the input is first intensity-normalized to
                [0, 1] using min=0 and max=P`percentile`(non-zero), matching
                Fast-SPECT-Imaging/preprocessing.py, before inference.
    Returns the path to the final NIfTI.
    """
    os.makedirs(output_dir, exist_ok=True)

    if mode == "auto":
        mode = "dir" if os.path.isdir(model_path) else "file"
    log(f"mode = {mode}")

    # ---- preprocessing: percentile normalization ----
    infer_input = input_url
    if normalize:
        if cancel_cb and cancel_cb():
            raise InferenceCancelled()
        base = os.path.basename(input_url).replace(".nii.gz", "").replace(".nii", "")
        norm_path = os.path.join(output_dir, f"{base}--normalized.nii.gz")
        log(f"preprocessing input -> normalize to P{percentile:g} percentile")
        if progress_cb:
            progress_cb(0, 1, 0.0, "normalizing input…")
        normalize_percentile(input_url, norm_path, percentile=percentile, log=log)
        infer_input = norm_path
        log(f"normalized input written: {os.path.basename(norm_path)}")
    else:
        log("normalization disabled — using the input image as-is.")

    # ---- preprocessing: crop to the body region ----
    # The models were TRAINED on cropped images, so inference must see the same
    # field of view. Cropping happens AFTER normalization, because the body
    # threshold (default 0.09) is expressed in normalized intensity units.
    crop_box = None
    uncropped_input = infer_input
    if crop_to_body_region:
        if cancel_cb and cancel_cb():
            raise InferenceCancelled()
        base = os.path.basename(infer_input).replace(".nii.gz", "").replace(".nii", "")
        crop_path = os.path.join(output_dir, f"{base}--cropped.nii.gz")
        log("preprocessing input -> crop to body region "
            f"(threshold={crop_threshold}, margin={crop_margin_mm}mm)")
        if progress_cb:
            progress_cb(0, 1, 0.0, "cropping to body…")
        try:
            crop_path, crop_box = crop_to_body(
                infer_input, crop_path, threshold=crop_threshold,
                margin_mm=crop_margin_mm, log=log)
            infer_input = crop_path
            log(f"cropped input written: {os.path.basename(crop_path)}")
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: body cropping failed ({exc}); "
                f"continuing with the uncropped image.")
            crop_box = None
    else:
        log("body cropping disabled — the model was trained on cropped images, "
            "so results may be degraded.")

    def _finalize(pred_path):
        """Put a prediction made on the cropped volume back into the original
        image geometry, so it overlays the scan the user loaded."""
        if not (restore_original_geometry and crop_box is not None and pred_path):
            return pred_path
        try:
            base = os.path.basename(pred_path).replace(".nii.gz", "").replace(".nii", "")
            full = os.path.join(output_dir, f"{base}--fullfov.nii.gz")
            paste_into_reference(pred_path, input_url, full, log=log)
            return full
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: could not restore original geometry ({exc}); "
                f"returning the cropped prediction.")
            return pred_path

    if mode == "file":
        return _finalize(infer_single_model(
            model_path, infer_input, output_dir,
            device=device, suffix="",
            sliding_overlap=sliding_overlap,
            sw_batch_size=sw_batch_size,
            decimal_places=decimal_places,
            use_autocast_inference=use_autocast_inference,
            autocast_dtype=autocast_dtype,
            progress_cb=progress_cb, cancel_cb=cancel_cb,
            fold_index=0, fold_total=1, fold_tag="model",
            log=log,
        ))

    # -------- ensemble across folds --------
    model_directory = model_path
    all_folds = list_fold_dirs(model_directory)
    if not all_folds:
        raise RuntimeError(f"No fold* subfolders found under: {model_directory}")

    selected = resolve_folds(all_folds, folds)
    if not selected:
        raise RuntimeError(f"No fold matched folds={folds!r} (found {len(all_folds)} folds).")

    found_nums = [(_fold_number(d)) for d in selected]
    log(f"found {len(all_folds)} folds; using {len(selected)} -> fold numbers {found_nums}")

    input_name = os.path.basename(infer_input).replace(".nii.gz", "").replace(".nii", "")
    total_units = len(selected) + (1 if len(selected) > 1 else 0)
    fold_outputs = []
    unit = 0
    for fd in selected:
        if cancel_cb and cancel_cb():
            raise InferenceCancelled()
        fold_name = os.path.basename(fd.rstrip("/\\"))
        # find the checkpoint: requested criteria first, then common fallbacks,
        # at the fold root or one level down.
        names = [model_criteria] if model_criteria and model_criteria != "all" else []
        for alt in ("BestTrainMetricModel-inference.pth",
                    "BestTrainMetricModel-share.pth",
                    "BestTrainMetricModel-Full.pth",
                    "*Model-Full.pth", "*Model-full.pth", "*full.pth"):
            if alt not in names:
                names.append(alt)
        candidates = []
        for nm in names:
            candidates += glob.glob(os.path.join(fd, nm))
            candidates += glob.glob(os.path.join(fd, "*", nm))
        if not candidates:
            for pat in ("*-inference.pth", "*-share.pth", "*-Full.pth",
                        "*full.pth", "*.pth"):
                candidates += glob.glob(os.path.join(fd, pat))
                candidates += glob.glob(os.path.join(fd, "*", pat))
                if candidates:
                    break
        candidates = [c for c in candidates if os.path.isfile(c)]
        if not candidates:
            log(f"  (skip {fold_name}: no checkpoint .pth found)")
            unit += 1
            continue
        log(f"--- fold {fold_name} ---")
        out = infer_single_model(
            candidates[0], infer_input, output_dir,
            device=device, suffix=f"_{fold_name}",
            sliding_overlap=sliding_overlap,
            sw_batch_size=sw_batch_size,
            decimal_places=decimal_places,
            use_autocast_inference=use_autocast_inference,
            autocast_dtype=autocast_dtype,
            progress_cb=progress_cb, cancel_cb=cancel_cb,
            fold_index=unit, fold_total=total_units, fold_tag=f"{fold_name}",
            log=log,
        )
        fold_outputs.append(out)
        unit += 1

    if not fold_outputs:
        raise RuntimeError("No fold produced an output; check model criteria / folds.")

    if len(fold_outputs) == 1:
        log("only one fold -> no averaging needed.")
        return _finalize(fold_outputs[0])

    ensemble_path = os.path.join(output_dir, f"{input_name}_Ensemble.nii.gz")
    log(f"ensembling {len(fold_outputs)} folds -> {os.path.basename(ensemble_path)}")
    if progress_cb:
        progress_cb(total_units - 1, total_units, 0.5, "ensembling folds…")
    ensemble_image_regression(fold_outputs, target_url=ensemble_path)

    for p in fold_outputs:
        try:
            os.remove(p)
        except Exception:
            pass
    if progress_cb:
        progress_cb(total_units, total_units, 1.0, "done")

    return _finalize(ensemble_path)
