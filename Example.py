# -*- coding: utf-8 -*-
"""
Example.py -- how to use fastdatscan.

Run the pieces you need; each example is standalone.

Before you start:
  1. pip install git+https://github.com/YazdanSalimi/Fast-SPECT-Imaging.git
  2. Download the trained models from the Yareta link in the README and unzip
     them somewhere, e.g.

         /data/models/
             BrainDaTscan-I123/
                 20%/
                     unet--light--2.0mm--96/
                         fold--0/BestTrainMetricModel-inference.pth
                         fold--1/...
"""

import os

MODELS_ROOT = r"/data/models"
MODEL_DIR = os.path.join(MODELS_ROOT, "BrainDaTscan-I123", "20%",
                         "unet--light--2.0mm--96")
OUTPUT_DIR = r"/data/out"


# --------------------------------------------------------------------------- #
# 0. What models do I have?
# --------------------------------------------------------------------------- #
def example_list_models():
    from fastdatscan import list_models, list_folds

    for model in list_models(MODELS_ROOT):
        print(model["name"], "->", list_folds(model["path"]))


# --------------------------------------------------------------------------- #
# 1. One image (the common case)
#
#    Normalization (99th percentile) and cropping to the body region are applied
#    automatically, because the models were trained that way. The prediction is
#    resampled back onto the input grid, so it overlays the original scan.
# --------------------------------------------------------------------------- #
def example_single_image():
    from fastdatscan import predict_image

    output = predict_image(
        input_url=r"/data/in/patient01.nii.gz",
        model_directory=MODEL_DIR,
        output_dir=OUTPUT_DIR,
    )
    print("enhanced image:", output)


# --------------------------------------------------------------------------- #
# 2. A folder of images
# --------------------------------------------------------------------------- #
def example_batch():
    from fastdatscan import predict_batch

    outputs = predict_batch(
        list_images=r"/data/in/*.nii.gz",     # or a Python list of paths
        model_directory=MODEL_DIR,
        output_dir=OUTPUT_DIR,
        device="cuda",
    )
    print(f"{sum(1 for o in outputs if o)}/{len(outputs)} succeeded")


# --------------------------------------------------------------------------- #
# 3. Controlling the run
# --------------------------------------------------------------------------- #
def example_options():
    from fastdatscan import predict_image

    output = predict_image(
        input_url=r"/data/in/patient01.nii.gz",
        model_directory=MODEL_DIR,
        output_dir=OUTPUT_DIR,

        folds=[0, 2, 4],        # "all", 3 (first three), [0, 2, 4], or "0,2,4"
        device="cuda",          # "auto" | "cuda" | "cpu"
        use_autocast=True,      # bfloat16 on CUDA: much faster

        percentile=99.0,        # normalization upper bound
        crop_threshold=0.09,    # body threshold, in normalized units
        crop_margin_mm=10.0,

        sw_batch_size=1,        # raise to 2-4 to keep a GPU busier
        sliding_overlap="from-model",
    )
    print(output)


# --------------------------------------------------------------------------- #
# 4. Progress reporting (e.g. to drive a GUI progress bar)
# --------------------------------------------------------------------------- #
def example_progress():
    from fastdatscan import predict_image

    def on_progress(fold_index, fold_total, fraction, message):
        pct = 100.0 * (fold_index + (fraction or 0.0)) / max(1, fold_total)
        print(f"  {pct:5.1f}%  {message}")

    predict_image(r"/data/in/patient01.nii.gz", MODEL_DIR, OUTPUT_DIR,
                  progress_cb=on_progress, verbose=False)


# --------------------------------------------------------------------------- #
# 5. Using the preprocessing on its own
#
#    Useful if you want to inspect what the network actually sees, or to build a
#    custom pipeline. This reproduces exactly what predict_image() does
#    internally.
# --------------------------------------------------------------------------- #
def example_preprocessing_only():
    from fastdatscan import normalize_to_percentile, crop_to_body

    normalized = normalize_to_percentile(
        r"/data/in/patient01.nii.gz",
        os.path.join(OUTPUT_DIR, "patient01--normalized.nii.gz"),
        percentile=99.0,
    )
    cropped, crop_box = crop_to_body(
        normalized,
        os.path.join(OUTPUT_DIR, "patient01--cropped.nii.gz"),
        threshold=0.09,
        margin_mm=10.0,
    )
    print("normalized:", normalized)
    print("cropped   :", cropped)
    print("crop box  :", crop_box["crop_sizes"])
if __name__ == "__main__":
    example_single_image()
