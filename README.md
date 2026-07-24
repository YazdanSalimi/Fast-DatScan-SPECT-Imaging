# Fast DatScan SPECT Imaging with AI
## Deep Learning pipeline for fast DatScan SPECT imaging.
It contains the trained models for brain DatScan SPECT.

## Models
Please download the trained models and give the path on your machine where you saved the models to the inference function. Models are available for diferrent levels of time reduction. 20%, 25%, and 50%. Please select a model closer to the acquisition time.

- [Brain DaTscan I123](https://doi.org/10.26037/yareta:z37sdseqrra4ziesaoovdyomba) — these models are trained to convert a fast (20, 25, and 50%) I123-ioflupane brain image to a standard 15 minute scan.

Inference examples are provided below. Please check the SNMMI [abstract](https://jnm.snmjournals.org/content/65/supplement_2/241761.abstract) for more information. The full paper including the methodolgy and resutls will be available soon.

---

## Two ways to use it

| | Python package | 3D Slicer module |
|---|---|---|
| Best for | batches, scripting, research pipelines | single patients, clinical review, no coding |
| Input | NIfTI files | any volume loaded in Slicer (incl. DICOM) |
| Output | NIfTI files | volume in the scene, optional DICOM export |

---

## 1. Python package

To install this repository, simply run:

```bash
pip install git+https://github.com/YazdanSalimi/Fast-SPECT-Imaging.git
```

### Example

```python
from fastdatscan import predict_image

output = predict_image(
    input_url="patient01.nii.gz",       # a fast-20% DaTscan SPECT image. 
    model_directory="/data/models/BrainDaTscan-I123/20%/unet--light--2.46mm--96",
    output_dir="/data/out",
)
print(output)
```

A whole folder:

```python
from fastdatscan import predict_batch

predict_batch("/data/in/*.nii.gz", model_directory=MODEL_DIR,
              output_dir="/data/out", device="cuda")
```

The preprocessing the models expect is applied automatically, and the prediction
is resampled back onto the input grid so it overlays the original scan.
See `Example.py` for options, progress callbacks and batch processing.

### Command line

```bash
fastdatscan --list-models /data/models
fastdatscan --input scan.nii.gz --model-dir MODEL_DIR --output-dir out
fastdatscan --input "cases/*.nii.gz" --model-dir MODEL_DIR --output-dir out --folds 0,2,4
```

### Preprocessing

The models were trained on **normalized and cropped** images, so inference must
present the data the same way:

1. **Normalize** to the 99th percentile of the non-zero voxels
   (`min = 0`, `max = P99`), using the unclipped rescaled image.
2. **Crop to the body region** — threshold the normalized image at `0.09`, take
   the largest connected component, and crop to its bounding box with a 10 mm
   margin.

```python
import SimpleITK as sitk

image = sitk.ReadImage(image_normalized_url)
body_segment = sitk.Cast(image > .09, sitk.sitkUInt8)

component_image = sitk.ConnectedComponent(body_segment)
sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
largest_component_binary_image = sum([sorted_component_image == label for label in range(1, 1 + 1)])

image_cropped = crop_image_to_segment(
    image=image,
    segment=largest_component_binary_image,
    margin_mm=10,
)[0]
```

Skipping either step will degrade the output.

---

## 2. 3D Slicer module

An easy alternative is downloading the trained models using the link above and
the ready-to-use **3D Slicer** module.

### Install

1. Download the slicer module available in the repository and unzip it.
2. Download / clone the trained models uisng this [link](https://drive.google.com/drive/folders/13AhUZCJ7lqmLxkswYxYXNMuw4iBXHnaA?usp=sharing).
3. In Slicer: **Edit ▸ Application Settings ▸ Modules ▸ Additional module paths** → add that folder containing the unzip slicer module.
4. Restart Slicer. The module appears under **Deep Learning ▸ Fast DaTscan SPECT**.

### Use

**1 · Dependencies (run once).** Installs torch, monai, nibabel, SimpleITK and
friends into Slicer's Python. For an NVIDIA GPU, use *Auto-detect CUDA* and
*Install GPU torch (no admin)*, then restart Slicer.

**2 · Choose model.** Download the models yourself from the Yareta link above and
unzip them into a folder laid out as `<task>/<level>/<config>/fold*/`. Point the
module at that folder and press **Scan Models folder** — nothing is downloaded
automatically. Then pick task, acquisition level, config, and which folds to
ensemble.

**3 · Input.** Select the fast-acquisition volume loaded in the scene.

**4 · Options & output.** Normalization (P99) and body cropping are on by default
and reproduce the training preprocessing described above; the prediction is placed
back into the original image geometry so it overlays the scan. Optionally export
the result to DICOM, reusing the input's patient/study tags.

### How it runs

Inference runs in a **separate Python process**, the same approach the nnUNet
Slicer extension uses. The worker gets its own interpreter and CUDA context, so
Slicer stays responsive, progress streams into the log, and *Cancel* stops the
job immediately. GPU runs use bfloat16 autocast; CPU runs use all cores.


## Repository layout

```
fastdatscan/            installable Python package
    api.py              predict_image / predict_batch / building blocks
    cli.py              the `fastdatscan` command
    share.py            prepare checkpoints for distribution
    _core.py            preprocessing + inference implementation
FastDatScanSPECT/       3D Slicer module (add this folder to Slicer)
Example.py              worked examples
setup.py
```

## Requirements

Python ≥ 3.6 with `torch`, `monai`, `SimpleITK`, `nibabel`, `numpy`, `pandas`,
`tqdm`, `termcolor`, `natsort`, `glob2`, `multiprocess`. A CUDA GPU is optional
but much faster.

## Feedback

We welcome any feedback, suggestions, or contributions to improve this project!

For any further questions please email me at: salimiyazdan@gmail.com
