# Fast-SPECT-Imaging
## Deep Learning pipeline for fast SPECT imaging.
It contains the trained models for multiple tracers for brain SPECT and other applications.

## Models
Please download the trained models for each task and give the path on your machine where you saved the models to the inference function.

- [Brain DaTscan I123](https://doi.org/10.26037/yareta:z37sdseqrra4ziesaoovdyomba) — these models are trained to convert a three minute I123-ioflupane brain image to a standard 15 minute scan.

Inference examples are provided below. Please check the SNMMI [abstract](https://jnm.snmjournals.org/content/65/supplement_2/241761.abstract) for more information. The other models dedicated to acquisition levels of four and eight minutes will be updated soon.

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
from Fast_SPECT import DL_inference

"""
Easy to use!
list_images is a list containing the address to your NIFTI input files on your machine.
model_directory is where you saved the downloaded models on your machine.
predict_directory is where you want to see your outputs
"""
list_ensembled_images = ensemble_regression_folds(
    list_images,
    model_directory,
    predict_directory,
    model_criteria="all",
)
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

1. Download / clone the `FastDatScanSPECT` folder.
2. In Slicer: **Edit ▸ Application Settings ▸ Modules ▸ Additional module paths** → add that folder.
3. Restart Slicer. The module appears under **Deep Learning ▸ Fast DaTscan SPECT**.

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

---

## Sharing trained models

Checkpoints saved during training can contain references to the private training
package and to absolute paths from the training machine. Before distributing a
`.pth`, prepare it:

```python
yazdan.DL.prepare_saved_pth_for_sharing(model_url)   # writes *-share.pth
```

This strips the private package and bookkeeping, keeps only the fields needed for
inference, and writes a clean `-share.pth`. Verify it on a machine where the
training package is **not** importable — if it loads there, it will load for
everyone.

---

## Requirements

Python ≥ 3.6 with `torch`, `monai`, `SimpleITK`, `nibabel`, `numpy`, `pandas`,
`tqdm`, `termcolor`, `natsort`, `glob2`, `multiprocess`. A CUDA GPU is optional
but much faster.

## Feedback

We welcome any feedback, suggestions, or contributions to improve this project!

For any further questions please email me at: salimiyazdan@gmail.com
