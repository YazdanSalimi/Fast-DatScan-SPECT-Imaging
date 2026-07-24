from setuptools import setup, find_packages
import os
import re


def _read(name):
    here = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(here, name)
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _version():
    src = _read(os.path.join("fastdatscan", "__init__.py"))
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', src, re.M)
    return m.group(1) if m else "0.1.0"


setup(
    name="fastdatscan",
    version=_version(),
    author="Yazdan Salimi",
    author_email="salimiyazdan@gmail.com",
    description=("Deep learning enhancement of fast-acquisition brain DaTscan "
                 "SPECT images"),
    long_description=_read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/YazdanSalimi/Fast-SPECT-Imaging",
    project_urls={
        "Source": "https://github.com/YazdanSalimi/Fast-SPECT-Imaging",
        "Models": "https://doi.org/10.26037/yareta:z37sdseqrra4ziesaoovdyomba",
        "Abstract": ("https://jnm.snmjournals.org/content/65/supplement_2/"
                     "241761.abstract"),
    },
    packages=find_packages(include=["fastdatscan", "fastdatscan.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "monai",
        "SimpleITK",
        "nibabel",
        "numpy",
        "natsort",
        "tqdm",
        "termcolor",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "fastdatscan=fastdatscan.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=("SPECT DaTscan ioflupane nuclear-medicine deep-learning "
              "image-enhancement denoising medical-imaging monai"),
    include_package_data=True,
    zip_safe=False,
)
