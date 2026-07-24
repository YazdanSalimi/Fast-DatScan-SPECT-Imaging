# -*- coding: utf-8 -*-
"""
fastdatscan.cli -- command-line interface.

Installed as the `fastdatscan` command:

    fastdatscan --input patient.nii.gz --model-dir MODELS --output-dir out
    fastdatscan --input "cases/*.nii.gz" --model-dir MODELS --output-dir out
    fastdatscan --list-models /data/models

It is also the worker the 3D Slicer module launches as a subprocess, which is why
progress is printed as machine-readable `PROGRESS<TAB>fraction<TAB>message` lines
and the final path as `RESULT<TAB>path`.
"""

import os
import sys
import glob
import time
import argparse
import traceback


def _build_parser():
    p = argparse.ArgumentParser(
        prog="fastdatscan",
        description="Deep-learning enhancement of fast brain DaTscan SPECT images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  fastdatscan --input scan.nii.gz --model-dir MODELS --output-dir out
  fastdatscan --input "cases/*.nii.gz" --model-dir MODELS --output-dir out --folds 0,2,4
  fastdatscan --list-models /data/models
""")
    p.add_argument("--input", help="input NIfTI, or a glob such as 'cases/*.nii.gz'")
    p.add_argument("--output-dir", help="where to write the results")
    p.add_argument("--model-dir", help="model directory containing fold*/ subfolders")
    p.add_argument("--model-file", help="a single .pth instead of a fold ensemble")
    p.add_argument("--list-models", metavar="ROOT",
                   help="list the models found under ROOT and exit")

    g = p.add_argument_group("inference")
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    g.add_argument("--folds", default="all",
                   help="'all', an integer N (first N), or '0,2,4'")
    g.add_argument("--criteria", default="BestTrainMetricModel-inference.pth",
                   help="checkpoint filename to prefer inside each fold")
    g.add_argument("--overlap", default="from-model",
                   help="sliding-window overlap, or 'from-model'")
    g.add_argument("--sw-batch-size", type=int, default=1)
    g.add_argument("--decimals", type=int, default=4)
    g.add_argument("--no-autocast", action="store_true",
                   help="disable bfloat16 autocast on CUDA")
    g.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads (0 = all cores)")

    g = p.add_argument_group("preprocessing")
    g.add_argument("--percentile", type=float, default=99.0)
    g.add_argument("--no-normalize", action="store_true",
                   help="skip percentile normalization (NOT recommended)")
    g.add_argument("--no-crop", action="store_true",
                   help="skip body cropping (NOT recommended: models were "
                        "trained on cropped images)")
    g.add_argument("--crop-threshold", type=float, default=0.09)
    g.add_argument("--crop-margin", type=float, default=10.0)
    g.add_argument("--keep-cropped-fov", action="store_true",
                   help="return the prediction on the cropped grid instead of "
                        "resampling it back to the input geometry")

    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--version", action="store_true")
    return p


def main(argv=None):
    # Windows consoles use a legacy code page; force UTF-8 so a parent process
    # reading our stdout never hits an undecodable byte.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = _build_parser().parse_args(argv)

    if args.version:
        from . import __version__
        print(f"fastdatscan {__version__}")
        return 0

    from . import api

    if args.list_models:
        models = api.list_models(args.list_models)
        if not models:
            print(f"no models found under {args.list_models}")
            print("expected layout: <task>/<level>/<config>/fold*/...pth")
            return 1
        print(f"{len(models)} model(s) under {args.list_models}:\n")
        for m in models:
            folds = api.list_folds(m["path"])
            print(f"  {m['name']}")
            print(f"      folds: {folds}")
            print(f"      path : {m['path']}")
        return 0

    if not args.input or not args.output_dir:
        _build_parser().print_help()
        return 2
    model = args.model_file or args.model_dir
    if not model:
        print("error: give --model-dir or --model-file", file=sys.stderr)
        return 2

    verbose = not args.quiet
    t0 = time.perf_counter()

    if verbose:
        import torch
        n_cpu = os.cpu_count() or 1
        want = args.threads if args.threads > 0 else n_cpu
        try:
            torch.set_num_threads(want)
        except Exception:
            pass
        print(f"[fastdatscan] torch {torch.__version__} | "
              f"threads {torch.get_num_threads()}/{n_cpu} | "
              f"cuda={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            try:
                print(f"[fastdatscan] gpu: {torch.cuda.get_device_name(0)}", flush=True)
            except Exception:
                pass
        elif args.device == "cuda":
            print("[fastdatscan] WARNING: --device cuda requested but this "
                  "interpreter has no CUDA support; falling back to CPU.",
                  flush=True)

    def progress_cb(fold_i, fold_n, frac, msg):
        base = fold_i / float(max(1, fold_n))
        f = base + (frac or 0.0) / float(max(1, fold_n)) if frac is not None else base
        print(f"PROGRESS\t{min(1.0, max(0.0, f)):.4f}\t{msg}", flush=True)

    common = dict(
        model_directory=model,
        output_dir=args.output_dir,
        folds=args.folds,
        device=args.device,
        normalize=not args.no_normalize,
        percentile=args.percentile,
        crop=not args.no_crop,
        crop_threshold=args.crop_threshold,
        crop_margin_mm=args.crop_margin,
        restore_original_geometry=not args.keep_cropped_fov,
        use_autocast=not args.no_autocast,
        sliding_overlap=args.overlap,
        sw_batch_size=args.sw_batch_size,
        decimal_places=args.decimals,
        model_criteria=args.criteria,
        verbose=verbose,
    )

    is_glob = any(ch in args.input for ch in "*?[")
    try:
        if is_glob:
            files = sorted(glob.glob(args.input))
            if not files:
                print(f"error: no files matched {args.input}", file=sys.stderr)
                return 1
            outs = api.predict_batch(files, **common)
            failed = sum(1 for o in outs if not o)
            if verbose:
                print(f"[fastdatscan] total {time.perf_counter() - t0:.1f}s",
                      flush=True)
            return 1 if failed else 0
        else:
            out = api.predict_image(args.input, progress_cb=progress_cb, **common)
    except Exception:
        traceback.print_exc()
        return 1

    if not out or not os.path.exists(out):
        print("error: inference produced no output", file=sys.stderr)
        return 3

    if verbose:
        print(f"[fastdatscan] total {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"RESULT\t{os.path.abspath(out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
