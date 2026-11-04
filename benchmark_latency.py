"""Single-image inference latency benchmark for all six backbones, in ONE mode.

The original README table was not like-for-like: the five baselines were timed
with eager ``model(x)`` calls inside train_baseline_comparison.py under a
``mixed_float16`` policy, while the proposed model (float32) was timed in a
separate run. This script times every model through the same code path, same
dtype (float32), same input, batch size 1.

Requires Keras >= the version that saved the archives (3.14.0, Python >= 3.11).

Modes:
  compiled  model call wrapped in ``tf.function`` (graph mode, no XLA). This is
            the path ``model.predict`` uses in app.py and eval_final.py.
  eager     plain ``model(x, training=False)`` Python call, op-by-op dispatch.

Usage:
  python benchmark_latency.py --mode compiled
  python benchmark_latency.py --mode eager --runs 200
  python benchmark_latency.py --mode compiled --device cpu   # hide GPUs
"""

import argparse
import json
import platform
import tempfile
import time
import zipfile
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from src.custom_layers import MedSwish, WBCAttentionBlock
from src.custom_losses import WBCFocalLoss

IMG_SIZE = 224

MODELS = {
    "VGG16": "outputs/baseline_results/VGG16_best.keras",
    "ResNet50V2": "outputs/baseline_results/ResNet50V2_best.keras",
    "MobileNetV2": "outputs/baseline_results/MobileNetV2_best.keras",
    "EfficientNetB0": "outputs/baseline_results/EfficientNetB0_best.keras",
    "DenseNet121_vanilla": "outputs/baseline_results/DenseNet121_vanilla_best.keras",
    "Proposed": "data/models/wbc_final_model_densenet.keras",
}

CUSTOM_OBJECTS = {
    "MedSwish": MedSwish,
    "WBCAttentionBlock": WBCAttentionBlock,
    "WBCFocalLoss": WBCFocalLoss,
    "wbc_focal_loss": WBCFocalLoss,
}


def to_float32_config(node):
    """Rewrite ``mixed_float16`` / ``float16`` dtypes in a saved config to ``float32``.

    The five baselines were trained with a ``mixed_float16`` policy while the
    proposed model is float32. Timing them as saved would compare precisions,
    not architectures (and on CPU, float16 kernels are emulated). Mixed
    precision stores variables as float32, so the weights load unchanged.
    """
    if isinstance(node, dict):
        return {
            k: ("float32" if k in ("dtype", "name") and v in ("mixed_float16", "float16")
                else to_float32_config(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [to_float32_config(v) for v in node]
    return node


def saved_keras_version(path):
    """Keras version recorded in a .keras archive's metadata.json."""
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("metadata.json"))["keras_version"]


def check_keras_version(path):
    """Fail fast with a clear message instead of a deserialization traceback.

    Archives written by a newer Keras carry layer kwargs (``quantization_config``,
    BatchNormalization ``renorm``) that older versions reject.
    """
    saved = saved_keras_version(path)
    have = keras.__version__

    def key(v):
        return tuple(int(p) for p in v.split(".")[:2])

    if key(have) < key(saved):
        raise RuntimeError(
            f"{path} was saved with Keras {saved}; this env has {have}. "
            f"Install keras>={saved} (needs Python >= 3.11)."
        )


def load_model_float32(path, tmp_dir):
    """Load a .keras archive with all dtypes forced to float32 (via a temp copy)."""
    check_keras_version(path)
    src = Path(path)
    dst = Path(tmp_dir) / src.name
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                data = json.dumps(to_float32_config(json.loads(data))).encode("utf-8")
            zout.writestr(item, data)
    return keras.models.load_model(dst, custom_objects=CUSTOM_OBJECTS, compile=False)


def make_infer_fn(model, mode):
    """Return a callable that runs one forward pass in the requested mode."""
    if mode == "eager":
        return lambda x: model(x, training=False)
    if mode == "compiled":
        return tf.function(lambda x: model(x, training=False))
    raise ValueError(f"Unknown mode: {mode!r} (expected 'compiled' or 'eager')")


def _block(outputs):
    """Force materialization so async dispatch does not hide compute time."""
    for t in tf.nest.flatten(outputs):
        np.asarray(t)


def time_model(infer_fn, warmup, runs, seed=0):
    """Time ``runs`` single-image forward passes after ``warmup`` untimed passes."""
    rng = np.random.default_rng(seed)
    x = tf.constant(rng.random((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
    for _ in range(warmup):
        _block(infer_fn(x))
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _block(infer_fn(x))
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def summarize(times_ms):
    """Mean / median / p95 / std / min in milliseconds."""
    arr = np.asarray(times_ms, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("summarize() needs at least one timing")
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "runs": int(arr.size),
    }


def format_table(results, mode):
    """Markdown table, one row per model, sorted as in MODELS."""
    lines = [
        f"| Model | Params | Mean ({mode}) | Median | p95 |",
        "| :-- | --: | --: | --: | --: |",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['total_params'] / 1e6:.2f} M | "
            f"{r['mean_ms']:.1f} ms | {r['median_ms']:.1f} ms | {r['p95_ms']:.1f} ms |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["compiled", "eager"], default="compiled")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto",
                        help="auto = GPU if TF sees one; cpu = hide all GPUs")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--out", default="outputs/latency_benchmark")
    args = parser.parse_args()

    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    devices = [d.name for d in tf.config.get_visible_devices()]
    gpus = [tf.config.experimental.get_device_details(g).get("device_name", g.name)
            for g in tf.config.get_visible_devices("GPU")]
    print(f"TF {tf.__version__} | Keras {keras.__version__} | devices: {devices} | GPUs: {gpus} | mode: {args.mode}")

    results = []
    tmp = tempfile.TemporaryDirectory()
    for name in args.models:
        keras.backend.clear_session()
        model = load_model_float32(MODELS[name], tmp.name)
        stats = summarize(time_model(make_infer_fn(model, args.mode), args.warmup, args.runs))
        row = {"model": name, "total_params": int(model.count_params()), **stats}
        results.append(row)
        print(f"  {name:<20} mean={stats['mean_ms']:7.1f} ms  median={stats['median_ms']:7.1f}  p95={stats['p95_ms']:7.1f}")
        del model
    tmp.cleanup()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "warmup": args.warmup,
        "runs": args.runs,
        "batch_size": 1,
        "dtype": "float32",
        "tf_version": tf.__version__,
        "keras_version": keras.__version__,
        "gpus": gpus,
        "devices": devices,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "results": results,
    }
    device_tag = "gpu" if gpus else "cpu"
    out_json = out_dir / f"latency_{args.mode}_{device_tag}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + format_table(results, args.mode))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
