"""Build notebooks/wbc_analyzer_kaggle_quickstart.ipynb.

The notebook must run on Kaggle without this repo, so the project sources
(src/custom_layers.py, src/custom_losses.py, src/preprocessing.py) and the five
sample images in docs/samples/ are embedded verbatim at build time. Re-run this
script after touching any of them:

    python notebooks/build_kaggle_notebook.py

tests/test_notebook.py fails if the committed notebook has drifted from src/.
"""

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "wbc_analyzer_kaggle_quickstart.ipynb"

CLASS_NAMES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
SOURCE_FILES = ["custom_layers.py", "custom_losses.py", "preprocessing.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text, hide_input=False):
    meta = {"_kg_hide-input": True} if hide_input else {}
    return {
        "cell_type": "code",
        "metadata": meta,
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n"),
    }


def verbatim(filename):
    body = (ROOT / "src" / filename).read_text(encoding="utf-8").rstrip("\n")
    return code(f"# ===== src/{filename} (verbatim copy) =====\n{body}", hide_input=True)


def sample_images_cell():
    lines = [
        "# Five Raabin-WBC sample images (one per class), embedded so this notebook",
        "# runs with nothing attached except the model.",
        "SAMPLE_IMAGES_B64 = {",
    ]
    for name in CLASS_NAMES:
        raw = (ROOT / "docs" / "samples" / f"{name.lower()}.jpg").read_bytes()
        lines.append(f'    "{name}": "{base64.b64encode(raw).decode("ascii")}",')
    lines.append("}")
    return code("\n".join(lines), hide_input=True)


TITLE = """
# WBC Analyzer: white blood cell classification quickstart

Classifies a peripheral blood smear crop into one of five leukocyte types:
**Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil**.

The model is a DenseNet121 backbone with a CBAM-style `WBCAttentionBlock` and a
learnable `MedSwish` activation, fed by a deterministic *Medical Enhanced Filter*
(percentile clip + LAB CLAHE + bilateral + edge-weighted sharpen). It has a 5-class
head (`main_out`) and an auxiliary Neutrophil-vs-Lymphocyte sigmoid (`aux_binary_out`).

| Split | Distribution | n | Accuracy | Weighted F1 |
| :-- | :-- | --: | --: | --: |
| TestA | in-distribution (professional lab camera) | 4,339 | 98.53% | 0.9854 |
| TestB | out-of-distribution (smartphone, Lymphocyte + Neutrophil only) | 2,119 | 89.05% | 0.9111 |
| Combined | | 6,458 | 95.42% | 0.9554 |

These are the paper's figures, which use the full inference stack (Reinhard colour
normalisation, Neutrophil/Lymphocyte routing on TestB, 8x test-time augmentation). Measured with
this notebook's code on the full splits, single pass, no TTA:

| Pipeline | TestA | TestB |
| :-- | --: | --: |
| plain (`medical_enhanced` only) | 97.76% | 56.11% |
| + Reinhard + TestB routing | 98.32% | 88.49% |

Section 7 reproduces both rows. The gap on TestB is the domain-shift problem the paper addresses.

**What this notebook does**

1. Loads the model (with a fix for models saved by a newer Keras than the one installed).
2. Runs it on five bundled example images, no dataset needed.
3. Shows what the preprocessing does.
4. Explains predictions with Grad-CAM.
5. Classifies your own folder of images and writes a CSV.
6. Optionally evaluates on Raabin-WBC TestA/TestB if the dataset is attached.

> **Research use only.** Not a medical device and not for clinical decisions.

**Setup on Kaggle:** *Add Input* and attach this model. Optionally also attach the
[Raabin-WBC dataset](https://raabindata.com/free-data/) for section 7. Source code:
[github.com/yildirimemirhan/wbc-analyzer](https://github.com/yildirimemirhan/wbc-analyzer).
"""

SETUP = """
import base64
import glob
import json
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

CLASS_NAMES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
IMG_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAIN_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

print("TensorFlow", tf.__version__, "| Keras", keras.__version__, "| OpenCV", cv2.__version__)
print("GPU:", tf.config.list_physical_devices("GPU") or "none (CPU is fine for this notebook)")
"""

LOAD_MODEL = """
CUSTOM_OBJECTS = {
    "MedSwish": MedSwish,
    "WBCAttentionBlock": WBCAttentionBlock,
    "WBCFocalLoss": WBCFocalLoss,
    "wbc_focal_loss": WBCFocalLoss,
}

# The model file was saved with Keras 3.14. Older Keras versions reject a few constructor
# arguments that newer versions add to the saved config. They are inert at inference, so we
# drop them from a temporary copy of the config and leave the weights untouched.
_NEWER_KERAS_KWARGS = {"renorm", "renorm_clipping", "renorm_momentum", "quantization_config"}


def _strip_kwargs(obj):
    if isinstance(obj, dict):
        return {k: _strip_kwargs(v) for k, v in obj.items() if k not in _NEWER_KERAS_KWARGS}
    if isinstance(obj, list):
        return [_strip_kwargs(v) for v in obj]
    return obj


def find_model_file():
    \"\"\"Locate the .keras file: $WBC_MODEL_PATH, then any attached Kaggle input, then data/models.\"\"\"
    explicit = os.environ.get("WBC_MODEL_PATH")
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(f"WBC_MODEL_PATH does not exist: {explicit}")
        return Path(explicit)

    hits = []
    for root in ("/kaggle/input", "/kaggle/working", "data/models"):
        hits += glob.glob(os.path.join(root, "**", "*.keras"), recursive=True)
    hits = sorted(set(hits), key=lambda p: ("wbc" not in os.path.basename(p).lower(), p))
    if not hits:
        raise FileNotFoundError(
            "No .keras model file found. On Kaggle use 'Add Input' and attach the WBC Analyzer model, "
            "or set the WBC_MODEL_PATH environment variable."
        )
    if len(hits) > 1:
        print("Several .keras files found, using the first:", *hits, sep="\\n  ")
    return Path(hits[0])


def load_wbc_model(path):
    try:
        return keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
    except (TypeError, ValueError) as err:
        print(f"Direct load failed on Keras {keras.__version__} (model was saved with Keras 3.14).")
        print("Retrying with a compatibility copy of the config.")

    compat_path = Path(tempfile.mkdtemp()) / "wbc_model_compat.keras"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(compat_path, "w", zipfile.ZIP_STORED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                data = json.dumps(_strip_kwargs(json.loads(data))).encode("utf-8")
            zout.writestr(item, data)
    return keras.models.load_model(compat_path, custom_objects=CUSTOM_OBJECTS, compile=False)


MODEL_PATH = find_model_file()
model = load_wbc_model(MODEL_PATH)
print(f"Loaded {MODEL_PATH.name}: {model.count_params():,} parameters")
print("Input :", model.input_shape)
print("Output:", [(layer, model.get_layer(layer).output.shape[-1]) for layer in ("main_out", "aux_binary_out")])
"""

SAMPLES_MD = """
## 2. Example data

Five Raabin-WBC crops, one per class, are embedded in this notebook. They are real
images from the dataset the model was trained on, **so accuracy on them says little about
generalisation**. Section 7 evaluates on the held-out test splits.

Input contract: any RGB image, resized to 224x224. No other preparation is needed.
"""

SAMPLES_LOAD = """
SAMPLE_DIR = Path(tempfile.mkdtemp()) / "samples"
for _class_name in CLASS_NAMES:
    (SAMPLE_DIR / _class_name).mkdir(parents=True, exist_ok=True)
    (SAMPLE_DIR / _class_name / f"{_class_name.lower()}.jpg").write_bytes(
        base64.b64decode(SAMPLE_IMAGES_B64[_class_name])
    )


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


samples = [(read_rgb(SAMPLE_DIR / c / f"{c.lower()}.jpg"), c) for c in CLASS_NAMES]

fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
for ax, (img, label) in zip(axes, samples):
    ax.imshow(img)
    ax.set_title(f"{label}\\n{img.shape[1]}x{img.shape[0]} px")
    ax.axis("off")
plt.tight_layout()
plt.show()
"""

PREPROCESS_MD = """
## 3. Preprocessing

The model was trained on images passed through `PreprocessingFilters.medical_enhanced`, so
inference must use the same filter. Skipping it hurts accuracy. The odd colours (green red
cells, pale background) are expected: the filter stretches contrast for the network, not for
human eyes.
"""

PREPROCESS = """
def preprocess(image_rgb):
    \"\"\"uint8 RGB image of any size -> float32 [0, 1] array of shape (224, 224, 3).\"\"\"
    resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    return PreprocessingFilters.medical_enhanced(resized)


fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
for col, (img, label) in enumerate(samples):
    axes[0, col].imshow(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    axes[0, col].set_title(label)
    axes[1, col].imshow(preprocess(img))
    for row in (0, 1):
        axes[row, col].axis("off")
axes[0, 0].text(-0.05, 0.5, "original", transform=axes[0, 0].transAxes, rotation=90, ha="right", va="center")
axes[1, 0].text(-0.05, 0.5, "enhanced", transform=axes[1, 0].transAxes, rotation=90, ha="right", va="center")
plt.tight_layout()
plt.show()
"""

PREDICT_MD = """
## 4. Predict

`predict_probabilities` returns an `(N, 5)` array of class probabilities ordered as
`CLASS_NAMES`. `tta=True` averages 8 deterministic flips, rotations and brightness
shifts: slower, slightly more stable.
"""

PREDICT = """
def _main_head(raw):
    \"\"\"The model has two heads; return the 5-class one as a float32 array.\"\"\"
    if isinstance(raw, dict):
        raw = raw["main_out"] if "main_out" in raw else next(iter(raw.values()))
    elif isinstance(raw, (list, tuple)):
        raw = raw[0]
    return np.asarray(raw, dtype=np.float32)


def tta_variants(x):
    return np.stack([
        x, np.fliplr(x), np.flipud(x),
        np.rot90(x, 1), np.rot90(x, 2), np.rot90(x, 3),
        np.clip(x * 1.1, 0.0, 1.0), np.clip(x * 0.9, 0.0, 1.0),
    ]).astype(np.float32)


def predict_probabilities(processed, tta=False, batch_size=32):
    \"\"\"processed: float32 array (N, 224, 224, 3) from preprocess(). Returns (N, 5).\"\"\"
    processed = np.asarray(processed, dtype=np.float32)
    if not tta:
        return _main_head(model.predict(processed, batch_size=batch_size, verbose=0))
    return np.stack([
        _main_head(model.predict(tta_variants(x), verbose=0)).mean(axis=0) for x in processed
    ])


def classify(image_rgb, tta=False):
    probs = predict_probabilities(preprocess(image_rgb)[None], tta=tta)[0]
    top = int(np.argmax(probs))
    return CLASS_NAMES[top], float(probs[top]), probs


rows = []
for img, true_label in samples:
    pred, conf, probs = classify(img)
    rows.append({"true": true_label, "predicted": pred, "confidence": round(conf, 4),
                 "correct": pred == true_label, **{c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)}})
pd.DataFrame(rows)
"""

PROBS_PLOT = """
fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
for ax, (img, true_label) in zip(axes, samples):
    pred, conf, probs = classify(img)
    ax.bar(range(5), probs, color=["#2a9d8f" if i == int(np.argmax(probs)) else "#b8c0c8" for i in range(5)])
    ax.set_xticks(range(5), [c[:4] for c in CLASS_NAMES])
    ax.set_title(f"true: {true_label}\\npred: {pred} ({conf:.0%})", fontsize=10)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("probability")
plt.tight_layout()
plt.show()
"""

GRADCAM_MD = """
## 5. Why did it predict that? (Grad-CAM)

Guided Grad-CAM on the last spatial `Conv2D` layer, masked to the cell so background
staining does not dominate the picture. A sound prediction highlights the nucleus or
granules, not the surrounding red cells.

The map comes from a 7x7 feature grid, so it is coarse and can spill onto neighbouring red
cells. Use it to sanity-check a prediction, not as proof of one.
"""

GRADCAM = """
def _last_spatial_conv_name():
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            shape = layer.output.shape
            if len(shape) == 4 and shape[1] and shape[2] and shape[1] > 1 and shape[2] > 1:
                return layer.name
    raise RuntimeError("No spatial Conv2D layer found for Grad-CAM.")


GRADCAM_LAYER = _last_spatial_conv_name()
_grad_model = keras.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(GRADCAM_LAYER).output, model.get_layer("main_out").output],
)


def gradcam_overlay(image_rgb, class_index=None, alpha=0.45):
    \"\"\"Return (overlay_rgb_uint8, predicted_index). Heatmap is None-safe: falls back to the plain image.\"\"\"
    x = tf.convert_to_tensor(preprocess(image_rgb)[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = _grad_model(x, training=False)
        preds = tf.cast(preds, tf.float32)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)

    base = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    if grads is None:
        return base, class_index

    conv_out = tf.cast(conv_out[0], tf.float32)
    grads = tf.cast(grads[0], tf.float32)
    guided = tf.cast(conv_out > 0, tf.float32) * tf.cast(grads > 0, tf.float32) * grads
    weights = tf.reduce_mean(guided, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights, axis=-1)).numpy()
    if cam.max() <= 0:
        return base, class_index

    cam = cv2.GaussianBlur(cv2.resize(cam / cam.max(), (IMG_SIZE, IMG_SIZE)), (0, 0), sigmaX=2.0)
    mask = PreprocessingFilters.estimate_foreground_mask(base)
    cam = cam * (0.05 + 0.95 * mask)
    cam = cam / (cam.max() + 1e-8)

    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    strength = (alpha * np.clip(cam, 0, 1))[..., None]
    overlay = (base.astype(np.float32) * (1 - strength) + heat.astype(np.float32) * strength).astype(np.uint8)
    return overlay, class_index


fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
for col, (img, true_label) in enumerate(samples):
    overlay, idx = gradcam_overlay(img)
    axes[0, col].imshow(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    axes[0, col].set_title(f"true: {true_label}")
    axes[1, col].imshow(overlay)
    axes[1, col].set_title(f"Grad-CAM -> {CLASS_NAMES[idx]}")
    for row in (0, 1):
        axes[row, col].axis("off")
plt.tight_layout()
plt.show()
"""

OWN_MD = """
## 6. Classify your own images

Point `IMAGE_DIR` at any folder of images (searched recursively). On Kaggle that is usually
`/kaggle/input/<your-dataset>/...`. By default it runs on the bundled samples so this cell
works out of the box. Results are written to `predictions.csv`.
"""

OWN = """
IMAGE_DIR = SAMPLE_DIR          # <- change me, e.g. Path("/kaggle/input/my-smears")
USE_TTA = False
OUTPUT_CSV = Path("/kaggle/working/predictions.csv") if Path("/kaggle/working").is_dir() else Path("predictions.csv")

def predict_folder(folder, tta=False, batch_size=32):
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images found under {folder}")

    records = []
    for start in range(0, len(paths), batch_size):
        chunk, batch = [], []
        for path in paths[start:start + batch_size]:
            try:
                batch.append(preprocess(read_rgb(path)))
                chunk.append(path)
            except ValueError as err:
                print("skipped:", err)
        if not batch:
            continue
        probs = predict_probabilities(np.stack(batch), tta=tta)
        for path, p in zip(chunk, probs):
            top = int(np.argmax(p))
            records.append({"file": str(path), "predicted": CLASS_NAMES[top], "confidence": float(p[top]),
                            **{f"prob_{c}": float(v) for c, v in zip(CLASS_NAMES, p)}})
    return pd.DataFrame(records)


predictions = predict_folder(IMAGE_DIR, tta=USE_TTA)
predictions.to_csv(OUTPUT_CSV, index=False)
print(f"{len(predictions)} images -> {OUTPUT_CSV}")
predictions[["file", "predicted", "confidence"]].assign(
    file=lambda d: d["file"].str.replace(str(IMAGE_DIR), "", regex=False)
).round(4)
"""

EVAL_MD = """
## 7. Optional: evaluate on Raabin-WBC TestA / TestB

Attach the Raabin-WBC dataset (folders `Train/`, `TestA/`, `TestB/`, each with one subfolder
per class) and this section finds it automatically. It evaluates a random `SAMPLE_FRACTION`
of each split, keeping the original class proportions so the numbers stay comparable to the
full-split results above. Set `SAMPLE_FRACTION = None` to use everything.

| Switch | What it does | Paper protocol |
| :-- | :-- | :-- |
| `USE_REINHARD` | match each image's LAB statistics to the Train set | on |
| `TESTB_BINARY_ROUTING` | TestB has only Lymphocyte and Neutrophil, so pick between those two | on |
| `USE_TTA_EVAL` | 8x test-time augmentation | on (slow without a GPU) |

Turn `USE_REINHARD` and `TESTB_BINARY_ROUTING` off to see the single-pass baseline. The gap
on TestB is the domain-shift story from the paper.
"""

EVAL = """
SAMPLE_FRACTION = 0.15    # share of each class to evaluate; None = every image
USE_REINHARD = True
TESTB_BINARY_ROUTING = True
USE_TTA_EVAL = False
SEED = 42


def find_raabin_root():
    explicit = os.environ.get("WBC_DATA_ROOT")
    roots = [explicit] if explicit else []
    roots += ["/kaggle/input", "data/raabin-wbc-data", "../data/raabin-wbc-data"]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for current, dirs, _ in os.walk(root):
            if current.count(os.sep) - base_depth >= 5:
                dirs[:] = []
                continue
            if "TestA" in dirs and "TestB" in dirs:
                return Path(current)
    return None


def collect(split_dir, fraction, seed=SEED):
    rng = np.random.default_rng(seed)
    items = []
    for class_name in CLASS_NAMES:
        class_dir = Path(split_dir) / class_name
        if not class_dir.is_dir():
            continue
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if fraction is not None and files:
            keep = max(1, round(len(files) * fraction))
            files = [files[i] for i in sorted(rng.choice(len(files), keep, replace=False))]
        items += [(p, MAIN_IDX[class_name]) for p in files]
    return items


def estimate_reinhard_reference(train_dir, fraction=0.12, seed=SEED):
    # ~1,200 Train images, as in the paper's evaluation script
    items = collect(train_dir, fraction=fraction, seed=seed)
    means, stds = [], []
    for path, _ in items:
        lab = cv2.cvtColor(cv2.resize(read_rgb(path), (IMG_SIZE, IMG_SIZE)), cv2.COLOR_RGB2LAB).astype(np.float32)
        means.append(lab.mean(axis=(0, 1)))
        stds.append(lab.std(axis=(0, 1)) + 1e-6)
    return np.mean(means, axis=0), np.mean(stds, axis=0)


def reinhard(image_rgb, target_mean, target_std):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    norm = (lab - lab.mean(axis=(0, 1))) / (lab.std(axis=(0, 1)) + 1e-6)
    norm = np.clip(norm * target_std + target_mean, 0, 255).astype(np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_LAB2RGB)


def evaluate_split(split_dir, ref, route_binary):
    items = collect(split_dir, SAMPLE_FRACTION)
    batch, y_true = [], []
    for path, label in items:
        image = cv2.resize(read_rgb(path), (IMG_SIZE, IMG_SIZE))
        if ref is not None:
            image = reinhard(image, *ref)
        batch.append(PreprocessingFilters.medical_enhanced(image))
        y_true.append(label)
    probs = predict_probabilities(np.stack(batch), tta=USE_TTA_EVAL)
    if route_binary:
        keep = [MAIN_IDX["Lymphocyte"], MAIN_IDX["Neutrophil"]]
        y_pred = np.array(keep)[np.argmax(probs[:, keep], axis=1)]
    else:
        y_pred = np.argmax(probs, axis=1)
    return np.array(y_true), y_pred


RAABIN_ROOT = find_raabin_root()
if RAABIN_ROOT is None:
    print("Raabin-WBC not found. Attach it via 'Add Input' (or set WBC_DATA_ROOT) to run this section.")
else:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    print("Dataset:", RAABIN_ROOT)
    ref = estimate_reinhard_reference(RAABIN_ROOT / "Train") if USE_REINHARD else None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, split in zip(axes, ("TestA", "TestB")):
        y_true, y_pred = evaluate_split(RAABIN_ROOT / split, ref, route_binary=(split == "TestB" and TESTB_BINARY_ROUTING))
        present = sorted(set(y_true) | set(y_pred))
        print(f"\\n===== {split}: n={len(y_true)}, accuracy={accuracy_score(y_true, y_pred):.4f} =====")
        print(classification_report(y_true, y_pred, labels=present,
                                    target_names=[CLASS_NAMES[i] for i in present], zero_division=0, digits=4))
        cm = confusion_matrix(y_true, y_pred, labels=range(5))
        ax.imshow(cm, cmap="Blues")
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               xlabel="predicted", ylabel="true", title=f"{split} (n={len(y_true)})")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.show()
"""

OUTRO = """
## Notes and limits

- **Input**: one centred leukocyte per image (Raabin-WBC style crops, Giemsa stain). It does not
  detect or segment cells in a full field of view.
- **Classes**: five normal leukocyte types only. Blasts, band cells, artefacts and anything else
  will still be forced into one of the five.
- **Domain shift**: accuracy drops on unseen cameras and stains (see TestB above). Treat low
  confidence as a signal to review manually.
- **Keras versions**: the loader handles models saved by Keras 3.14 on older Keras 3.x. If loading
  still fails, run `pip install -U keras tensorflow` and restart the kernel.

## Citation

```bibtex
@article{yildirim2026wbc,
  title={Achieving Robust Out-of-Distribution Generalization in Peripheral Blood Smears via Custom
         Attention Mechanisms, Medical Enhanced Filtering, and Inference-Time Domain Adaptation},
  author={Yildirim, Emirhan},
  doi={10.13140/RG.2.2.34201.79208},
  year={2026}
}
```

The sample images and test splits come from Raabin-WBC (Kouzehkanan et al., *Scientific Reports*
12, 1123, 2022). Please cite it too and check its license before redistributing the data.
Code and model are released under Apache-2.0.
"""


def build():
    cells = [
        md(TITLE),
        md("## 1. Setup and model"),
        code(SETUP),
        md(
            "The three cells below are verbatim copies of `src/custom_layers.py`, `src/custom_losses.py`"
            " and `src/preprocessing.py`. They must be defined before the model can be deserialised."
        ),
        *[verbatim(name) for name in SOURCE_FILES],
        code(LOAD_MODEL),
        md(SAMPLES_MD),
        sample_images_cell(),
        code(SAMPLES_LOAD),
        md(PREPROCESS_MD),
        code(PREPROCESS),
        md(PREDICT_MD),
        code(PREDICT),
        code(PROBS_PLOT),
        md(GRADCAM_MD),
        code(GRADCAM),
        md(OWN_MD),
        code(OWN),
        md(EVAL_MD),
        code(EVAL),
        md(OUTRO),
    ]
    for index, cell in enumerate(cells):
        digest = hashlib.sha1(f"{index}:{cell['source']}".encode("utf-8")).hexdigest()
        cell["id"] = digest[:8]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def main():
    notebook = build()
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(notebook['cells'])} cells, {OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
