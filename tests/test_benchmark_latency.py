import json
import zipfile

import keras
import numpy as np
import pytest

import benchmark_latency as bl


def _tiny_model():
    inp = keras.Input(shape=(8,))
    x = keras.layers.Dense(4, activation="relu")(inp)
    x = keras.layers.BatchNormalization()(x)
    out = keras.layers.Dense(3, activation="softmax", dtype="float32")(x)
    return keras.Model(inp, out)


def test_to_float32_config_rewrites_nested_dtypes_only():
    cfg = {
        "name": "dense_float16_lookalike",
        "dtype": {"class_name": "DTypePolicy", "config": {"name": "mixed_float16"}},
        "layers": [
            {"config": {"dtype": "float16", "units": 4}},
            {"config": {"dtype": "float32", "name": "head"}},
            {"inbound": [{"config": {"dtype": "float16", "shape": [None, 4]}}]},
        ],
    }
    out = bl.to_float32_config(cfg)
    assert out["dtype"]["config"]["name"] == "float32"
    assert out["layers"][0]["config"] == {"dtype": "float32", "units": 4}
    assert out["layers"][1]["config"] == {"dtype": "float32", "name": "head"}
    assert out["layers"][2]["inbound"][0]["config"]["dtype"] == "float32"
    # Layer names that merely contain "float16" are not touched.
    assert out["name"] == "dense_float16_lookalike"
    assert "float16" not in json.dumps(out).replace("dense_float16_lookalike", "")


def test_load_model_float32_strips_mixed_precision_and_keeps_weights(tmp_path):
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = _tiny_model()
    finally:
        keras.mixed_precision.set_global_policy("float32")
    assert model.layers[1].compute_dtype == "float16"
    path = tmp_path / "tiny.keras"
    model.save(path)

    work = tmp_path / "work"
    work.mkdir()
    loaded = bl.load_model_float32(path, work)

    assert all(layer.compute_dtype == "float32" for layer in loaded.layers)
    for a, b in zip(model.get_weights(), loaded.get_weights()):
        np.testing.assert_array_equal(a, b)


def _fake_archive(path, version):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("metadata.json", json.dumps({"keras_version": version}))
    return path


def test_check_keras_version_rejects_newer_archive(tmp_path):
    p = _fake_archive(tmp_path / "new.keras", "99.0.0")
    with pytest.raises(RuntimeError, match="saved with Keras 99.0.0"):
        bl.check_keras_version(p)


def test_check_keras_version_accepts_same_or_older(tmp_path):
    bl.check_keras_version(_fake_archive(tmp_path / "same.keras", keras.__version__))
    bl.check_keras_version(_fake_archive(tmp_path / "old.keras", "3.0.0"))


def test_compiled_and_eager_agree_and_bad_mode_raises():
    model = _tiny_model()
    x = np.random.default_rng(0).random((1, 8), dtype=np.float32)
    eager = bl.make_infer_fn(model, "eager")(x).numpy()
    compiled = bl.make_infer_fn(model, "compiled")(x).numpy()
    np.testing.assert_allclose(eager, compiled, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="Unknown mode"):
        bl.make_infer_fn(model, "xla")


def test_time_model_counts_runs_and_excludes_warmup():
    calls = []

    def fake_infer(x):
        calls.append(x.shape)
        return np.zeros((1, 5), dtype=np.float32)

    times = bl.time_model(fake_infer, warmup=3, runs=7)
    assert len(times) == 7
    assert len(calls) == 10
    assert calls[0] == (1, bl.IMG_SIZE, bl.IMG_SIZE, 3)
    assert all(t >= 0 for t in times)


def test_summarize_stats():
    s = bl.summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["mean_ms"] == pytest.approx(22.0)
    assert s["median_ms"] == pytest.approx(3.0)
    assert s["min_ms"] == pytest.approx(1.0)
    assert s["p95_ms"] == pytest.approx(80.8)
    assert s["runs"] == 5
    with pytest.raises(ValueError):
        bl.summarize([])


def test_format_table_has_one_row_per_model():
    rows = [
        {"model": "A", "total_params": 1_500_000, "mean_ms": 1.234, "median_ms": 1.2, "p95_ms": 2.0},
        {"model": "B", "total_params": 7_830_000, "mean_ms": 14.0, "median_ms": 13.9, "p95_ms": 15.5},
    ]
    table = bl.format_table(rows, "compiled").splitlines()
    assert len(table) == 4
    assert "Mean (compiled)" in table[0]
    assert table[3] == "| B | 7.83 M | 14.0 ms | 13.9 ms | 15.5 ms |"
