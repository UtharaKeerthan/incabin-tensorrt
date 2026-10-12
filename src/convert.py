"""Inspect TFLite models, convert them to ONNX, and verify the conversion.

Workflow (notebook 01):
    tflite_summary()   -> what ops and tensors are in the TFLite model?
    tf2onnx_convert()  -> TFLite -> ONNX
    onnx_summary()     -> what did the converter produce?
    run_tflite() / run_onnx() + compare_outputs() -> is it numerically the same?
"""

from __future__ import annotations

import collections
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# TFLite inspection
# --------------------------------------------------------------------------
def tflite_op_counts(path: Path) -> collections.Counter:
    """Count operators by parsing the .tflite flatbuffer directly.

    We parse the file instead of loading it in an interpreter because an
    interpreter refuses to load models with unregistered *custom* ops,
    and custom ops are exactly what we want to detect before converting.
    """
    from tf2onnx.tflite.BuiltinOperator import BuiltinOperator
    from tf2onnx.tflite.Model import Model

    names = {v: k for k, v in vars(BuiltinOperator).items() if not k.startswith("_")}
    buf = Path(path).read_bytes()
    model = Model.GetRootAsModel(buf, 0)

    codes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        custom = oc.CustomCode()
        if custom:
            codes.append("CUSTOM:" + custom.decode())
        else:
            # Newer files store the code in BuiltinCode, older in DeprecatedBuiltinCode.
            code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
            codes.append(names.get(code, f"UNKNOWN_{code}"))

    counts: collections.Counter = collections.Counter()
    for s in range(model.SubgraphsLength()):
        sg = model.Subgraphs(s)
        for j in range(sg.OperatorsLength()):
            counts[codes[sg.Operators(j).OpcodeIndex()]] += 1
    return counts


def load_interpreter(path: Path):
    """TFLite interpreter: LiteRT (new name) if installed, else tf.lite."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it


def tflite_io(path: Path) -> Tuple[List[dict], List[dict]]:
    """Input and output tensor details (name, shape, dtype)."""
    it = load_interpreter(path)
    keep = lambda d: {"name": d["name"], "shape": tuple(int(x) for x in d["shape"]), "dtype": np.dtype(d["dtype"]).name}
    return [keep(d) for d in it.get_input_details()], [keep(d) for d in it.get_output_details()]


def run_tflite(path_or_interp, inputs: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Run once. ``inputs`` must follow the interpreter's input order."""
    it = load_interpreter(path_or_interp) if isinstance(path_or_interp, (str, Path)) else path_or_interp
    for d, x in zip(it.get_input_details(), inputs):
        it.set_tensor(d["index"], x.astype(d["dtype"]))
    it.invoke()
    return {d["name"]: it.get_tensor(d["index"]).copy() for d in it.get_output_details()}


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------
def tf2onnx_convert(
    tflite_path: Path,
    onnx_path: Path,
    opset: int = 17,
    nchw_inputs: Optional[Sequence[str]] = None,
    extra_args: Sequence[str] = (),
) -> Tuple[bool, str]:
    """Convert TFLite -> ONNX with the tf2onnx command line tool.

    Args:
        nchw_inputs: input tensor names to re-declare as NCHW. TFLite is
            NHWC; TensorRT kernels prefer NCHW. Without this flag the
            converter wraps the graph in Transpose nodes.
    Returns:
        (success, converter log)
    """
    cmd = [sys.executable, "-m", "tf2onnx.convert", "--tflite", str(tflite_path),
           "--output", str(onnx_path), "--opset", str(opset)]
    if nchw_inputs:
        cmd += ["--inputs-as-nchw", ",".join(nchw_inputs)]
    cmd += list(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    # Drop TensorFlow's start-up noise so the converter's own messages are readable.
    noise = ("I0000", "W0000", "oneDNN", "cpu_feature_guard", "To enable the following",
             "WARNING: All log messages", "absl::InitializeLog")
    log = "\n".join(l for l in log.splitlines() if not any(n in l for n in noise))
    ok = proc.returncode == 0 and Path(onnx_path).exists()
    return ok, log


# --------------------------------------------------------------------------
# ONNX inspection
# --------------------------------------------------------------------------
def _shape(vi) -> tuple:
    return tuple(d.dim_value if d.dim_value else (d.dim_param or "?") for d in vi.type.tensor_type.shape.dim)


def onnx_op_counts(path: Path) -> collections.Counter:
    import onnx

    m = onnx.load(str(path))
    return collections.Counter(n.op_type for n in m.graph.node)


def onnx_summary(path: Path) -> Dict[str, object]:
    import onnx

    m = onnx.load(str(path))
    params = sum(int(np.prod(t.dims)) for t in m.graph.initializer)
    return {
        "file": Path(path).name,
        "size_kB": round(Path(path).stat().st_size / 1024, 1),
        "opset": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
        "nodes": len(m.graph.node),
        "parameters": params,
        "inputs": [(i.name, _shape(i)) for i in m.graph.input],
        "outputs": [(o.name, _shape(o)) for o in m.graph.output],
    }


def set_batch_dim(onnx_in: Path, onnx_out: Path, value="batch") -> None:
    """Rename (or fix) the first dimension of every input and output.

    tf2onnx names an unknown batch dim something like 'unk__63'. A readable
    name ('batch') helps later when TensorRT optimization profiles refer to
    it; an integer (1) makes the model fully static.
    """
    import onnx

    m = onnx.load(str(onnx_in))
    for vi in list(m.graph.input) + list(m.graph.output):
        d = vi.type.tensor_type.shape.dim[0]
        if isinstance(value, int):
            d.ClearField("dim_param")
            d.dim_value = value
        else:
            d.ClearField("dim_value")
            d.dim_param = value
    onnx.checker.check_model(m)
    onnx.save(m, str(onnx_out))


# --------------------------------------------------------------------------
# Running ONNX and comparing outputs
# --------------------------------------------------------------------------
def make_ort_session(path: Path, provider: str = "cpu"):
    """ONNX Runtime session on 'cpu' or 'cuda'."""
    import onnxruntime as ort

    providers = {"cpu": ["CPUExecutionProvider"],
                 "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"]}[provider]
    sess = ort.InferenceSession(str(path), providers=providers)
    if provider == "cuda" and "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError("CUDA provider failed to load; check onnxruntime-gpu vs CUDA/cuDNN versions")
    return sess


def run_onnx(sess, inputs: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    feeds = {i.name: x.astype(np.float32) for i, x in zip(sess.get_inputs(), inputs)}
    outs = sess.run(None, feeds)
    return {o.name: v for o, v in zip(sess.get_outputs(), outs)}


def compare_outputs(ref: Dict[str, np.ndarray], test: Dict[str, np.ndarray], label: str = "test"):
    """Per-output error metrics of ``test`` against ``ref``.

    Outputs are matched by name when names agree, otherwise by position.

    Metrics:
        max_abs   : worst single-element difference
        mean_abs  : average difference
        rel_max   : max_abs divided by the largest |ref| value (scale-free)
        cosine    : direction similarity of the flattened outputs (1.0 = identical)
    """
    import pandas as pd

    ref_items, test_items = list(ref.items()), list(test.items())
    if set(ref) == set(test):
        pairs = [(k, ref[k], test[k]) for k in ref]
    else:
        pairs = [(rk, rv, tv) for (rk, rv), (_, tv) in zip(ref_items, test_items)]

    rows = []
    for name, a, b in pairs:
        a = np.asarray(a, np.float64).reshape(-1)
        b = np.asarray(b, np.float64).reshape(-1)
        diff = np.abs(a - b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        rows.append({
            "compare": label,
            "output": name,
            "elements": a.size,
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "rel_max": float(diff.max() / (np.abs(a).max() + 1e-12)),
            "cosine": float(a @ b / denom) if denom > 0 else float("nan"),
        })
    return pd.DataFrame(rows)
