# In-cabin TensorRT: optimization and profiling on NVIDIA GPUs

Taking MediaPipe's face detector and face landmark models from TFLite through ONNX to TensorRT
(FP32, FP16, INT8) on an NVIDIA T4, with latency breakdowns, roofline analysis and Nsight profiling.

## Notebooks

| # | Notebook | Topic | Status |
|---|---|---|---|
| 00 | `00_setup.ipynb` | Versions, timing methods, measured T4 ceilings, PCIe copy cost | ready |
| 01 | `01_conversion.ipynb` | TFLite → ONNX, NHWC vs NCHW, verification, first latency | ready |
| 02 | `02_first_engine.ipynb` | FP32/FP16 TensorRT engines, layer fusion | planned |
| 03 | `03_roofline.ipynb` | Arithmetic intensity, batch sweep, Nsight Compute | planned |
| 04 | `04_int8.ipynb` | Calibration, mixed precision, accuracy impact | planned |
| 05 | `05_launch_overhead.ipynb` | Kernel launch cost, CUDA Graphs | planned |
| 06 | `06_pipeline_streams.ipynb` | Full video pipeline, GPU preprocessing, CUDA streams | planned |
| 07 | `07_graph_surgery.ipynb` | Unsupported ops: four ways to fix them | planned |

## Shared code (`src/`)

- `bench.py`: wall-clock and CUDA-event benchmarks, p50/p99, stage timer, tables and plots
- `models.py`: download and unpack MediaPipe model bundles
- `convert.py`: TFLite inspection, tf2onnx conversion, ONNX inspection, output comparison

## Running (VS Code + Colab extension)

1. Push this repo to GitHub (public, or the clone in the setup cell needs a token).
2. Set `REPO_URL` in the first code cell of each notebook if your repo URL differs.
3. Open a notebook in VS Code, select the Colab kernel, choose a **T4 GPU** runtime.
4. Run top to bottom. Run 00 before 01.

The Colab runtime is a remote machine: it cannot see your local folder, so each notebook
clones or pulls the repo itself.

## Environment

Recorded by notebook 00 in `results/env.json` (TensorRT, CUDA, driver and GPU versions).
TensorRT engines only run on the TensorRT version and GPU that built them.
