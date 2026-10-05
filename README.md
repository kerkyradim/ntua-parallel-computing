# NTUA — Parallel matrix multiplication for neural networks

**Program:** MSc Data Science & Machine Learning, NTUA (ECE)  
**Course:** Parallel Computing  
**Author:** [Kerkyra Dimisianou](https://github.com/kerkyradim)

Compare **sequential**, **multiprocessing**, and **CUDA** implementations of matrix multiplication used in a small neural-network training loop.

---

## Layout

| Path | Description |
|------|-------------|
| `nn.py` | Simple NN training driver |
| `Sequential/SequentialMatMul.py` | Baseline CPU implementation |
| `Multiprocessing/` | Parallel CPU matmul + helpers |
| `CudaGPU/` | CUDA kernels (simple, shared memory, transpose variants) |
| `test*.py` | Benchmark / correctness tests |
| `tasks.txt` | Assignment task list |

---

## Requirements

- Python 3.10+
- NumPy
- **CUDA** toolkit + GPU (for `CudaGPU/` scripts)
- Optional: `numba` / PyCUDA depending on your local setup (see imports in CUDA modules)

---

## Run (examples)

```bash
python testMultiprocessingMatMul.py
python testCUDAMatMul.py
python testNN.py
```

Adjust paths and device settings as in the course environment.

---

## License

Academic coursework — reference with attribution.
