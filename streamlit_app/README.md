# SCALE-Sim Explorer (Streamlit)

An interactive companion for the *Simulators Part I – ScaleSim* exercises. It runs SCALE-Sim for
you, renders the results, and explains them. Each tab maps to one section of the handout:

| Tab | What it does |
|---|---|
| **Part I – Run** | Pick a config (and optionally tweak knobs), run it, and see the terminal metrics, every report CSV with each column explained, how the layer folds onto the array, and an optional cycle-by-cycle SRAM trace explorer. |
| **Part II – How it works** | The `scale.py → scalesim → simulator → single_layer_sim` call chain, plus the three handout questions answered with code excerpts read live from this checkout. The DRAM-bandwidth formula is checked against your own run's reports. |
| **Exercise I – Compare configs** | A knob-by-knob diff of two `.cfg` files with an explanation of each knob, the 6-subplot comparison the handout asks for, and a data-backed "why are they different" section that includes an analytical fold table. |
| **Exercise II – Design space** | Sweeps array size, `FilterSRAMBankBandwidth` and `IfmapSramSzkB` (3 values each), either one knob at a time (9 runs) or full factorial (27). It can also re-run every point in the other bandwidth mode, and it generates a subplot grid plus a "what changed" summary. |

## Running it

From the repo root:

```bash
# with uv (uses the `app` dependency group in pyproject.toml)
uv run --group app streamlit run streamlit_app/app.py

# or in the course conda env
pip install streamlit plotly
streamlit run streamlit_app/app.py
```

The app always simulates with the `scalesim/` code in this checkout, even if a different copy is
pip-installed. That way the numbers match the code shown in Part II.

### On the cluster

Start the app on your interactive GPU node, then forward the port to your laptop:

```bash
# on the compute node (inside srun)
streamlit run streamlit_app/app.py --server.headless true --server.port 8501
hostname   # e.g. gpu-node-07

# on your laptop
ssh -L 8501:gpu-node-07:8501 <username>@<cluster-login-host>
# then open http://localhost:8501
```

Set **Parallel simulations** in the sidebar to the number of cores you requested (`-c 8`).

## Notes

- Results are cached in `results/streamlit_runs/<hash>/`, keyed by the exact config and topology,
  so re-running a design point is instant. Each folder contains the `config.cfg` that was run, the
  captured terminal output and the standard SCALE-Sim reports. Clear the cache from the sidebar.
- Trace saving is off by default because traces are about 130 MB per run for `alexnet_part` on a
  32×32 array. Turn it on in Part I to use the trace explorer.
- In USER bandwidth mode, `IfmapSRAMBankBandwidth` and `FilterSRAMBankBandwidth` must be
  multiples of the matching `*SRAMBankNum` (10 by default). Otherwise the simulator hits an
  assertion. The app warns about this before running.
