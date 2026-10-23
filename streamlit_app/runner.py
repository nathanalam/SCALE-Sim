"""
Helpers for running SCALE-Sim from the Streamlit app.

Each run is described by a config dict (the parsed .cfg, section -> key -> value) plus a
topology file. Runs are content-addressed: the cfg text + topology text are hashed, and the
results live in results/streamlit_runs/<hash>/ so repeated runs of the same design point
are instant. Simulations run in worker processes so a sweep can use every core.
"""

import configparser
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_ROOT, 'results', 'streamlit_runs')

# Simulate with this checkout's scalesim (the code Part II shows), even if a different copy is
# pip-installed in the environment.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# scalesim always parses a layout file, even when custom layouts are off; this is the same
# default that scalesim/scale.py uses.
DEFAULT_LAYOUT = os.path.join(REPO_ROOT, 'layouts', 'conv_nets', 'test.csv')

# Report columns as written by scalesim/simulator.py, mapped to friendly names.
COMPUTE_COLS = ['LayerID', 'Total Cycles', 'Compute Cycles', 'Stall Cycles',
                'Overall Util %', 'Mapping Efficiency %', 'Compute Util %']
BANDWIDTH_COLS = ['LayerID', 'Avg IFMAP SRAM BW', 'Avg FILTER SRAM BW', 'Avg OFMAP SRAM BW',
                  'Avg IFMAP DRAM BW', 'Avg FILTER DRAM BW', 'Avg OFMAP DRAM BW']


def read_cfg(path):
    """Parse a SCALE-Sim .cfg into {section: {key: value}}, preserving key case and order."""
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path)
    return {s: dict(parser.items(s)) for s in parser.sections()}


def cfg_to_text(cfg):
    """Serialize a config dict back to .cfg text in the same style as configs/*.cfg."""
    lines = []
    for section, items in cfg.items():
        lines.append('[' + section + ']')
        for key, value in items.items():
            lines.append(key + ' : ' + str(value))
        lines.append('')
    return '\n'.join(lines)


def run_key(cfg_text, topo_text, save_traces):
    """Content hash identifying one design point."""
    h = hashlib.sha1()
    h.update(cfg_text.encode())
    h.update(topo_text.encode())
    h.update(b'traces' if save_traces else b'notraces')
    return h.hexdigest()[:12]


def _clean_log(text):
    # tqdm writes carriage-return progress bars; keep only meaningful lines.
    out = []
    for line in text.replace('\r', '\n').split('\n'):
        if 'it/s' in line or 's/it' in line or re.match(r'^\s*\d+%\|', line):
            continue
        if line.strip() == '' and out and out[-1].strip() == '':
            continue
        out.append(line)
    return '\n'.join(out).strip()


def _simulate(cfg_text, topo_path, out_dir, save_traces):
    """
    Worker entry point: run one SCALE-Sim simulation in this process. Mirrors what
    scalesim/scale.py does, but lets us choose whether to save traces and capture stdout.
    """
    from scalesim.scale_sim import scalesim

    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, 'config.cfg')
    with open(cfg_path, 'w') as f:
        f.write(cfg_text)

    buf = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        sim = scalesim(save_disk_space=not save_traces,
                       verbose=True,
                       config=cfg_path,
                       topology=topo_path,
                       layout=DEFAULT_LAYOUT,
                       input_type_gemm=False)
        if save_traces:
            # single_layer_sim.save_traces() shells out to `mkdir`, which fails on Windows
            # with forward-slash paths, so create the per-layer trace dirs up front.
            report_dir = os.path.join(out_dir, sim.config.get_run_name())
            for i in range(sim.topo.get_num_layers()):
                os.makedirs(os.path.join(report_dir, 'layer' + str(i)), exist_ok=True)
        sim.run_scale(top_path=out_dir)
    elapsed = time.time() - start

    with open(os.path.join(out_dir, 'stdout.txt'), 'w') as f:
        f.write(_clean_log(buf.getvalue()))
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump({'elapsed_s': elapsed, 'topology': topo_path}, f)
    return out_dir


def _report_dir(out_dir):
    """The simulator nests reports under <out_dir>/<run_name>/."""
    for name in os.listdir(out_dir):
        sub = os.path.join(out_dir, name)
        if os.path.isdir(sub) and os.path.exists(os.path.join(sub, 'COMPUTE_REPORT.csv')):
            return sub
    return None


def _read_report(path, columns=None):
    df = pd.read_csv(path, skipinitialspace=True)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df.columns = [c.strip() for c in df.columns]
    if columns is not None:
        df.columns = columns
    return df


def load_result(out_dir):
    """Load all reports for a finished run into a dict of DataFrames + summary metrics."""
    rdir = _report_dir(out_dir)
    compute = _read_report(os.path.join(rdir, 'COMPUTE_REPORT.csv'), COMPUTE_COLS)
    bandwidth = _read_report(os.path.join(rdir, 'BANDWIDTH_REPORT.csv'), BANDWIDTH_COLS)
    detail = _read_report(os.path.join(rdir, 'DETAILED_ACCESS_REPORT.csv'))
    time_report = _read_report(os.path.join(rdir, 'TIME_REPORT.csv'))

    log_path = os.path.join(out_dir, 'stdout.txt')
    log = open(log_path).read() if os.path.exists(log_path) else ''
    meta_path = os.path.join(out_dir, 'meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    traces = {}
    for layer_dir in sorted(os.listdir(rdir)):
        full = os.path.join(rdir, layer_dir)
        if layer_dir.startswith('layer') and os.path.isdir(full):
            files = sorted(f for f in os.listdir(full) if f.endswith('.csv'))
            if files:
                traces[layer_dir] = {f: os.path.join(full, f) for f in files}

    return {
        'out_dir': out_dir,
        'report_dir': rdir,
        'compute': compute,
        'bandwidth': bandwidth,
        'detail': detail,
        'time': time_report,
        'summary': summarize(compute, bandwidth),
        'log': log,
        'elapsed_s': meta.get('elapsed_s'),
        'traces': traces,
    }


def summarize(compute, bandwidth):
    """
    Collapse per-layer reports into one row per run. Cycles are summed across layers;
    utilizations and bandwidths are averaged weighted by each layer's compute cycles, which
    is exactly the single-layer value when the topology has one layer (e.g. alexnet_part).
    """
    w = compute['Compute Cycles'].astype(float)
    wsum = w.sum() if w.sum() else 1.0

    def wavg(series):
        return float((series.astype(float) * w).sum() / wsum)

    out = {
        'Total Cycles': int(compute['Total Cycles'].sum()),
        'Compute Cycles': int(compute['Compute Cycles'].sum()),
        'Stall Cycles': int(compute['Stall Cycles'].sum()),
        'Overall Util %': wavg(compute['Overall Util %']),
        'Mapping Efficiency %': wavg(compute['Mapping Efficiency %']),
        'Compute Util %': wavg(compute['Compute Util %']),
    }
    for col in BANDWIDTH_COLS[1:]:
        out[col] = wavg(bandwidth[col])
    return out


def is_cached(cfg_text, topo_path, save_traces):
    key = run_key(cfg_text, open(topo_path).read(), save_traces)
    out_dir = os.path.join(RUNS_DIR, key)
    return os.path.isdir(out_dir) and _report_dir(out_dir) is not None


def run_many(jobs, max_workers=None, on_progress=None):
    """
    Run a list of jobs, each a dict with 'label', 'cfg' (config dict), 'topology' (path) and
    optionally 'save_traces'. Cached design points are loaded from disk instead of re-run.
    Returns {label: result dict} in the order of `jobs`; a failed job's result is
    {'error': message}. `on_progress(done, total, label,
    cached)` is called as each job finishes.
    """
    pending = {}
    results = {}
    total = len(jobs)
    done = 0

    for job in jobs:
        cfg_text = cfg_to_text(job['cfg'])
        topo_path = os.path.abspath(job['topology'])
        save_traces = job.get('save_traces', False)
        key = run_key(cfg_text, open(topo_path).read(), save_traces)
        out_dir = os.path.join(RUNS_DIR, key)
        if os.path.isdir(out_dir) and _report_dir(out_dir) is not None:
            results[job['label']] = load_result(out_dir)
            done += 1
            if on_progress:
                on_progress(done, total, job['label'], True)
        else:
            pending[job['label']] = (cfg_text, topo_path, out_dir, save_traces)

    if pending:
        workers = max_workers or min(len(pending), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_simulate, *args): label for label, args in pending.items()}
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    results[label] = load_result(fut.result())
                except Exception as exc:  # one bad design point shouldn't sink a sweep
                    results[label] = {'error': '{}: {}'.format(type(exc).__name__, exc)}
                done += 1
                if on_progress:
                    on_progress(done, total, label, False)

    return {job['label']: results[job['label']] for job in jobs}
