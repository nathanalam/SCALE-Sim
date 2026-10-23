"""
SCALE-Sim Explorer: a Streamlit companion for "Simulators Part I - ScaleSim".

Run from the repo root:
    streamlit run streamlit_app/app.py
"""

import copy
import glob
import itertools
import math
import os
import re
import shutil

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import runner

REPO = runner.REPO_ROOT
CONFIG_DIR = os.path.join(REPO, 'configs')
TOPO_DIR = os.path.join(REPO, 'topologies', 'conv_nets')

# Categorical palette (fixed order, validated for colour-vision deficiency).
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SERIES = [BLUE, ORANGE, AQUA]

# The six metrics the handout asks for in Exercise I, in the handout's order.
CORE_METRICS = ['Total Cycles', 'Compute Cycles', 'Overall Util %', 'Mapping Efficiency %',
                'Avg IFMAP SRAM BW', 'Avg OFMAP DRAM BW']
ALL_METRICS = ['Total Cycles', 'Compute Cycles', 'Stall Cycles', 'Overall Util %',
               'Mapping Efficiency %', 'Compute Util %', 'Avg IFMAP SRAM BW',
               'Avg FILTER SRAM BW', 'Avg OFMAP SRAM BW', 'Avg IFMAP DRAM BW',
               'Avg FILTER DRAM BW', 'Avg OFMAP DRAM BW']

METRIC_HELP = {
    'Total Cycles': 'End-to-end latency: first DRAM read to last OFMAP DRAM write. Includes '
                    'the initial SRAM fill (prefetch). Report column "Total Cycles (incl. '
                    'prefetch)"; printed as "Total cycles" in the terminal.',
    'Compute Cycles': 'Cycles the array spends working through its folds (including any '
                      'stalls). Report column "Total Cycles"; printed as "Compute cycles".',
    'Stall Cycles': 'Cycles the array sat idle waiting on memory. Always 0 in CALC mode, '
                    'because CALC assumes DRAM is as fast as it needs to be.',
    'Overall Util %': 'MACs performed / (compute cycles x number of PEs). How busy the whole '
                      'array is over the run.',
    'Mapping Efficiency %': 'Fraction of PEs that have work mapped to them, averaged over '
                            'folds. Purely a function of layer shape, array shape and dataflow.',
    'Compute Util %': 'Like overall utilization but measured per fold, excluding the time to '
                      'fill and drain the array.',
    'Avg IFMAP SRAM BW': 'Words/cycle the array reads from the IFMAP scratchpad.',
    'Avg FILTER SRAM BW': 'Words/cycle the array reads from the filter scratchpad.',
    'Avg OFMAP SRAM BW': 'Words/cycle the array writes to the OFMAP scratchpad.',
    'Avg IFMAP DRAM BW': 'Words/cycle moved DRAM -> IFMAP SRAM. In CALC mode this is the '
                         'bandwidth needed to never stall; in USER mode it is capped by the '
                         'configured bandwidth.',
    'Avg FILTER DRAM BW': 'Words/cycle moved DRAM -> filter SRAM.',
    'Avg OFMAP DRAM BW': 'Words/cycle written OFMAP SRAM -> DRAM.',
}

UNITS = {m: ('cycles' if 'Cycles' in m else '%' if '%' in m else 'words/cycle')
         for m in ALL_METRICS}

# What each .cfg knob does, keyed by (section, key).
KNOB_HELP = {
    ('general', 'run_name'): 'Name of the output folder inside results/. No effect on numbers.',
    ('architecture_presets', 'ArrayHeight'): 'Rows of processing elements (PEs) in the '
                                             'systolic array.',
    ('architecture_presets', 'ArrayWidth'): 'Columns of PEs in the systolic array.',
    ('architecture_presets', 'IfmapSramSzkB'): 'IFMAP (input activation) scratchpad size, kB. '
                                               'Double-buffered: half is filled while half is '
                                               'read.',
    ('architecture_presets', 'FilterSramSzkB'): 'Filter (weight) scratchpad size, kB.',
    ('architecture_presets', 'OfmapSramSzkB'): 'OFMAP (output) scratchpad size, kB.',
    ('architecture_presets', 'IfmapOffset'): 'Base address for IFMAP in the traces, so each '
                                             'operand has its own address range.',
    ('architecture_presets', 'FilterOffset'): 'Base address for filters in the traces.',
    ('architecture_presets', 'OfmapOffset'): 'Base address for OFMAP in the traces.',
    ('architecture_presets', 'Bandwidth'): 'DRAM interface bandwidth (words/cycle). Read only '
                                           'in USER mode; used for the OFMAP write-back path.',
    ('architecture_presets', 'Dataflow'): 'Which operand stays put in the PEs: ws = weight '
                                          'stationary, os = output stationary, is = input '
                                          'stationary. Decides how the layer maps onto rows '
                                          'and columns.',
    ('architecture_presets', 'MemoryBanks'): 'Number of DRAM banks (used by the Ramulator '
                                             'flow).',
    ('architecture_presets', 'ReadRequestBuffer'): 'Outstanding read-request queue depth '
                                                   '(Ramulator flow).',
    ('architecture_presets', 'WriteRequestBuffer'): 'Outstanding write-request queue depth '
                                                    '(Ramulator flow).',
    ('layout', 'IfmapCustomLayout'): 'Use a custom on-chip data layout for IFMAP.',
    ('layout', 'IfmapSRAMBankBandwidth'): 'USER mode only: words/cycle that can be filled into '
                                          'the IFMAP SRAM from DRAM. Must be a multiple of '
                                          'IfmapSRAMBankNum.',
    ('layout', 'IfmapSRAMBankNum'): 'Number of IFMAP SRAM banks.',
    ('layout', 'IfmapSRAMBankPort'): 'Ports per IFMAP SRAM bank.',
    ('layout', 'FilterCustomLayout'): 'Use a custom on-chip data layout for filters.',
    ('layout', 'FilterSRAMBankBandwidth'): 'USER mode only: words/cycle that can be filled into '
                                           'the filter SRAM from DRAM. Must be a multiple of '
                                           'FilterSRAMBankNum.',
    ('layout', 'FilterSRAMBankNum'): 'Number of filter SRAM banks.',
    ('layout', 'FilterSRAMBankPort'): 'Ports per filter SRAM bank.',
    ('run_presets', 'InterfaceBandwidth'): 'CALC: assume DRAM keeps up and report the '
                                           'bandwidth that would need. USER: cap DRAM bandwidth '
                                           'at the configured values and model the stalls.',
    ('run_presets', 'UseRamulatorTrace'): 'Drive DRAM timing from a Ramulator trace.',
}

# What the per-cycle access pattern of each SRAM trace shows, by (dataflow, operand).
TRACE_NOTES = {
    ('ws', 'FILTER'): 'Weight stationary: each spike is the array loading a new tile of weights '
                      'at the start of a fold, and the weights then stay put. Count the spikes '
                      'to count the folds.',
    ('ws', 'IFMAP'): 'Weight stationary: inputs stream in continuously while weights stay put. '
                     'Dips are fold boundaries, where the array drains and loads new weights.',
    ('ws', 'OFMAP'): 'Weight stationary: partial sums flow out of the bottom of the array every '
                     'cycle once it is full. Dips are fold boundaries.',
    ('os', 'OFMAP'): 'Output stationary: outputs accumulate in place and are written out only '
                     'at the end of each fold, hence the spikes.',
}

DATAFLOW_NAMES = {'ws': 'Weight stationary', 'os': 'Output stationary',
                  'is': 'Input stationary'}

st.set_page_config(page_title='SCALE-Sim Explorer', page_icon=':material/grid_on:',
                   layout='wide')


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def rel(path):
    return os.path.relpath(path, REPO).replace('\\', '/')


def fmt(value, metric):
    if 'Cycles' in metric:
        return '{:,.0f}'.format(value)
    if '%' in metric:
        return '{:.2f}%'.format(value)
    return '{:.3f}'.format(value)


def list_configs():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(CONFIG_DIR, '*.cfg')))


def list_topologies():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(TOPO_DIR, '*.csv')))


def load_topology(name):
    df = pd.read_csv(os.path.join(TOPO_DIR, name), skipinitialspace=True)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df.columns = [c.strip() for c in df.columns]
    df['Layer name'] = df['Layer name'].astype(str).str.strip()
    return df


def layer_shapes(topo_df):
    """
    Per-layer im2col shapes: OFMAP pixels, window size (R*S*C) and number of filters.
    OFMAP size rounds up like topology_utils.topo_calc_hyperparams() does, i.e. a partial last
    window still produces an output pixel (55x55 for AlexNet Conv1, not the textbook 54x54).
    """
    rows = []
    for _, r in topo_df.iterrows():
        s = int(r['Strides'])
        oh = math.ceil((int(r['IFMAP Height']) - int(r['Filter Height']) + s) / s)
        ow = math.ceil((int(r['IFMAP Width']) - int(r['Filter Width']) + s) / s)
        rows.append({'layer': r['Layer name'], 'ofmap_px': oh * ow,
                     'window': int(r['Filter Height']) * int(r['Filter Width'])
                     * int(r['Channels']),
                     'filters': int(r['Num Filter'])})
    return rows


def analytic_mapping(topo_df, rows, cols, dataflow):
    """
    How the layer folds onto the array (SCALE-Sim paper, Sec. III): each dataflow maps two
    of {OFMAP pixels, window, filters} onto rows (S_r) and columns (S_c) and streams the
    third over time (T). Mapping efficiency = S_r*S_c / (folds * rows * cols).
    """
    out = []
    for s in layer_shapes(topo_df):
        if dataflow == 'os':
            sr, sc, t = s['ofmap_px'], s['filters'], s['window']
            names = ('OFMAP pixels', 'filters', 'window (R*S*C)')
        elif dataflow == 'ws':
            sr, sc, t = s['window'], s['filters'], s['ofmap_px']
            names = ('window (R*S*C)', 'filters', 'OFMAP pixels')
        else:
            sr, sc, t = s['window'], s['ofmap_px'], s['filters']
            names = ('window (R*S*C)', 'OFMAP pixels', 'filters')
        fr, fc = math.ceil(sr / rows), math.ceil(sc / cols)
        out.append({'Layer': s['layer'], 'S_r': sr, 'S_c': sc, 'T': t,
                    'rows map': names[0], 'cols map': names[1], 'time streams': names[2],
                    'Row folds': fr, 'Col folds': fc, 'Folds': fr * fc,
                    'Mapping eff %': 100.0 * sr * sc / (fr * fc * rows * cols)})
    return pd.DataFrame(out)


def cfg_value(cfg, key):
    for items in cfg.values():
        if key in items:
            return items[key]
    return None


def set_knob(cfg, key, value):
    """Set a knob by name in whichever section holds it."""
    if key == 'ArraySize':
        cfg['architecture_presets']['ArrayHeight'] = str(value)
        cfg['architecture_presets']['ArrayWidth'] = str(value)
        return
    for items in cfg.values():
        if key in items:
            items[key] = str(value)
            return
    raise KeyError(key)


def validate_cfg(cfg):
    """Catch config combinations that make the simulator assert, before we run them."""
    problems = []
    if cfg['run_presets'].get('InterfaceBandwidth') == 'USER':
        for op in ('Ifmap', 'Filter'):
            bw = int(cfg['layout'][op + 'SRAMBankBandwidth'])
            banks = int(cfg['layout'][op + 'SRAMBankNum'])
            if bw % banks != 0 or bw < banks:
                problems.append('{op}SRAMBankBandwidth={bw} must be a positive multiple of '
                                '{op}SRAMBankNum={banks} in USER mode (read_buffer.py asserts '
                                'this).'.format(op=op, bw=bw, banks=banks))
    return problems


def cli_command(cfg_name, topo_name):
    return ('python3 -m scalesim.scale -c configs/{} -t topologies/conv_nets/{}'
            .format(cfg_name, topo_name))


def run_jobs(jobs, workers):
    """Run jobs with a progress bar; returns {label: result}."""
    # One-at-a-time sweeps revisit the base point once per knob; simulate it once.
    jobs = list({j['label']: j for j in jobs}.values())
    n_cached = sum(runner.is_cached(runner.cfg_to_text(j['cfg']), j['topology'],
                                    j.get('save_traces', False)) for j in jobs)
    n_new = len(jobs) - n_cached
    msg = 'Simulating {} design point{}'.format(n_new, '' if n_new == 1 else 's')
    if n_cached:
        msg += ' ({} cached)'.format(n_cached)
    bar = st.progress(0.0, text=msg + '...')

    def progress(done, total, label, cached):
        bar.progress(done / total, text='{}/{} done - finished {}{}'.format(
            done, total, label, ' (cached)' if cached else ''))

    results = runner.run_many(jobs, max_workers=workers, on_progress=progress)
    bar.empty()
    for label, res in results.items():
        if 'error' in res:
            st.error('**{}** failed: `{}`'.format(label, res['error']))
    return results


def summary_frame(results):
    rows = {label: res['summary'] for label, res in results.items() if 'summary' in res}
    return pd.DataFrame(rows).T


def style_fig(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=60, b=10),
                      font=dict(size=12), hoverlabel=dict(font_size=12),
                      legend=dict(orientation='h', yanchor='bottom', y=1.04, x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.18)', zeroline=False)
    return fig


def show_fig(fig, key):
    st.plotly_chart(fig, key=key, config={'displaylogo': False,
                                          'toImageButtonOptions': {'scale': 2}})


def excerpt(rel_path, pattern, before=0, after=0):
    """Return (line_no, numbered snippet) around the first line matching `pattern`."""
    path = os.path.join(REPO, rel_path)
    lines = open(path).read().split('\n')
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            lo, hi = max(0, i - before), min(len(lines), i + after + 1)
            width = len(str(hi))
            body = '\n'.join('{:>{w}} | {}'.format(n + 1, lines[n], w=width)
                             for n in range(lo, hi))
            return i + 1, body
    return None, '(pattern not found: {})'.format(pattern)


@st.cache_data(show_spinner=False)
def sram_trace_activity(path, max_cols=None):
    """Per-cycle count of valid accesses (address != -1) in an SRAM trace, plus a PE map."""
    cycles, counts, grid = [], [], []
    with open(path) as f:
        for n, line in enumerate(f):
            fields = line.rstrip(',\n').split(',')
            vals = [float(x) for x in fields[1:] if x.strip() != '']
            cycles.append(float(fields[0]))
            counts.append(sum(1 for v in vals if v >= 0))
            if n < 400:
                grid.append([1 if v >= 0 else 0 for v in vals[:max_cols]])
    return pd.DataFrame({'cycle': cycles, 'accesses': counts}), grid


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.title('SCALE-Sim Explorer')
    st.caption('Companion app for *Simulators Part I - ScaleSim*.')

    topos = list_topologies()
    topo_name = st.selectbox(
        'Workload (topology)', topos,
        index=topos.index('alexnet_part.csv') if 'alexnet_part.csv' in topos else 0,
        help='The handout uses alexnet_part.csv (AlexNet Conv1 only). Multi-layer topologies '
             'are summed across layers and take proportionally longer.')
    topo_path = os.path.join(TOPO_DIR, topo_name)
    topo_df = load_topology(topo_name)
    st.dataframe(topo_df, hide_index=True, height=min(38 + 35 * len(topo_df), 250))

    cpu = os.cpu_count() or 1
    workers = st.slider('Parallel simulations', 1, max(cpu, 1), min(cpu, 9),
                        help='Each design point runs in its own process. On the cluster, '
                             'match this to the -c you asked srun for.')

    with st.expander('Result cache'):
        st.caption('Every design point is stored under `{}/` keyed by a hash of its config '
                   'and topology, so re-running is instant.'.format(rel(runner.RUNS_DIR)))
        if st.button('Clear cache', icon=':material/delete:'):
            shutil.rmtree(runner.RUNS_DIR, ignore_errors=True)
            for k in ('p1', 'ex1', 'ex2'):
                st.session_state.pop(k, None)
            st.success('Cleared.')

configs = list_configs()

tab_run, tab_code, tab_ex1, tab_ex2 = st.tabs([
    ':material/play_arrow: Part I - Run',
    ':material/code: Part II - How it works',
    ':material/compare_arrows: Exercise I - Compare configs',
    ':material/grid_view: Exercise II - Design space',
])


# --------------------------------------------------------------------------------------
# Part I - run the simulator and explore its outputs
# --------------------------------------------------------------------------------------

with tab_run:
    st.header('Part I: Running the simulator')
    st.markdown('Pick a hardware config, run it on the workload, then read the same numbers '
                'the terminal prints and the report files written to `results/`.')

    c1, c2 = st.columns([1, 1])
    with c1:
        p1_cfg_name = st.selectbox('Hardware config', configs,
                                   index=configs.index('scale.cfg') if 'scale.cfg' in configs
                                   else 0, key='p1_cfg')
    p1_cfg = runner.read_cfg(os.path.join(CONFIG_DIR, p1_cfg_name))
    arch = p1_cfg['architecture_presets']

    with c2:
        save_traces = st.toggle(
            'Save cycle-by-cycle traces', value=False,
            help='Writes IFMAP/FILTER/OFMAP SRAM and DRAM traces (~130 MB for alexnet_part on '
                 'a 32x32 array). The CLI always saves them; leave this off for speed unless '
                 'you want the trace explorer below.')

    with st.expander('Tweak knobs (optional)', icon=':material/tune:'):
        st.caption('Start from the preset above and override the most important knobs. '
                   'Changes apply to this tab only.')
        k1, k2, k3, k4 = st.columns(4)
        arr_h = k1.number_input('ArrayHeight', 1, 1024, int(arch['ArrayHeight']))
        arr_w = k2.number_input('ArrayWidth', 1, 1024, int(arch['ArrayWidth']))
        df_opts = ['ws', 'os', 'is']
        dataflow = k3.selectbox('Dataflow', df_opts, index=df_opts.index(arch['Dataflow']),
                                format_func=lambda d: '{} ({})'.format(d, DATAFLOW_NAMES[d]))
        bw_opts = ['CALC', 'USER']
        bw_mode = k4.selectbox('InterfaceBandwidth', bw_opts,
                               index=bw_opts.index(p1_cfg['run_presets']['InterfaceBandwidth']))
        k5, k6, k7, k8 = st.columns(4)
        ifm_kb = k5.number_input('IfmapSramSzkB', 1, 65536, int(arch['IfmapSramSzkB']))
        fil_kb = k6.number_input('FilterSramSzkB', 1, 65536, int(arch['FilterSramSzkB']))
        ofm_kb = k7.number_input('OfmapSramSzkB', 1, 65536, int(arch['OfmapSramSzkB']))
        dram_bw = k8.number_input('Bandwidth (USER mode)', 1, 4096,
                                  int(str(arch.get('Bandwidth', '10')).split(',')[0]))

    p1_run_cfg = copy.deepcopy(p1_cfg)
    overrides = {'ArrayHeight': arr_h, 'ArrayWidth': arr_w, 'Dataflow': dataflow,
                 'InterfaceBandwidth': bw_mode, 'IfmapSramSzkB': ifm_kb,
                 'FilterSramSzkB': fil_kb, 'OfmapSramSzkB': ofm_kb, 'Bandwidth': dram_bw}
    changed = []
    for key, value in overrides.items():
        if str(cfg_value(p1_cfg, key)).strip() != str(value):
            set_knob(p1_run_cfg, key, value)
            changed.append('{}={}'.format(key, value))

    st.markdown('**Equivalent command**' + (' (plus your knob overrides: {})'.format(
        ', '.join(changed)) if changed else ''))
    st.code(cli_command(p1_cfg_name, topo_name), language='bash')

    problems = validate_cfg(p1_run_cfg)
    for p in problems:
        st.warning(p)

    if st.button('Run simulation', type='primary', icon=':material/play_arrow:',
                 disabled=bool(problems), key='p1_run'):
        label = p1_cfg_name + (' (modified)' if changed else '')
        res = run_jobs([{'label': label, 'cfg': p1_run_cfg, 'topology': topo_path,
                         'save_traces': save_traces}], workers)
        st.session_state['p1'] = {'label': label, 'res': res[label], 'cfg': p1_run_cfg,
                                  'topo': topo_name}

    p1 = st.session_state.get('p1')
    if p1 and 'error' not in p1['res']:
        res, s = p1['res'], p1['res']['summary']
        st.divider()
        st.subheader('Results: {} on {}'.format(p1['label'], p1['topo']))
        if res.get('elapsed_s'):
            st.caption('Simulated in {:.1f} s.'.format(res['elapsed_s']))

        m = st.columns(5)
        m[0].metric('Total cycles', fmt(s['Total Cycles'], 'Total Cycles'),
                    help=METRIC_HELP['Total Cycles'])
        m[1].metric('Compute cycles', fmt(s['Compute Cycles'], 'Compute Cycles'),
                    help=METRIC_HELP['Compute Cycles'])
        m[2].metric('Stall cycles', fmt(s['Stall Cycles'], 'Stall Cycles'),
                    help=METRIC_HELP['Stall Cycles'])
        m[3].metric('Overall utilization', fmt(s['Overall Util %'], '%'),
                    help=METRIC_HELP['Overall Util %'])
        m[4].metric('Mapping efficiency', fmt(s['Mapping Efficiency %'], '%'),
                    help=METRIC_HELP['Mapping Efficiency %'])

        bw = pd.DataFrame({
            'SRAM (array side)': [s['Avg IFMAP SRAM BW'], s['Avg FILTER SRAM BW'],
                                  s['Avg OFMAP SRAM BW']],
            'DRAM (off-chip side)': [s['Avg IFMAP DRAM BW'], s['Avg FILTER DRAM BW'],
                                     s['Avg OFMAP DRAM BW']],
        }, index=['IFMAP', 'FILTER', 'OFMAP'])
        left, right = st.columns([1, 1])
        with left:
            st.markdown('**Average bandwidth (words/cycle)**')
            st.dataframe(bw.style.format('{:.3f}'))
            st.caption('Read across a row: how many words/cycle each operand needs at the '
                       'array vs. from DRAM. SRAM reuse is what lets the DRAM column be '
                       'smaller.')
        with right:
            st.markdown('**Terminal output**')
            st.code(res['log'] or '(no output captured)', language='text', height=260)

        st.markdown('#### What the simulator wrote to `results/`')
        st.caption('Folder: `{}`'.format(rel(res['report_dir'])))
        r1, r2, r3, r4 = st.tabs(['COMPUTE_REPORT.csv', 'BANDWIDTH_REPORT.csv',
                                  'DETAILED_ACCESS_REPORT.csv', 'TIME_REPORT.csv'])
        with r1:
            st.dataframe(res['compute'], hide_index=True)
            st.markdown(
                '- **Total Cycles** (file: *Total Cycles (incl. prefetch)*): '
                + METRIC_HELP['Total Cycles'] + '\n'
                '- **Compute Cycles** (file: *Total Cycles*): ' + METRIC_HELP['Compute Cycles']
                + '\n- **Stall Cycles**: ' + METRIC_HELP['Stall Cycles']
                + '\n- **Overall Util %**: ' + METRIC_HELP['Overall Util %']
                + '\n- **Mapping Efficiency %**: ' + METRIC_HELP['Mapping Efficiency %']
                + '\n- **Compute Util %**: ' + METRIC_HELP['Compute Util %'])
            st.info('Heads up: the file\'s column called "Total Cycles" is what the terminal '
                    'prints as *Compute cycles*. This app uses the terminal\'s names.',
                    icon=':material/info:')
        with r2:
            st.dataframe(res['bandwidth'], hide_index=True)
            st.markdown('SRAM BW = SRAM accesses / compute cycles. DRAM BW = DRAM accesses / '
                        'the cycles between the first and last DRAM access for that operand '
                        '(see Part II, Q3).')
        with r3:
            st.dataframe(res['detail'], hide_index=True)
            st.markdown('Start/stop cycle and access count for each operand at each level of '
                        'the hierarchy. Negative DRAM start cycles are the prefetch that '
                        'happens before compute begins at cycle 0.')
        with r4:
            st.dataframe(res['time'], hide_index=True)
            st.markdown('Wall-clock estimate per layer. Without a `TimeLinearModel` in the '
                        'config this is just the compute cycle count.')

        with st.expander('How this layer maps onto the array', icon=':material/grid_on:'):
            a = p1['cfg']['architecture_presets']
            mp = analytic_mapping(topo_df, int(a['ArrayHeight']), int(a['ArrayWidth']),
                                  a['Dataflow'])
            st.markdown('For **{}** on a **{}x{}** array, SCALE-Sim lowers the convolution to a '
                        'matrix multiply and maps it like this:'.format(
                            DATAFLOW_NAMES[a['Dataflow']], a['ArrayHeight'], a['ArrayWidth']))
            st.dataframe(mp.style.format({'Mapping eff %': '{:.2f}'}), hide_index=True)
            st.caption('Mapping efficiency here is computed by hand from the layer shape; '
                       'compare it to the simulator\'s Mapping Efficiency above.')

        if res['traces']:
            st.markdown('#### Trace explorer')
            layer = st.selectbox('Layer', list(res['traces']), key='trace_layer')
            files = [f for f in res['traces'][layer] if 'SRAM' in f]
            trace_file = st.radio('SRAM trace', files, horizontal=True, key='trace_file',
                                  index=files.index('IFMAP_SRAM_TRACE.csv')
                                  if 'IFMAP_SRAM_TRACE.csv' in files else 0)
            with st.spinner('Reading trace...'):
                act, grid = sram_trace_activity(res['traces'][layer][trace_file])
            bucket = max(1, len(act) // 600)
            binned = act.groupby(act['cycle'] // bucket * bucket)['accesses'].mean()
            t1, t2 = st.columns([3, 2])
            with t1:
                fig = go.Figure(go.Scatter(x=binned.index, y=binned.values, mode='lines',
                                           line=dict(color=BLUE, width=2),
                                           hovertemplate='cycle %{x:,}<br>%{y:.1f} '
                                                         'words/cycle<extra></extra>'))
                fig.update_layout(title='Words accessed per cycle ({}-cycle average)'.format(
                    bucket), xaxis_title='cycle', yaxis_title='words/cycle')
                show_fig(style_fig(fig, 320), 'trace_line')
                st.caption(TRACE_NOTES.get((p1['cfg']['architecture_presets']['Dataflow'],
                                            trace_file.split('_')[0]),
                                           'Dips are fold boundaries: the array drains and '
                                           'refills before the next tile of the matrix '
                                           'multiply.'))
            with t2:
                fig = go.Figure(go.Heatmap(z=grid, colorscale=[[0, 'rgba(0,0,0,0)'],
                                                               [1, BLUE]],
                                           showscale=False,
                                           hovertemplate='cycle %{y}<br>port %{x}'
                                                         '<extra></extra>'))
                fig.update_layout(title='First 400 cycles: which ports are active',
                                  xaxis_title='array port (row/col)', yaxis_title='cycle',
                                  yaxis_autorange='reversed')
                show_fig(style_fig(fig, 320), 'trace_heat')
                st.caption('The diagonal edge is the systolic skew: data enters one row/column '
                           'per cycle.')
            st.caption('Each trace row is `cycle, addr, addr, ...` with one address per array '
                       'port; -1 means that port is idle that cycle.')
    elif p1 and 'error' in p1['res']:
        st.error(p1['res']['error'])


# --------------------------------------------------------------------------------------
# Part II - code walkthrough
# --------------------------------------------------------------------------------------

with tab_code:
    st.header('Part II: How the simulator works')
    st.markdown('Every snippet below is read live from this checkout, so line numbers match '
                'the files on disk.')

    st.subheader('The call chain')
    st.graphviz_chart('''
digraph G {
  rankdir=LR; node [shape=box, style="rounded", fontname="Helvetica", fontsize=11];
  edge [fontname="Helvetica", fontsize=9];
  cli [label="scalesim/scale.py\\nparse -c -t -p -i"];
  ss [label="scale_sim.py: class scalesim\\nset_params(): read .cfg + topology\\nrun_scale() -> run_once()"];
  sim [label="simulator.py: class simulator\\nrun(): one single_layer_sim per layer\\ngenerate_reports(): *_REPORT.csv"];
  sls [label="single_layer_sim.py\\nrun() / calc_report_data()"];
  op [label="compute/operand_matrix.py\\nIFMAP / filter / OFMAP address matrices"];
  cs [label="compute/systolic_compute_{ws,os,is}.py\\nprefetch + demand matrices per fold"];
  mem [label="memory/double_buffered_scratchpad_mem.py\\nSRAM buffers + DRAM fills, stalls, traces"];
  cli -> ss [label="scalesim(...)"]; ss -> sim [label="runner.run()"];
  sim -> sls [label="per layer"]; sls -> op [label="1. operands"];
  sls -> cs [label="2. dataflow"]; sls -> mem [label="3. memory"];
}
''')

    with st.expander('Where the classes are defined', icon=':material/folder_open:',
                     expanded=False):
        for path, pat in [('scalesim/scale.py', r'^\s+s = scalesim\('),
                          ('scalesim/scale_sim.py', r'^class scalesim'),
                          ('scalesim/simulator.py', r'^class simulator'),
                          ('scalesim/single_layer_sim.py', r'^class single_layer_sim')]:
            line, body = excerpt(path, pat, 0, 8 if 'scale.py' in path else 3)
            st.markdown('`{}:{}`'.format(path, line))
            st.code(body, language='python')

    st.subheader('Questions from the handout')

    st.markdown('##### Q1. Which function returns the operand matrix for the filter?')
    line, body = excerpt('scalesim/single_layer_sim.py', r'get_filter_matrix\(\)', 2, 2)
    st.code(body, language='python')
    st.markdown('`single_layer_sim.run()` asks the operand-matrix object for all three operands '
                'in step 1.1 (`scalesim/single_layer_sim.py:{}`). The filter one is '
                '**`get_filter_matrix()`**, defined on the `operand_matrix` class:'.format(line))
    line2, body2 = excerpt('scalesim/compute/operand_matrix.py', r'def get_filter_matrix\(', 0,
                           6)
    st.code(body2, language='python')
    st.caption('Each element of an operand matrix is a memory address, not a value: SCALE-Sim '
               'models data movement, not arithmetic. The compute system turns these into '
               'per-cycle demand matrices that the memory system services.')

    st.markdown('##### Q2. What is the default word size?')
    line, body = excerpt('scalesim/single_layer_sim.py', r'word_size = 1', 1, 2)
    st.code(body, language='python')
    st.markdown('**1 byte** (`scalesim/single_layer_sim.py:{}`). It is hard-coded rather than '
                'read from the config, so a "64 kB" SRAM holds 65,536 words (8-bit operands). '
                'The memory buffers divide byte capacity by it to get capacity in '
                'elements.'.format(line))

    st.markdown('##### Q3. How is DRAM bandwidth calculated?')
    line, body = excerpt('scalesim/single_layer_sim.py', r'# BW calc for DRAM access', 0, 6)
    st.code(body, language='python')
    st.latex(r'\text{Avg DRAM BW}_{op} = \frac{\text{DRAM accesses}_{op}}'
             r'{\text{stop cycle}_{op} - \text{start cycle}_{op} + 1}'
             r'\quad\left[\tfrac{\text{words}}{\text{cycle}}\right]')
    st.markdown('Each operand is averaged over **its own** DRAM activity window, not the whole '
                'run. Compare SRAM bandwidth, which divides by compute cycles:')
    st.code(excerpt('scalesim/single_layer_sim.py', r'self\.avg_ifmap_sram_bw = self', 0, 0)[1],
            language='python')

    p1 = st.session_state.get('p1')
    if p1 and 'error' not in p1['res']:
        st.markdown('**Check it yourself** against your Part I run ({}):'.format(p1['label']))
        d, b = p1['res']['detail'], p1['res']['bandwidth']
        rows = []
        for op, rd in (('IFMAP', 'Reads'), ('Filter', 'Reads'), ('OFMAP', 'Writes')):
            start = d['DRAM {} Start Cycle'.format(op)].iloc[0]
            stop = d['DRAM {} Stop Cycle'.format(op)].iloc[0]
            acc = d['DRAM {} {}'.format(op, rd)].iloc[0]
            rows.append({'Operand': op, 'DRAM accesses': acc, 'Start': start, 'Stop': stop,
                         'Accesses / (stop - start + 1)': acc / (stop - start + 1),
                         'BANDWIDTH_REPORT.csv':
                             b['Avg {} DRAM BW'.format(op.upper())].iloc[0]})
        st.dataframe(pd.DataFrame(rows).style.format(
            {'DRAM accesses': '{:,.0f}', 'Start': '{:,.0f}', 'Stop': '{:,.0f}',
             'Accesses / (stop - start + 1)': '{:.4f}', 'BANDWIDTH_REPORT.csv': '{:.4f}'}),
            hide_index=True)
        st.caption('Layer 0, from DETAILED_ACCESS_REPORT.csv. The last two columns match.')
    else:
        st.caption('Run a simulation in Part I to check this formula against real report '
                   'numbers.')


# --------------------------------------------------------------------------------------
# Exercise I - compare two configs
# --------------------------------------------------------------------------------------

def config_diff(cfg_a, cfg_b, name_a, name_b, only_diff):
    rows = []
    for section in list(dict.fromkeys(list(cfg_a) + list(cfg_b))):
        keys = list(dict.fromkeys(list(cfg_a.get(section, {})) + list(cfg_b.get(section, {}))))
        for key in keys:
            va = cfg_a.get(section, {}).get(key, '-')
            vb = cfg_b.get(section, {}).get(key, '-')
            if only_diff and va.strip() == vb.strip():
                continue
            rows.append({'Section': section, 'Knob': key, name_a: va, name_b: vb,
                         'What it does': KNOB_HELP.get((section, key), '')})
    return pd.DataFrame(rows)


def explain_comparison(name_a, cfg_a, sa, name_b, cfg_b, sb):
    """Turn the two runs into a short, data-backed explanation."""
    aa, ab = cfg_a['architecture_presets'], cfg_b['architecture_presets']
    pe_a = int(aa['ArrayHeight']) * int(aa['ArrayWidth'])
    pe_b = int(ab['ArrayHeight']) * int(ab['ArrayWidth'])
    out = []
    big, small = (name_b, name_a) if pe_b >= pe_a else (name_a, name_b)
    ratio = max(pe_a, pe_b) / min(pe_a, pe_b)
    out.append('**Raw compute.** {} has {:,} PEs vs {:,} - **{:.0f}x** more MAC units in '
               '{}.'.format(name_a, pe_a, pe_b, ratio, big) if ratio > 1 else
               '**Raw compute.** Both arrays have {:,} PEs.'.format(pe_a))
    cc = sa['Compute Cycles'] / sb['Compute Cycles']
    out.append('**Compute cycles** differ by **{:.1f}x** ({} vs {}). If utilization were '
               'equal this would track the PE ratio; the shortfall is lost to unmapped PEs '
               'and to filling/draining a bigger array on every fold.'.format(cc if cc >= 1 else 1 / cc,
                                 fmt(sa['Compute Cycles'], 'Cycles'),
                                 fmt(sb['Compute Cycles'], 'Cycles')))
    out.append('**Mapping.** Mapping efficiency is {:.1f}% vs {:.1f}%. A bigger array only '
               'helps if the layer has enough rows and columns of work to fill it - see the '
               'fold table below.'.format(sa['Mapping Efficiency %'],
                                           sb['Mapping Efficiency %']))
    if aa['Dataflow'] != ab['Dataflow']:
        out.append('**Dataflow.** {} uses {} and {} uses {}, so the same layer is laid out on '
                   'rows/columns differently.'.format(
                       name_a, DATAFLOW_NAMES[aa['Dataflow']], name_b,
                       DATAFLOW_NAMES[ab['Dataflow']]))
    ma, mb = cfg_a['run_presets']['InterfaceBandwidth'], cfg_b['run_presets']['InterfaceBandwidth']
    if ma != mb:
        user = name_a if ma == 'USER' else name_b
        out.append('**Bandwidth mode.** {} runs in USER mode, so its DRAM bandwidth is capped by '
                   'the config (that is why its DRAM BW numbers sit at the configured value) and '
                   'the initial SRAM fill is slow. The CALC config instead reports the bandwidth '
                   'it *would need* to never stall. Comparing DRAM BW across the two modes is '
                   'comparing a limit with a requirement.'.format(user))
    gap_a = sa['Total Cycles'] - sa['Compute Cycles']
    gap_b = sb['Total Cycles'] - sb['Compute Cycles']
    out.append('**Total vs compute cycles.** Prefetch/drain overhead is {} cycles for {} and {} '
               'for {}. Large SRAMs filled over a narrow DRAM link make that overhead '
               'dominate.'.format(fmt(gap_a, 'Cycles'), name_a, fmt(gap_b, 'Cycles'), name_b))
    return out


def bar_grid(df, metrics, colors, ncols=3):
    nrows = math.ceil(len(metrics) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=metrics,
                        vertical_spacing=0.18, horizontal_spacing=0.08)
    for i, metric in enumerate(metrics):
        r, c = i // ncols + 1, i % ncols + 1
        for j, label in enumerate(df.index):
            v = df.loc[label, metric]
            fig.add_trace(go.Bar(x=[label], y=[v], name=label, marker_color=colors[j],
                                 showlegend=(i == 0), legendgroup=label,
                                 text=[fmt(v, metric)], textposition='outside',
                                 cliponaxis=False,
                                 hovertemplate='%{x}<br>' + metric + ': %{text}<extra></extra>'),
                          row=r, col=c)
        fig.update_yaxes(title_text=UNITS[metric], row=r, col=c,
                         range=[0, df[metric].max() * 1.2 if df[metric].max() > 0 else 1])
    fig.update_traces(marker_line_width=0, width=0.5)
    return style_fig(fig, 320 * nrows)


with tab_ex1:
    st.header('Exercise I: Exploring simulator knobs')
    st.markdown('Compare two hardware configs on the same workload. The handout asks for '
                '`google.cfg` vs `scale.cfg`.')

    c1, c2 = st.columns(2)
    name_a = c1.selectbox('Config A', configs, index=configs.index('scale.cfg')
                          if 'scale.cfg' in configs else 0, key='ex1_a')
    name_b = c2.selectbox('Config B', configs, index=configs.index('google.cfg')
                          if 'google.cfg' in configs else 0, key='ex1_b')
    cfg_a = runner.read_cfg(os.path.join(CONFIG_DIR, name_a))
    cfg_b = runner.read_cfg(os.path.join(CONFIG_DIR, name_b))

    st.subheader('1. What is different between the two files?')
    only_diff = st.toggle('Show only knobs that differ', value=True, key='ex1_diff')
    st.dataframe(config_diff(cfg_a, cfg_b, name_a, name_b, only_diff), hide_index=True,
                 column_config={'What it does': st.column_config.TextColumn(width='large')})

    st.subheader('2. How do the outputs compare?')
    if st.button('Run both configs', type='primary', icon=':material/play_arrow:',
                 key='ex1_run', disabled=(name_a == name_b)):
        res = run_jobs([{'label': name_a, 'cfg': cfg_a, 'topology': topo_path},
                        {'label': name_b, 'cfg': cfg_b, 'topology': topo_path}], workers)
        st.session_state['ex1'] = {'res': res, 'a': name_a, 'b': name_b, 'topo': topo_name,
                                   'cfg_a': cfg_a, 'cfg_b': cfg_b}
    if name_a == name_b:
        st.caption('Pick two different configs.')

    ex1 = st.session_state.get('ex1')
    if ex1 and all('summary' in r for r in ex1['res'].values()):
        df = summary_frame(ex1['res'])
        metrics = st.multiselect('Metrics to plot', ALL_METRICS, default=CORE_METRICS,
                                 key='ex1_metrics',
                                 help='Defaults are the six metrics the handout asks for.')
        if metrics:
            st.caption('{} vs {} on {}. Hover any bar for the exact value; the camera icon '
                       'saves a PNG.'.format(ex1['a'], ex1['b'], ex1['topo']))
            show_fig(bar_grid(df[metrics], metrics, SERIES), 'ex1_bars')

        ratio = pd.DataFrame({ex1['a']: df.loc[ex1['a'], ALL_METRICS],
                              ex1['b']: df.loc[ex1['b'], ALL_METRICS]})
        ratio['B / A'] = ratio[ex1['b']] / ratio[ex1['a']].replace(0, float('nan'))
        with st.expander('All metrics as a table', icon=':material/table:'):
            st.dataframe(ratio.style.format('{:,.3f}', na_rep='-'))
            st.download_button('Download CSV', ratio.to_csv().encode(),
                               'exercise1_{}_vs_{}.csv'.format(ex1['a'], ex1['b']), 'text/csv')

        st.subheader('3. Why are they different?')
        for line in explain_comparison(ex1['a'], ex1['cfg_a'], df.loc[ex1['a']],
                                       ex1['b'], ex1['cfg_b'], df.loc[ex1['b']]):
            st.markdown('- ' + line)

        st.markdown('**Fold table** (how layer {} maps onto each array):'.format(
            topo_df['Layer name'].iloc[0]))
        fold_rows = []
        for name, cfg in ((ex1['a'], ex1['cfg_a']), (ex1['b'], ex1['cfg_b'])):
            a = cfg['architecture_presets']
            mp = analytic_mapping(topo_df, int(a['ArrayHeight']), int(a['ArrayWidth']),
                                  a['Dataflow']).iloc[0].to_dict()
            mp = dict({'Config': name, 'Array': '{}x{}'.format(a['ArrayHeight'],
                                                               a['ArrayWidth']),
                       'Dataflow': a['Dataflow']}, **mp)
            mp['Simulated mapping eff %'] = ex1['res'][name]['compute']['Mapping Efficiency %'
                                                                        ].iloc[0]
            fold_rows.append(mp)
        st.dataframe(pd.DataFrame(fold_rows).drop(columns=['Layer']).style.format(
            {'Mapping eff %': '{:.2f}', 'Simulated mapping eff %': '{:.2f}'}), hide_index=True)
        st.caption('S_r work items go on array rows and S_c on columns; anything that does not '
                   'fit becomes another fold. Unfilled rows/columns in the last fold are what '
                   'drag mapping efficiency down. Note SCALE-Sim rounds OFMAP size up, '
                   'ceil((H - R + stride) / stride), so AlexNet Conv1 has 55x55 = 3,025 output pixels, '
                   'not the textbook 54x54.')


# --------------------------------------------------------------------------------------
# Exercise II - design-space sweep
# --------------------------------------------------------------------------------------

SWEEP_PARAMS = {
    'ArraySize': {'label': 'Array size (N x N)', 'default': [16, 32, 64],
                  'fmt': lambda v: '{0}x{0}'.format(v)},
    'FilterSRAMBankBandwidth': {'label': 'FilterSRAMBankBandwidth (words/cycle)',
                                'default': [10, 50, 100], 'fmt': str},
    'IfmapSramSzkB': {'label': 'IfmapSramSzkB (kB)', 'default': [16, 64, 256], 'fmt': str},
}

PARAM_NOTES = {
    ('ArraySize', 'CALC'): 'More PEs means fewer folds and fewer compute cycles, but mapping '
                           'efficiency falls once the array is larger than the layer\'s '
                           'rows/columns of work, and the SRAM/DRAM bandwidth needed to keep the '
                           'array fed rises.',
    ('ArraySize', 'USER'): 'With DRAM capped, a bigger array mostly converts its speedup into '
                           'stall cycles: it can only compute as fast as data arrives.',
    ('FilterSRAMBankBandwidth', 'CALC'): 'This knob is only read in USER mode '
                                         '(`single_layer_sim.py`, the `use_user_dram_bandwidth()` '
                                         'branch). In CALC mode the simulator hard-codes the '
                                         'fill bandwidth, so every metric is identical.',
    ('FilterSRAMBankBandwidth', 'USER'): 'In USER mode this is the DRAM->filter SRAM fill rate. '
                                         'For a weight-stationary conv layer the filters are '
                                         'small and reused heavily, so only Filter DRAM BW moves; '
                                         'the bottleneck is IFMAP traffic.',
    ('IfmapSramSzkB', 'CALC'): 'Compute is unaffected; a larger IFMAP buffer changes how much '
                               'data is prefetched up front (Total Cycles) and how much DRAM '
                               'bandwidth is needed to refill it.',
    ('IfmapSramSzkB', 'USER'): 'With capped DRAM, the IFMAP buffer size sets how often the array '
                               'waits on refills, so stalls and total cycles move a lot (and not '
                               'always monotonically - refill granularity matters).',
}


def sweep_jobs(base_cfg, values, design, bw_modes):
    """Build the job list for a one-at-a-time or full-factorial sweep."""
    jobs = []
    base_vals = {'ArraySize': int(base_cfg['architecture_presets']['ArrayHeight']),
                 'FilterSRAMBankBandwidth': int(base_cfg['layout']['FilterSRAMBankBandwidth']),
                 'IfmapSramSzkB': int(base_cfg['architecture_presets']['IfmapSramSzkB'])}
    if design == 'oat':
        points = []
        for p, vals in values.items():
            for v in vals:
                point = dict(base_vals)
                point[p] = v
                points.append((p, point))
    else:
        points = [('all', dict(zip(values, combo)))
                  for combo in itertools.product(*values.values())]
    for bw in bw_modes:
        for swept, point in points:
            cfg = copy.deepcopy(base_cfg)
            cfg['run_presets']['InterfaceBandwidth'] = bw
            for k, v in point.items():
                set_knob(cfg, k, v)
            label = '{} | {}'.format(bw, ', '.join('{}={}'.format(k, v)
                                                   for k, v in point.items()))
            jobs.append({'label': label, 'cfg': cfg, 'topology': topo_path,
                         'meta': dict(point, mode=bw, swept=swept)})
    return jobs


def oat_figure(df, metrics, values, bw_modes, base_vals):
    params = list(values)
    fig = make_subplots(rows=len(metrics), cols=len(params), shared_xaxes=False,
                        column_titles=[SWEEP_PARAMS[p]['label'] for p in params],
                        row_titles=None, vertical_spacing=0.035, horizontal_spacing=0.07)
    for ci, p in enumerate(params):
        sub = df[df['swept'] == p]
        for ri, metric in enumerate(metrics):
            for mi, mode in enumerate(bw_modes):
                s = sub[sub['mode'] == mode].sort_values(p)
                xs = [SWEEP_PARAMS[p]['fmt'](v) + (' (base)' if v == base_vals[p] else '')
                      for v in s[p]]
                fig.add_trace(go.Scatter(
                    x=xs, y=s[metric], mode='lines+markers', name=mode + ' mode',
                    legendgroup=mode, showlegend=(ri == 0 and ci == 0),
                    line=dict(color=SERIES[mi], width=2),
                    marker=dict(size=9, line=dict(width=2, color='rgba(255,255,255,0.9)')),
                    customdata=[fmt(v, metric) for v in s[metric]],
                    hovertemplate=mode + ' | %{x}<br>' + metric + ': %{customdata}'
                                  '<extra></extra>'),
                    row=ri + 1, col=ci + 1)
            fig.update_yaxes(title_text=metric if ci == 0 else None, row=ri + 1, col=ci + 1,
                             rangemode='tozero', title_font=dict(size=11))
            fig.update_xaxes(type='category', row=ri + 1, col=ci + 1)
    return style_fig(fig, 210 * len(metrics) + 80)


def factorial_figure(df, metric, bw_mode, values):
    sizes = values['ArraySize']
    fbws = values['FilterSRAMBankBandwidth']
    fig = make_subplots(rows=1, cols=len(fbws), shared_yaxes=True,
                        subplot_titles=['FilterSRAMBankBandwidth = {}'.format(v) for v in fbws])
    sub = df[df['mode'] == bw_mode]
    for ci, fbw in enumerate(fbws):
        for si, size in enumerate(sizes):
            s = sub[(sub['FilterSRAMBankBandwidth'] == fbw)
                    & (sub['ArraySize'] == size)].sort_values('IfmapSramSzkB')
            fig.add_trace(go.Scatter(
                x=[str(v) for v in s['IfmapSramSzkB']], y=s[metric], mode='lines+markers',
                name='{0}x{0} array'.format(size), legendgroup=str(size),
                showlegend=(ci == 0), line=dict(color=SERIES[si % 3], width=2),
                marker=dict(size=9),
                customdata=[fmt(v, metric) for v in s[metric]],
                hovertemplate='IfmapSramSzkB %{x}<br>' + metric + ': %{customdata}'
                              '<extra></extra>'),
                row=1, col=ci + 1)
        fig.update_xaxes(title_text='IfmapSramSzkB', type='category', row=1, col=ci + 1)
    fig.update_yaxes(title_text=metric, row=1, col=1, rangemode='tozero')
    return style_fig(fig, 380)


def sensitivity(df, values, bw_modes, metrics):
    """For each (mode, param): which metrics moved by more than 0.5% across the sweep."""
    rows = []
    for mode in bw_modes:
        for p in values:
            s = df[(df['swept'] == p) & (df['mode'] == mode)]
            for m in metrics:
                lo, hi = s[m].min(), s[m].max()
                spread = (hi - lo) / abs(hi) if hi else 0.0
                rows.append({'mode': mode, 'param': p, 'metric': m, 'spread': spread})
    return pd.DataFrame(rows)


with tab_ex2:
    st.header('Exercise II: Understanding the design space')
    st.markdown('Start from `scale.cfg` + the selected workload and sweep three knobs, three '
                'values each.')

    base_name = st.selectbox('Base config', configs, index=configs.index('scale.cfg')
                             if 'scale.cfg' in configs else 0, key='ex2_base')
    base_cfg = runner.read_cfg(os.path.join(CONFIG_DIR, base_name))
    base_mode = base_cfg['run_presets']['InterfaceBandwidth']

    st.markdown('**Sweep values**')
    vcols = st.columns(3)
    values = {}
    for col, (p, spec) in zip(vcols, SWEEP_PARAMS.items()):
        with col:
            raw = st.text_input(spec['label'], ', '.join(str(v) for v in spec['default']),
                                key='ex2_vals_' + p, help='Comma-separated integers.')
            try:
                values[p] = sorted({int(x) for x in raw.split(',') if x.strip()})
            except ValueError:
                st.error('Use comma-separated integers.')
                values[p] = spec['default']

    o1, o2 = st.columns(2)
    with o1:
        design = st.radio(
            'Sweep design', ['oat', 'full'], horizontal=True, key='ex2_design',
            format_func=lambda d: ('One knob at a time ({} runs)'.format(
                sum(len(v) for v in values.values())) if d == 'oat' else
                'Full factorial ({} runs)'.format(
                    math.prod(len(v) for v in values.values()))),
            help='One-at-a-time varies each knob around the base config (the handout\'s '
                 '"9 configurations"). Full factorial runs every combination, which reveals '
                 'interactions between knobs.')
    with o2:
        other = 'USER' if base_mode == 'CALC' else 'CALC'
        also_other = st.toggle(
            'Also run in {} bandwidth mode'.format(other), value=True, key='ex2_both',
            help='{} uses InterfaceBandwidth = {}. Re-running every point in {} mode shows '
                 'which conclusions depend on that assumption - in particular, '
                 'FilterSRAMBankBandwidth does nothing in CALC mode.'.format(
                     base_name, base_mode, other))
    bw_modes = [base_mode, other] if also_other else [base_mode]

    jobs = sweep_jobs(base_cfg, values, design, bw_modes)
    bad = sorted({p for j in jobs for p in validate_cfg(j['cfg'])})
    for p in bad:
        st.warning(p + ' Pick multiples of 10 for FilterSRAMBankBandwidth.')

    st.caption('{} simulations total. Tiny arrays are the slowest points (more folds to step '
               'through).'.format(len(jobs)))
    if st.button('Run sweep', type='primary', icon=':material/play_arrow:', key='ex2_run',
                 disabled=bool(bad)):
        res = run_jobs(jobs, workers)
        rows = []
        for j in jobs:
            r = res[j['label']]
            if 'summary' in r:
                rows.append(dict(j['meta'], **r['summary']))
        base_vals = {'ArraySize': int(base_cfg['architecture_presets']['ArrayHeight']),
                     'FilterSRAMBankBandwidth':
                         int(base_cfg['layout']['FilterSRAMBankBandwidth']),
                     'IfmapSramSzkB': int(base_cfg['architecture_presets']['IfmapSramSzkB'])}
        st.session_state['ex2'] = {'df': pd.DataFrame(rows), 'values': values,
                                   'design': design, 'bw_modes': bw_modes,
                                   'base_vals': base_vals, 'base': base_name,
                                   'topo': topo_name}

    ex2 = st.session_state.get('ex2')
    if ex2 is not None and len(ex2['df']):
        df, vals, modes = ex2['df'], ex2['values'], ex2['bw_modes']
        st.divider()
        default_metrics = CORE_METRICS[:2] + ['Stall Cycles'] + CORE_METRICS[2:] + \
            ['Avg IFMAP DRAM BW', 'Avg FILTER DRAM BW']
        metrics = st.multiselect(
            'Metrics', ALL_METRICS, default=default_metrics, key='ex2_metrics',
            help='Defaults: the six Exercise I metrics, plus Stall Cycles and the IFMAP/FILTER '
                 'DRAM bandwidths, which are where the memory knobs show up.')

        if ex2['design'] == 'oat' and metrics:
            st.subheader('How each knob moves each metric')
            st.caption('Columns are knobs, rows are metrics. Each knob is varied while the '
                       'other two stay at the base value (marked "(base)").')
            show_fig(oat_figure(df, metrics, vals, modes, ex2['base_vals']), 'ex2_oat')

            st.subheader('What to take away')
            sens = sensitivity(df, vals, modes, metrics)
            for mode in modes:
                st.markdown('**{} mode**'.format(mode))
                for p in vals:
                    s = sens[(sens['mode'] == mode) & (sens['param'] == p)]
                    moved = s[s['spread'] > 0.005].sort_values('spread', ascending=False)
                    if moved.empty:
                        verdict = 'no metric changed.'
                    else:
                        verdict = 'moved ' + ', '.join(
                            '{} ({:.0f}%)'.format(r.metric, 100 * r.spread)
                            for r in moved.itertuples()) + '.'
                    st.markdown('- `{}`: {} {}'.format(p, verdict, PARAM_NOTES.get((p, mode),
                                                                                  '')))
                st.caption('Percent = (max - min) / max across the three values.')
        elif ex2['design'] == 'full' and metrics:
            st.subheader('Interactions between knobs')
            f1, f2 = st.columns(2)
            metric = f1.selectbox('Metric', metrics, key='ex2_fact_metric')
            mode = f2.radio('Bandwidth mode', modes, horizontal=True, key='ex2_fact_mode')
            show_fig(factorial_figure(df, metric, mode, vals), 'ex2_fact')
            st.caption('Lines that stay parallel mean the knobs act independently; lines that '
                       'cross or fan out mean they interact. Identical panels mean '
                       'FilterSRAMBankBandwidth has no effect in that mode.')

        with st.expander('All sweep results', icon=':material/table:'):
            st.dataframe(df, hide_index=True)
            st.download_button('Download CSV', df.to_csv(index=False).encode(),
                               'exercise2_sweep.csv', 'text/csv')
