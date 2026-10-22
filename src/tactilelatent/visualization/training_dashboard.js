/**
 * Compare saved training histories and live status samples in browser charts.
 * Inputs: training_dashboard.html controls and the read-only training API.
 * Outputs: filtered loss charts, run details, and optional CSV/PNG downloads.
 * Example: run `tl view --port 8080`, then open the Training monitor page.
 * Display smoothing changes plotted curves only; stored research metrics stay raw.
 */
'use strict';
const $ = id => document.getElementById(id);
const colors = ['#69d8c3', '#ffb76b', '#8eb7ff', '#dd9cf5', '#f28d9f', '#d5db78', '#8ed4f0', '#c5a585'];
const splits = {
  train: 'Training',
  validation: 'Validation',
  validation_original: 'Validation · original',
  validation_mirrored: 'Validation · mirrored'
};
const state = {
  catalog: null,
  runs: [],
  selected: new Set(),
  charts: [],
  zoom: new Map(),
  request: 0,
  busy: false,
  controller: null,
  initialized: false
};
const parameters = new URLSearchParams(location.search);
let basePath = location.pathname.replace(/\/(training|joint|markers)\/?$/, '').replace(/\/$/, '');
for (const link of document.querySelectorAll('nav a')) {
  const href = link.getAttribute('href');
  link.href = basePath + '/' + (href.startsWith('?') ? href : 'markers');
}

function api(path) {
  return basePath + '/api/training/' + path;
}

function node(tag, text, className) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (className) n.className = className;
  return n;
}

function sourceName(source) {
  return state.catalog?.source_names[source] || source;
}

function metricName(metric) {
  return state.catalog?.metrics[metric] || metric;
}

function shortName(run) {
  const parts = run.name.split(' / ');
  return parts.length > 1 ? parts.slice(-2).join(' / ') : run.name;
}

function number(value) {
  if (value === null || !Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  return Math.abs(value) < .001 || Math.abs(value) >= 10000 ? value.toExponential(3) : Number(value.toPrecision(5)).toString();
}

function save(blob, name) {
  const url = URL.createObjectURL(blob),
    a = node('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setConnection(text, error = false) {
  $('connection').textContent = text;
  $('connection').classList.toggle('error', error);
}
async function json(url, signal) {
  const response = await fetch(url, {
      signal,
      cache: 'no-store'
    }),
    data = await response.json();
  if (!response.ok) throw Error(data.error || response.statusText);
  return data;
}

function chosenMetric() {
  const available = new Set(state.runs.flatMap(r => r.metrics));
  return $('metric').value === 'auto' ? ['supervision_loss', 'loss', 'normalized_mae', 'native_mae', 'mae'].find(m => available.has(m)) || [...available][0] : $('metric').value;
}

function remember() {
  try {
    localStorage.setItem('tactile-training-models', JSON.stringify([...state.selected]));
  } catch {}
  const q = new URLSearchParams(location.search);
  q.set('page', 'training');
  q.set('runs', [...state.selected].join(','));
  if ($('datasetFilter').value !== 'all') q.set('dataset', $('datasetFilter').value);
  else q.delete('dataset');
  q.set('metric', $('metric').value);
  history.replaceState(null, '', location.pathname + '?' + q);
}

function selectedColor(id) {
  return colors[[...state.selected].indexOf(id) % colors.length];
}

function renderRunList() {
  const normalize = s => s.toLowerCase().replace(/[^a-z0-9]/g, ''),
    search = normalize($('runSearch').value),
    kind = $('kind').value;
  const runs = (state.catalog?.runs || []).filter(r => (kind === 'all' || r.kind === kind) && normalize([r.name, ...r.sources, ...r.sources.map(sourceName), `${r.latent_dim||''}D`].join(' ')).includes(search));
  $('runList').replaceChildren();
  $('selectionCount').textContent = `${state.selected.size} selected · ${runs.length} models`;
  for (const run of runs) {
    const label = node('label', undefined, 'run-option' + (state.selected.has(run.id) ? ' selected' : '')),
      box = node('input');
    box.type = 'checkbox';
    box.checked = state.selected.has(run.id);
    box.value = run.id;
    const text = node('div'),
      name = node('div', shortName(run), 'name'),
      meta = node('small', `${run.kind==='autoencoder'?'AE':run.kind==='teacher'?'Teacher':'Joint'} · ${run.latent_dim?run.latent_dim+'D · ':''}epoch ${run.completed_epochs}${run.total_epochs?'/'+run.total_epochs:''}${run.weights_available?'':' · history only'}`);
    const badge = node('span', run.state === 'inactive' ? 'Saved history' : run.state, 'badge ' + run.state);
    text.append(name, meta, badge);
    label.append(box, text);
    $('runList').append(label);
    box.onchange = () => {
      if (box.checked) {
        if (state.selected.size >= 8) {
          box.checked = false;
          setConnection('Select up to eight models at a time.', true);
          return;
        }
        state.selected.add(run.id);
      } else state.selected.delete(run.id);
      renderRunList();
      remember();
      loadDashboard(true);
    };
  }
  if (!runs.length) $('runList').append(node('p', 'No matching training histories.', 'note'));
}

function fillFilters() {
  const sources = [...new Set(state.runs.flatMap(r => r.sources))].sort((a, b) => sourceName(a).localeCompare(sourceName(b)));
  const old = $('datasetFilter').value,
    requested = !state.initialized ? parameters.get('dataset') : null;
  $('datasetFilter').replaceChildren(new Option('All datasets', 'all'), ...sources.map(s => new Option(sourceName(s), s)));
  $('datasetFilter').value = sources.includes(requested) ? requested : sources.includes(old) ? old : 'all';
  const metrics = [...new Set(state.runs.flatMap(r => r.metrics))],
    oldMetric = $('metric').value,
    requestedMetric = !state.initialized ? parameters.get('metric') : null;
  const ordered = Object.keys(state.catalog.metrics).filter(m => metrics.includes(m));
  $('metric').replaceChildren(new Option('Primary loss', 'auto'), ...ordered.map(m => new Option(metricName(m), m)));
  $('metric').value = ordered.includes(requestedMetric) ? requestedMetric : ordered.includes(oldMetric) ? oldMetric : 'auto';
  const counts = new Map();
  for (const run of state.runs)
    for (const sample of run.live_samples) counts.set(sample.source, (counts.get(sample.source) || 0) + 1);
  const mostSampled = [...counts].sort((a, b) => b[1] - a[1])[0]?.[0];
  const live = $('liveSource').value,
    preferred = $('datasetFilter').value !== 'all' ? $('datasetFilter').value : mostSampled || state.runs.find(r => r.state === 'running')?.status.source || 'egotouch';
  $('liveSource').replaceChildren(...sources.map(s => new Option(sourceName(s), s)));
  $('liveSource').value = sources.includes(live) ? live : sources.includes(preferred) ? preferred : sources[0] || '';
  state.initialized = true;
}

/** Smooth display values without mutating the server's original metric points. */
function smooth(points) {
  const weight = Number($('smoothing').value);
  let previous = null;
  return points.map(([x, y]) => {
    previous = previous === null ? y : weight * previous + (1 - weight) * y;
    return [x, previous, y];
  });
}

/** Render scalar histories, with hover values and independent horizontal zoom. */
class ScalarChart {
  constructor(parent, series, key, {
    live = false
  } = {}) {
    this.series = series;
    this.key = key;
    this.live = live;
    this.hover = null;
    this.drag = null;
    this.shell = node('div', undefined, 'chart-shell');
    this.canvas = node('canvas');
    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label', key + ' scalar curves');
    this.tip = node('div', undefined, 'tooltip');
    this.tip.hidden = true;
    this.shell.append(this.canvas, this.tip);
    parent.append(this.shell);
    this.canvas.addEventListener('pointermove', e => this.pointer(e));
    this.canvas.addEventListener('pointerleave', () => {
      if (this.drag === null) {
        this.hover = null;
        this.tip.hidden = true;
        this.draw();
      }
    });
    this.canvas.addEventListener('pointerdown', e => {
      if (e.button !== 0 || !this.bounds) return;
      this.drag = this.local(e).x;
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener('pointerup', e => {
      if (this.drag === null) return;
      const end = this.local(e).x,
        start = this.drag;
      this.drag = null;
      if (Math.abs(end - start) > 8 && this.bounds) {
        const a = this.toValue(Math.min(start, end)),
          b = this.toValue(Math.max(start, end));
        state.zoom.set(this.key, [a, b]);
        this.hover = null;
        this.tip.hidden = true;
      }
      this.draw();
    });
    this.canvas.addEventListener('pointercancel', () => {
      this.drag = null;
      this.draw();
    });
    this.canvas.ondblclick = () => {
      state.zoom.delete(this.key);
      this.draw();
    };
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(this.shell);
    this.draw();
  }
  destroy() {
    this.observer.disconnect();
  }
  local(e) {
    const r = this.canvas.getBoundingClientRect();
    return {
      x: e.clientX - r.left,
      y: e.clientY - r.top
    };
  }
  toValue(pixel) {
    const b = this.bounds;
    return b.xmin + Math.max(0, Math.min(1, (pixel - b.left) / b.width)) * (b.xmax - b.xmin);
  }
  pointer(e) {
    if (!this.bounds) return;
    const p = this.local(e);
    this.hover = p;
    if (this.drag !== null) {
      this.draw();
      return;
    }
    this.draw();
    if (p.x < this.bounds.left || p.x > this.bounds.left + this.bounds.width) {
      this.tip.hidden = true;
      return;
    }
    const x = this.toValue(p.x),
      lines = [];
    let closest = null;
    for (const s of this.series) {
      if (!s.points.length) continue;
      const point = s.points.reduce((a, b) => Math.abs(b[0] - x) < Math.abs(a[0] - x) ? b : a);
      if (closest === null || Math.abs(point[0] - x) < Math.abs(closest - x)) closest = point[0];
      lines.push(`${s.name}: ${number(point[1])}${Math.abs(point[0]-x)>(this.bounds.xmax-this.bounds.xmin)/8?' (nearest saved point)':''}`);
    }
    if (closest === null) {
      this.tip.hidden = true;
      return;
    }
    this.tip.textContent = (this.live ? new Date(closest * 1000).toLocaleTimeString() : `Epoch ${number(closest)}`) + '\n' + lines.join('\n');
    this.tip.hidden = false;
    this.tip.style.left = Math.max(0, Math.min(p.x + 12, this.shell.clientWidth - this.tip.offsetWidth - 8)) + 'px';
    this.tip.style.top = Math.max(0, Math.min(p.y - 45, this.shell.clientHeight - this.tip.offsetHeight)) + 'px';
  }
  draw() {
    const canvas = this.canvas,
      w = this.shell.clientWidth,
      h = this.shell.clientHeight;
    if (w < 20 || h < 20) return;
    const ratio = devicePixelRatio || 1;
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
    const c = canvas.getContext('2d');
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
    c.fillStyle = '#19212d';
    c.fillRect(0, 0, w, h);
    c.fillStyle = '#9eb3ca';
    c.font = '11px system-ui';
    const logarithmic = $('logScale').checked;
    const all = this.series.flatMap(s => s.points).filter(p => Number.isFinite(p[1]) && (!logarithmic || p[1] > 0));
    if (!all.length) {
      c.textAlign = 'center';
      c.fillText(this.live ? 'Waiting for live samples for this dataset…' : 'No values recorded for this metric / selection.', w / 2, h / 2);
      this.bounds = null;
      return;
    }
    const zoom = state.zoom.get(this.key);
    let xmin = zoom?.[0] ?? Math.min(...all.map(p => p[0])),
      xmax = zoom?.[1] ?? Math.max(...all.map(p => p[0]));
    if (xmax === xmin) {
      xmin -= this.live ? 5 : .5;
      xmax += this.live ? 5 : .5;
    }
    const visible = all.filter(p => p[0] >= xmin && p[0] <= xmax),
      ys = (visible.length ? visible : all).map(p => logarithmic ? Math.log10(p[1]) : p[1]);
    let ymin = Math.min(...ys),
      ymax = Math.max(...ys);
    const pad = (ymax - ymin) * .1 || Math.max(Math.abs(ymax) * .08, .01);
    ymin -= pad;
    ymax += pad;
    if (!logarithmic && ys.every(v => v >= 0)) ymin = Math.max(0, ymin);
    const left = 70,
      right = 14,
      top = 14,
      bottom = 36,
      width = w - left - right,
      height = h - top - bottom;
    this.bounds = {
      xmin,
      xmax,
      left,
      width
    };
    const px = x => left + (x - xmin) / (xmax - xmin) * width,
      py = y => top + (ymax - (logarithmic ? Math.log10(y) : y)) / (ymax - ymin) * height;
    c.strokeStyle = '#2c3a4b';
    c.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = top + i / 4 * height;
      c.beginPath();
      c.moveTo(left, y);
      c.lineTo(w - right, y);
      c.stroke();
      c.textAlign = 'right';
      const value = ymax - i / 4 * (ymax - ymin);
      c.fillText(number(logarithmic ? 10 ** value : value), left - 9, y + 4);
    }
    for (let i = 0; i <= 4; i++) {
      const x = left + i / 4 * width,
        value = xmin + i / 4 * (xmax - xmin);
      c.textAlign = 'center';
      c.fillText(this.live ? new Date(value * 1000).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      }) : number(value), x, h - 15);
    }
    c.save();
    c.beginPath();
    c.rect(left, top, width, height);
    c.clip();
    for (const s of this.series) {
      const points = smooth(s.points);
      c.strokeStyle = s.color;
      c.fillStyle = s.color;
      c.lineWidth = 1.8;
      c.setLineDash(s.split === 'train' || this.live ? [] : s.split === 'validation_mirrored' ? [2, 5] : [7, 4]);
      c.beginPath();
      let drawing = false;
      for (const [x, y] of points) {
        if (logarithmic && y <= 0) {
          drawing = false;
          continue;
        }
        const xx = px(x),
          yy = py(y);
        if (drawing) c.lineTo(xx, yy);
        else {
          c.moveTo(xx, yy);
          drawing = true;
        }
      }
      c.stroke();
      c.setLineDash([]);
      if (points.length < 70)
        for (const [x, y] of points) {
          if (logarithmic && y <= 0) continue;
          c.beginPath();
          c.arc(px(x), py(y), 2.3, 0, 2 * Math.PI);
          c.fill();
        }
    }
    if (this.hover) {
      c.strokeStyle = '#a1bacb66';
      c.setLineDash([3, 4]);
      c.beginPath();
      c.moveTo(this.hover.x, top);
      c.lineTo(this.hover.x, top + height);
      c.stroke();
      c.setLineDash([]);
    }
    if (this.drag !== null && this.hover) {
      c.fillStyle = '#56cbb622';
      c.fillRect(Math.min(this.drag, this.hover.x), top, Math.abs(this.hover.x - this.drag), height);
    }
    c.restore();
  }
  png(name) {
    saveDataUrl(this.canvas.toDataURL('image/png'), name);
  }
}

function saveDataUrl(url, name) {
  const a = node('a');
  a.href = url;
  a.download = name;
  a.click();
}

/** Select matching raw series; each source retains its own chart scale. */
function makeSeries(source, metric) {
  return state.runs.flatMap(run => run.curves.filter(c => c.source === source && c.metric === metric).map(c => ({
    ...c,
    color: selectedColor(run.id),
    name: shortName(run) + ' · ' + (splits[c.split] || c.split),
    runId: run.id,
    runName: run.name
  })));
}

function latestLines(parent, series) {
  const area = node('div', undefined, 'latest');
  for (const s of series) {
    const last = s.points.at(-1);
    if (!last) continue;
    const line = node('div', `${shortName(state.runs.find(r=>r.id===s.runId))} · ${splits[s.split]||s.split}: ${number(last[1])} · epoch ${last[0]}`);
    line.style.color = s.color;
    area.append(line);
  }
  parent.append(area);
}

function renderStatus() {
  const sources = new Set(state.runs.flatMap(r => r.sources));
  $('summary').replaceChildren();
  for (const [value, label] of [
      [state.runs.length, 'models selected'],
      [sources.size, 'datasets'],
      [state.runs.reduce((n, r) => n + r.record_count, 0), 'saved epoch records']
    ]) {
    const tile = node('div', undefined, 'stat');
    tile.append(node('strong', String(value)), document.createTextNode(label));
    $('summary').append(tile);
  }
  $('runStatus').replaceChildren();
  $('modelLegend').replaceChildren();
  $('selectedDetails').replaceChildren();
  for (const run of state.runs) {
    const status = run.status,
      active = run.state === 'running' || run.state === 'updating';
    const row = node('div', undefined, 'run-status'),
      title = node('div');
    title.style.color = selectedColor(run.id);
    title.textContent = `${shortName(run)} · ${run.state} · ${run.completed_epochs} completed epochs${run.total_epochs?' / '+run.total_epochs:''}`;
    row.append(title);
    if (active) {
      const progress = Number(status.batch) / Number(status.batches_per_epoch);
      row.append(node('div', `Epoch ${status.epoch??'—'}${Number.isFinite(progress)?' · batch '+status.batch.toLocaleString()+' / '+status.batches_per_epoch.toLocaleString()+' ('+(100*progress).toFixed(1)+'%)':''}${status.source?' · '+sourceName(status.source):''}${Number.isFinite(status.loss)?' · latest batch loss '+number(status.loss):''}`));
      if (Number.isFinite(progress)) {
        const bar = node('div', undefined, 'progress'),
          fill = node('span');
        fill.style.width = Math.max(0, Math.min(100, 100 * progress)) + '%';
        bar.append(fill);
        row.append(bar);
      }
    }
    $('runStatus').append(row);
    const legend = node('div', undefined, 'legend-item'),
      swatch = node('span', undefined, 'swatch');
    swatch.style.background = selectedColor(run.id);
    legend.append(swatch, document.createTextNode(shortName(run)));
    $('modelLegend').append(legend);
    $('selectedDetails').append(node('div', `${run.name} · ${run.latent_dim?run.latent_dim+'D · ':''}${run.latent_mode||run.kind}${run.seed!=null?' · seed '+run.seed:''} · ${run.path}${run.epoch_offset?' · zero-indexed logs displayed from epoch 1':''}`));
  }
}

/** Build per-source charts and distinguish epoch metrics from live samples. */
function renderCharts() {
  for (const chart of state.charts) chart.destroy();
  state.charts = [];
  $('datasetCharts').replaceChildren();
  $('overviewChart').replaceChildren();
  $('liveChart').replaceChildren();
  const metric = chosenMetric(),
    filter = $('datasetFilter').value;
  $('smoothingValue').textContent = Number($('smoothing').value).toFixed(2) + (Number($('smoothing').value) === 0 ? ' · raw' : ' · EMA');
  $('metricHelp').textContent = `Showing ${metricName(metric)||'saved scalars'}. Historical runs that recorded only validation MAE expose it under that metric; no validation loss is inferred.${$('logScale').checked?' Log scale omits zero and negative values.':''}${state.runs.some(r=>!r.metrics.includes(metric))?' Some selected models did not record this metric.':''}`;
  let sources = [...new Set(state.runs.flatMap(r => r.sources))].sort((a, b) => sourceName(a).localeCompare(sourceName(b)));
  if (filter !== 'all') sources = sources.filter(s => s === filter);
  for (const source of sources) {
    const card = node('section', undefined, 'panel chart-card');
    card.dataset.source = source;
    const heading = node('div', undefined, 'chart-heading'),
      button = node('button', 'Save PNG');
    heading.append(node('h2', sourceName(source)), button);
    card.append(heading, node('p', metricName(metric), 'metric-name'));
    $('datasetCharts').append(card);
    const series = makeSeries(source, metric),
      chart = new ScalarChart(card, series, 'source:' + source + ':' + metric);
    state.charts.push(chart);
    button.onclick = () => chart.png(source + '_' + metric + '.png');
    latestLines(card, series);
    let fallbackShown = false;
    if ($('metric').value === 'auto') {
      const alternatives = new Map();
      for (const run of state.runs) {
        if (series.some(s => s.runId === run.id && s.split.startsWith('validation'))) continue;
        const available = run.curves.filter(c => c.source === source && c.split.startsWith('validation'));
        const fallback = ['normalized_mae', 'native_mae', 'marker_mae_px', 'mae', 'selection_score'].find(m => available.some(c => c.metric === m));
        if (fallback) {
          if (!alternatives.has(fallback)) alternatives.set(fallback, []);
          alternatives.get(fallback).push(...makeSeries(source, fallback).filter(s => s.runId === run.id && s.split.startsWith('validation')));
        }
      }
      for (const [fallback, validation] of alternatives) {
        card.append(node('p', 'Validation · ' + metricName(fallback) + ' · separate scale', 'note'));
        const second = new ScalarChart(card, validation, 'source:' + source + ':fallback:' + fallback);
        state.charts.push(second);
        latestLines(card, validation);
        fallbackShown = true;
      }
    }
    if (series.some(s => s.split === 'train') && !series.some(s => s.split.startsWith('validation'))) {
      const hint = node('p', 'Validation objective loss was not logged.' + (fallbackShown ? ' The separate chart shows the recorded validation metric.' : ' Select an available MAE metric to inspect validation.'), 'hint');
      card.append(hint);
    }
  }
  if (!sources.length) $('datasetCharts').append(node('div', state.runs.length ? 'This run logs overall metrics rather than separate dataset curves.' : 'Select a model to view its saved curves.', 'panel empty'));
  let overview = filter !== 'all' ? makeSeries(filter, metric) : makeSeries('__macro__', metric);
  if (filter === 'all')
    for (const run of state.runs)
      if (!overview.some(s => s.runId === run.id)) overview.push(...makeSeries('__all__', metric).filter(s => s.runId === run.id));
  if (!overview.length && sources.length === 1) overview = makeSeries(sources[0], metric);
  $('overviewTitle').textContent = filter !== 'all' ? sourceName(filter) : overview.some(s => s.source === '__macro__') ? 'Source mean' : 'Overall / selected source';
  $('overviewNote').textContent = filter !== 'all' ? 'Saved curves for the selected dataset.' : overview.some(s => s.source === '__macro__') ? 'Equal-weight means over sources that logged this metric. Any logged overall curves retain their original weighting.' : 'These are the logged overall or single-dataset measurements; no source averages are inferred.';
  const overviewChart = new ScalarChart($('overviewChart'), overview, 'overview:' + metric);
  state.charts.push(overviewChart);
  $('overviewPng').onclick = () => overviewChart.png('source_mean_' + metric + '.png');
  const liveSource = $('liveSource').value,
    liveSeries = state.runs.map(run => ({
      name: shortName(run),
      color: selectedColor(run.id),
      split: 'train',
      points: run.live_samples.filter(r => r.source === liveSource && Number.isFinite(r.loss) && Number.isFinite(r.time)).map(r => [r.time, r.loss])
    }));
  const live = new ScalarChart($('liveChart'), liveSeries, 'live:' + liveSource, {
    live: true
  });
  state.charts.push(live);
  const count = liveSeries.reduce((n, s) => n + s.points.length, 0);
  $('liveNote').textContent = count ? `${count.toLocaleString()} recorded status samples for ${sourceName(liveSource)}. Sampling continues in the background while this viewer service runs.` : 'Collecting new samples. Old batch losses cannot be recovered from epoch-only logs.';
}

function render() {
  renderStatus();
  renderCharts();
}
/** Refresh read-only histories, cancelling superseded network requests. */
async function loadDashboard(force = false) {
  if (state.busy && !force) return;
  if (force) state.controller?.abort();
  const mine = ++state.request;
  state.busy = true;
  state.controller = new AbortController();
  const signal = state.controller.signal;
  try {
    const catalog = await json(api('catalog' + (force ? '?refresh=1' : '')), signal);
    if (mine !== state.request) return;
    state.catalog = catalog;
    if (!state.initialized && !state.selected.size) {
      let ids = parameters.get('runs')?.split(',');
      if (!ids?.length) try {
        ids = JSON.parse(localStorage.getItem('tactile-training-models') || '[]');
      } catch {}
      for (const id of ids || [])
        if (catalog.runs.some(r => r.id === id) && state.selected.size < 8) state.selected.add(id);
      if (!state.selected.size && catalog.runs.length) state.selected.add((catalog.runs.find(r => r.state === 'running') || catalog.runs[0]).id);
    }
    for (const id of [...state.selected])
      if (!catalog.runs.some(r => r.id === id)) state.selected.delete(id);
    renderRunList();
    const data = state.selected.size ? await json(api('history?runs=' + encodeURIComponent([...state.selected].join(','))), signal) : {
      runs: []
    };
    if (mine !== state.request) return;
    state.runs = [...state.selected].map(id => data.runs.find(r => r.id === id)).filter(Boolean);
    fillFilters();
    render();
    const time = new Date(catalog.server_time * 1000).toLocaleTimeString();
    setConnection(`Updated ${time} · ${catalog.runs.length} training histories${$('autoRefresh').checked?' · refreshes every 5 seconds':' · automatic refresh paused'}${data.collector_error?' · live collector: '+data.collector_error:''}`, !!data.collector_error);
  } catch (error) {
    if (error.name !== 'AbortError') setConnection('Could not refresh: ' + error.message + '. Existing curves remain visible.', true);
  } finally {
    if (mine === state.request) state.busy = false;
  }
}

$('runSearch').oninput = renderRunList;
$('kind').onchange = renderRunList;
$('clearSelection').onclick = () => {
  state.selected.clear();
  state.initialized = true;
  remember();
  loadDashboard(true);
};
for (const id of ['datasetFilter', 'metric']) $(id).onchange = () => {
  if (id === 'datasetFilter' && $('datasetFilter').value !== 'all') $('liveSource').value = $('datasetFilter').value;
  remember();
  renderCharts();
};
for (const id of ['smoothing', 'logScale']) $(id).oninput = renderCharts;
$('liveSource').onchange = renderCharts;
$('refresh').onclick = () => loadDashboard(true);
$('autoRefresh').onchange = () => {
  $('autoRefresh').checked ? loadDashboard() : setConnection('Automatic refresh paused. The server continues collecting live status samples.');
};
$('resetZoom').onclick = () => {
  state.zoom.clear();
  for (const chart of state.charts) chart.draw();
};
$('exportCsv').onclick = () => {
  const metric = chosenMetric();
  const rows = [
    ['model_id', 'model', 'dataset', 'split', 'metric', 'epoch', 'raw_value']
  ];
  const seen = new Set();
  for (const chart of state.charts)
    if (!chart.live)
      for (const curve of chart.series)
        for (const [epoch, value] of curve.points) {
          const key = [curve.runId, curve.source, curve.split, curve.metric, epoch].join(':');
          if (!seen.has(key)) {
            seen.add(key);
            rows.push([curve.runId, curve.runName, curve.source, curve.split, curve.metric, epoch, value]);
          }
        }
  const cell = v => '"' + String(typeof v === 'string' && /^[=+@-]/.test(v) ? "'" + v : v).replaceAll('"', '""') + '"';
  save(new Blob([rows.map(r => r.map(cell).join(',')).join('\r\n')], {
    type: 'text/csv;charset=utf-8'
  }), 'tactile_training_' + metric + '.csv');
};
loadDashboard();
setInterval(() => {
  if ($('autoRefresh').checked && !document.hidden) loadDashboard();
}, 5000);
