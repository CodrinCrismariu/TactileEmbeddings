/**
 * Browse native sensor reconstructions, video predictions, and mirrored pairs.
 * Inputs: viewer.html controls and JSON from the shared viewer API.
 * Outputs: scene canvases, native target panels, and diagnostic text in the DOM.
 * Example: run `tl view --port 8080`, then open the reconstruction page.
 * This browser code requests inference; it never starts model training.
 */
'use strict';
const $ = id => document.getElementById(id);
const requested = new URLSearchParams(location.search).get('page') || 'reconstruction';
const page = requested === 'joint' ? 'prediction' : requested === 'sensor' ? 'reconstruction' : requested;
const base = new URL(location.pathname.replace(/\/?$/, '/'), location.origin);
const api = (name, params = {}) => new URL('api/' + name + '?' + new URLSearchParams(params), base);
let catalog, episodes = [],
  data = null,
  mirroredData = null,
  playing = false,
  busy = false,
  pendingFrame = false,
  timer = null;
const sourceSpec = () => catalog.sources.find(item => item.id === $('source').value);
const predictionMode = () => page === 'prediction' || (page === 'mirror' && $('mirrorMode').value === 'prediction');

function options(select, values, chosen) {
  select.replaceChildren();
  for (const item of values) {
    const option = document.createElement('option');
    option.value = item.value;
    option.textContent = item.label;
    select.append(option);
  }
  if (values.some(item => item.value === chosen)) select.value = chosen;
}
async function json(url) {
  const response = await fetch(url);
  const result = await response.json();
  if (!response.ok) throw Error(result.error || 'Request failed');
  return result;
}

function error(problem) {
  $('error').textContent = problem ? problem.message || String(problem) : '';
}
async function loadCatalog(refresh = false) {
  // The server's registry determines runnable models and supported sources.
  try {
    catalog = await json(api('catalog', {
      refresh: refresh ? '1' : '0'
    }));
    const chosen = $('source').value;
    options($('source'), catalog.sources.map(item => ({
      value: item.id,
      label: item.id.replaceAll('_', ' ') + (item.heldout ? ' · held out' : '')
    })), chosen || catalog.sources[0]?.id);
    await sourceChanged();
  } catch (problem) {
    error(problem);
  }
}
async function sourceChanged() {
  const spec = sourceSpec();
  options($('split'), spec.splits.map(value => ({
    value,
    label: value
  })), spec.splits.includes('val') ? 'val' : spec.splits[0]);
  $('mirrored').disabled = !spec.mirrored || page === 'mirror';
  if (!spec.mirrored) $('mirrored').checked = false;
  modelsChanged();
  await loadEpisodes();
}

function modelsChanged() {
  const spec = sourceSpec();
  const previous = $('model').value;
  const candidates = catalog.models.filter(item => item.sources.includes(spec.id));
  if (predictionMode()) {
    options($('model'), candidates.map(item => ({
      value: item.id,
      label: item.name
    })), previous);
    $('model').disabled = false;
    $('epoch').disabled = false;
    epochsChanged();
    $('modelNote').textContent = candidates.length ? 'Each epoch uses its own recorded sensor decoder.' : 'No compatible current model for this source. Earlier models and transfer probes remain in Historical models.';
  } else {
    options($('model'), [{
      value: 'teacher',
      label: spec.teacher ? 'Accepted sensor AE · ' + spec.code_dim + 'D' : 'Direct numeric sensor · no AE'
    }]);
    $('model').disabled = true;
    options($('epoch'), [{
      value: 'accepted',
      label: 'Best accepted teacher'
    }]);
    $('epoch').disabled = true;
    $('modelNote').textContent = 'Original and mirrored measurements use the same accepted decoder. All older sizes and wide variants are available in Historical models.';
  }
  $('attention').disabled = !predictionMode();
  if (!predictionMode()) $('attention').checked = false;
}

function epochsChanged() {
  const model = catalog.models.find(item => item.id === $('model').value);
  options($('epoch'), (model?.epochs || []).map(value => ({
    value,
    label: value
  })), model?.epochs?.at(-1));
}
async function loadEpisodes() {
  try {
    error(null);
    episodes = await json(api('episodes', {
      source: $('source').value,
      split: $('split').value,
      filter: $('filter').value
    }));
    options($('episode'), episodes.map(item => ({
      value: item.id,
      label: item.id + ' · ' + item.frames + ' frames'
    })));
    await episodeChanged();
  } catch (problem) {
    error(problem);
  }
}
async function episodeChanged() {
  playing = false;
  $('play').textContent = '▶ Play';
  const episode = episodes.find(item => item.id === $('episode').value);
  if (!episode) return;
  const maximum = episode.frames - 1;
  $('slider').max = $('frame').max = maximum;
  $('frame').value = $('slider').value = 0;
  await loadFrame();
}

/** Build one frame request without transforming arbitrary sensor-code axes. */
function requestParameters(mirror = false) {
  return {
    source: $('source').value,
    episode: $('episode').value,
    frame: $('frame').value,
    page: predictionMode() ? 'prediction' : 'reconstruction',
    mirror: mirror ? '1' : '0',
    model: $('model').value,
    epoch: $('epoch').value,
    attention: $('attention').checked ? '1' : '0'
  };
}
/** Fetch the selected frame and coalesce slider changes while a request runs. */
async function loadFrame() {
  if (busy) {
    pendingFrame = true;
    return;
  }
  if (!$('episode').value) return;
  if (page === 'mirror' && !sourceSpec().mirrored) {
    error(Error('This source has no verified mirror mapping. Choose another source.'));
    return;
  }
  busy = true;
  pendingFrame = false;
  $('status').textContent = predictionMode() ? 'Computing video features and current-frame prediction…' : 'Loading native measurements and reconstruction…';
  try {
    error(null);
    data = await json(api('frame', requestParameters(page === 'mirror' ? false : $('mirrored').checked)));
    mirroredData = page === 'mirror' ? await json(api('frame', requestParameters(true))) : null;
    draw();
    $('status').textContent = 'Frame ' + data.frame + ' · ' + data.decoder + (data.epoch ? ' · epoch ' + data.epoch : '');
  } catch (problem) {
    error(problem);
    playing = false;
    $('play').textContent = '▶ Play';
  } finally {
    busy = false;
    if (pendingFrame) {
      pendingFrame = false;
      loadFrame();
    } else if (playing) timer = setTimeout(() => step(1), 100);
  }
}

function draw() {
  if (!data) return;
  $('frameLabel').textContent = '/ ' + (data.frames - 1) + ' · history [' + data.history_frame_ids.join(', ') + ']';
  const selected = $('camera').value;
  options($('camera'), data.scenes.map(item => ({
    value: String(item.slot),
    label: item.label
  })), selected);
  if (data.attention) {
    options($('layer'), Array.from({
      length: data.attention.layers
    }, (_, i) => ({
      value: String(i),
      label: 'Layer ' + (i + 1)
    })), $('layer').value || String(data.attention.layers - 1));
    options($('head'), [{
      value: 'all',
      label: 'Mean of heads'
    }, ...Array.from({
      length: data.attention.heads
    }, (_, i) => ({
      value: String(i),
      label: 'Head ' + (i + 1)
    }))], $('head').value || 'all');
  }
  drawScene(data, $('scene'));
  renderTarget(data.target, $('target'));
  $('metrics').textContent = metricText(data);
  $('comparison').hidden = !mirroredData;
  if (mirroredData) {
    drawScene(mirroredData, $('mirrorScene'));
    renderTarget(mirroredData.target, $('mirrorTarget'));
    $('mirrorMetrics').textContent = metricText(mirroredData);
  }
  $('targetTitle').textContent = (predictionMode() ? 'Measured vs predicted' : 'Measured vs reconstructed') + ' · native ' + data.target.shape.join(' × ');
  $('sceneNote').textContent = data.attention ? 'Overlay: current query → contextual patches. One common intensity scale across cameras; background darkened.' : 'Eight RGB frames are input; the target belongs to the current frame.';
}

function metricText(value) {
  const target = value.target;
  return 'MAE ' + (target.mae === null ? 'unavailable' : target.mae.toPrecision(5)) + ' ' + target.units + ' · ' + target.valid_count + ' valid values' + (value.mirrored ? ' · mirrored' : '') + (target.valid_count ? '' : ' · no measured target at this frame');
}
/** Draw RGB and optional query-to-patch weights for one camera/time group. */
async function drawScene(value, canvas) {
  const scene = value.scenes.find(item => String(item.slot) === $('camera').value) || value.scenes[0];
  if (!scene) return;
  const group = Number($('group').value);
  const enabled = $('attention').checked && value.attention;
  const source = enabled ? scene.history_groups[group] : scene.image;
  const image = new Image();
  image.src = source;
  await image.decode();
  const context = canvas.getContext('2d');
  context.globalAlpha = 1;
  context.clearRect(0, 0, 384, 384);
  context.drawImage(image, 0, 0, 384, 384);
  if (!enabled) return;
  context.fillStyle = 'rgba(0,0,0,.70)';
  context.fillRect(0, 0, 384, 384);
  const weights = value.attention.weights[Number($('layer').value) || 0];
  const choices = $('head').value === 'all' ? weights : [weights[Number($('head').value)]];
  let maximum = 0;
  const grid = value.attention.grid;
  const selected = Array.from({
    length: grid
  }, () => Array(grid).fill(0));
  for (let t = 0; t < 4; t++)
    for (let camera = 0; camera < 3; camera++)
      for (let y = 0; y < grid; y++)
        for (let x = 0; x < grid; x++) {
          const score = choices.reduce((sum, head) => sum + head[t][camera][y][x], 0) / choices.length;
          maximum = Math.max(maximum, score);
          if (t === group && camera === scene.slot) selected[y][x] = score;
        }
  const cell = 384 / grid;
  for (let y = 0; y < grid; y++)
    for (let x = 0; x < grid; x++) {
      const level = maximum ? selected[y][x] / maximum : 0;
      if (level < .04) continue;
      context.fillStyle = 'hsla(' + (60 * (1 - level)) + ',100%,55%,' + (.85 * Math.sqrt(level)) + ')';
      context.fillRect(x * cell, y * cell, cell + .2, cell + .2);
    }
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

/** Dispatch native target metadata to grid, numeric, or marker renderers. */
function renderTarget(target, container) {
  container.replaceChildren();
  if (target.kind === 'marker_displacement') {
    drawMarkers(target, container);
    return;
  }
  if (target.values) {
    drawNumbers(target, container);
    return;
  }
  const grid = node('div', undefined, 'sensorGrid');
  for (const panel of target.panels || []) {
    const card = node('div', undefined, 'panel');
    card.append(node('h3', panel.label));
    const pair = node('div', undefined, 'pair');
    for (const [label, url] of [
        ['Measured', panel.original],
        [predictionMode() ? 'Predicted' : 'Reconstructed', panel.prediction]
      ]) {
      const column = node('div');
      column.append(node('small', label));
      if (url) {
        const image = node('img');
        image.src = url;
        column.append(image);
      }
      pair.append(column);
    }
    card.append(pair);
    grid.append(card);
  }
  container.append(grid);
}

/** Compare channels on server-declared fixed scales, preserving signed values. */
function drawNumbers(target, container) {
  const wrapper = node('div', undefined, 'numeric');
  const canvas = node('canvas');
  canvas.width = 800;
  canvas.height = Math.max(140, target.values.length * 27 + 35);
  wrapper.append(canvas);
  container.append(wrapper);
  const ctx = canvas.getContext('2d');
  ctx.font = '12px system-ui';
  for (let i = 0; i < target.values.length; i++) {
    const y = 20 + i * 27;
    const center = 400;
    const span = 300;
    ctx.fillStyle = '#a9bfd5';
    ctx.fillText(String(target.labels[i]), 5, y + 4);
    ctx.strokeStyle = '#496075';
    ctx.beginPath();
    ctx.moveTo(center, y - 10);
    ctx.lineTo(center, y + 12);
    ctx.stroke();
    if (!target.valid[i]) {
      ctx.fillStyle = '#9ba4b0';
      ctx.fillText('missing', 430, y + 4);
      continue;
    }
    const scale = target.limits[i];
    for (const [value, color, offset] of [
        [target.values[i], '#5ee1d0', -8],
        [target.prediction?.[i], '#ffb75b', 1]
      ]) {
      if (value === undefined) continue;
      ctx.fillStyle = color;
      const width = Math.max(-span, Math.min(span, value / scale * span));
      ctx.fillRect(center + Math.min(0, width), y + offset, Math.abs(width), 7);
    }
    ctx.fillStyle = '#c3d5e6';
    ctx.fillText(target.values[i].toPrecision(4) + (target.prediction ? ' / ' + target.prediction[i].toPrecision(4) : ''), center + span + 10, y + 3);
  }
  container.prepend(node('p', 'Cyan: measured · orange: reconstruction/prediction. Fixed source scale; zero is unchanged.', 'note'));
}

/** Draw signed measured/predicted arrows with a display-only magnification. */
function drawMarkers(target, container) {
  const gain = Number($('gain').value);
  const surfaces = [...new Set(target.surface)].sort((a, b) => a - b);
  const grid = node('div', undefined, 'sensorGrid');
  for (const surface of surfaces) {
    const card = node('div', undefined, 'panel');
    card.append(node('h3', 'Surface ' + surface + ' · arrows ×' + gain));
    const canvas = node('canvas', undefined, 'marker');
    canvas.width = canvas.height = 384;
    card.append(canvas);
    grid.append(card);
    const ctx = canvas.getContext('2d');
    const [height, width] = target.surface_shape || [384, 384];
    ctx.strokeStyle = '#2d4358';
    for (let i = 0; i <= 4; i++) {
      ctx.beginPath();
      ctx.moveTo(i * 96, 0);
      ctx.lineTo(i * 96, 384);
      ctx.moveTo(0, i * 96);
      ctx.lineTo(384, i * 96);
      ctx.stroke();
    }
    for (let i = 0; i < target.surface.length; i++) {
      if (target.surface[i] !== surface) continue;
      const [u, v] = target.reference[i];
      const x = u * 384,
        y = v * 384;
      ctx.fillStyle = target.valid[i] ? '#8194a8' : '#4f5965';
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
      if (!target.valid[i]) continue;
      for (const [values, color] of [
          [target.values, '#5ee1d0'],
          [target.prediction, '#ffb75b']
        ]) {
        if (!values) continue;
        const dx = values[i][0] * 384 / width * gain,
          dy = values[i][1] * 384 / height * gain;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1.7;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + dx, y + dy);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x + dx, y + dy, 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  container.append(node('p', 'Cyan: measured · orange: reconstruction/prediction · gray: reference or missing. Arrow amplification changes display only.', 'note'), grid);
}

function step(amount) {
  const maximum = Number($('frame').max);
  const next = Math.max(0, Math.min(maximum, Number($('frame').value) + amount));
  if (next === Number($('frame').value)) {
    playing = false;
    $('play').textContent = '▶ Play';
    return;
  }
  $('frame').value = $('slider').value = next;
  loadFrame();
}
for (const link of document.querySelectorAll('nav a[data-page]')) link.classList.toggle('active', link.dataset.page === page);
$('mirrorModeLabel').hidden = page !== 'mirror';
$('mirrored').parentElement.hidden = page === 'mirror';
$('source').onchange = sourceChanged;
$('split').onchange = loadEpisodes;
$('filter').oninput = () => {
  clearTimeout(timer);
  timer = setTimeout(loadEpisodes, 250);
};
$('episode').onchange = episodeChanged;
$('model').onchange = () => {
  epochsChanged();
  loadFrame();
};
$('epoch').onchange = loadFrame;
$('refresh').onclick = () => loadCatalog(true);
$('mirrorMode').onchange = () => {
  modelsChanged();
  loadFrame();
};
$('mirrored').onchange = loadFrame;
$('attention').onchange = loadFrame;
$('frame').onchange = () => {
  $('slider').value = $('frame').value;
  loadFrame();
};
$('slider').oninput = () => {
  $('frame').value = $('slider').value;
  clearTimeout(timer);
  timer = setTimeout(loadFrame, 120);
};
$('previous').onclick = () => step(-1);
$('next').onclick = () => step(1);
$('play').onclick = () => {
  playing = !playing;
  $('play').textContent = playing ? '❚❚ Pause' : '▶ Play';
  if (playing) step(1);
  else clearTimeout(timer);
};
for (const id of ['layer', 'head', 'group', 'camera']) $(id).onchange = draw;
$('gain').oninput = () => {
  $('gainLabel').textContent = $('gain').value + ' · display only';
  draw();
};
loadCatalog();
