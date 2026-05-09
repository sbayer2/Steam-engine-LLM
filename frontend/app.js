const BIT_LEVELS = [32, 16, 8, 4, 2, 1.58, 1];
const BIT_LABELS = ['32-bit FP32', '16-bit FP16', '8-bit INT8', '4-bit INT4', '2-bit INT2', '1.58-bit Ternary', '1-bit Binary'];
const LATENT_DIMS = [64, 48, 32, 16, 8, 4, 2, 1];
const WINDOW_SIZES = [64, 48, 32, 24, 16, 12, 8, 4, 2, 1];
const SEQ_LEN = 64;
const PATTERNS = ['Arithmetic', 'Geometric', 'Quadratic', 'Exponential', 'Fibonacci', 'AR1', 'Periodic', 'Damped', 'RandomWalk', 'Random'];
const PATTERN_COLORS = ['#58a6ff', '#3fb950', '#f0883e', '#ff7b72', '#d2a8ff', '#79c0ff', '#ffa657', '#7ee787', '#a5a5a5', '#8b949e'];

const COLOR_CLASS = '#3fb950';
const COLOR_PRED = '#f0883e';
const COLOR_LATENT = '#d2a8ff';

let charts = {};
let sweepCache = null;
let lastEval = null;
let mode = 'both';
let debounceTimer = null;

Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';

document.addEventListener('DOMContentLoaded', init);

async function init() {
    pollUntilReady();
}

async function pollUntilReady() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.ready) {
            document.getElementById('loading').classList.add('hidden');
            document.getElementById('app').classList.remove('hidden');
            setupControls();
            setupModeToggle();
            applyMode();
            await loadSweeps();
            await evaluate();
            return;
        }
    } catch (e) {}
    setTimeout(pollUntilReady, 600);
}

function setupControls() {
    ['bits-slider', 'latent-slider', 'state-slider'].forEach(id => {
        document.getElementById(id).addEventListener('input', () => {
            updateLabels();
            debouncedEvaluate();
        });
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        document.getElementById('bits-slider').value = 0;
        document.getElementById('latent-slider').value = 0;
        document.getElementById('state-slider').value = 0;
        updateLabels();
        evaluate();
    });

    document.getElementById('btn-max').addEventListener('click', () => {
        document.getElementById('bits-slider').value = BIT_LEVELS.length - 1;
        document.getElementById('latent-slider').value = LATENT_DIMS.length - 1;
        document.getElementById('state-slider').value = WINDOW_SIZES.length - 1;
        updateLabels();
        evaluate();
    });

    updateLabels();
}

function setupModeToggle() {
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            mode = btn.dataset.mode;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('active', b === btn));
            applyMode();
            redrawSweeps();
            if (lastEval) {
                renderAll(lastEval);
            }
        });
    });
}

function applyMode() {
    const showClass = mode === 'classify' || mode === 'both';
    const showPred = mode === 'predict' || mode === 'both';

    document.getElementById('card-class').style.display = showClass ? '' : 'none';
    document.getElementById('card-retained').style.display = showClass ? '' : 'none';
    document.getElementById('card-pred').style.display = showPred ? '' : 'none';
    document.getElementById('card-pred-retained').style.display = showPred ? '' : 'none';
    document.getElementById('card-divergence').style.display = showPred ? '' : 'none';

    document.body.dataset.mode = mode;

    const title = document.getElementById('samples-title');
    if (mode === 'classify') title.textContent = 'Live Classifications';
    else if (mode === 'predict') title.textContent = 'Live Forecasts';
    else title.textContent = 'Live Predictions & Forecasts';

    renderLegend();
}

function renderLegend() {
    const el = document.getElementById('per-class-legend');
    if (!el) return;
    const items = [];
    if (mode === 'classify' || mode === 'both') {
        items.push(`<span class="legend-item"><span class="legend-dot" style="background:${COLOR_CLASS}"></span>Accuracy</span>`);
    }
    if (mode === 'predict' || mode === 'both') {
        items.push(`<span class="legend-item"><span class="legend-dot" style="background:${COLOR_PRED}"></span>Raw MSE</span>`);
        items.push(`<span class="legend-item"><span class="legend-dot" style="background:${COLOR_LATENT}"></span>Latent MSE</span>`);
    }
    el.innerHTML = items.join('');
}

function updateLabels() {
    const bi = +document.getElementById('bits-slider').value;
    const li = +document.getElementById('latent-slider').value;
    const si = +document.getElementById('state-slider').value;

    document.getElementById('bits-label').textContent = BIT_LABELS[bi];
    document.getElementById('bits-comp').textContent = (32 / BIT_LEVELS[bi]).toFixed(1) + 'x';

    document.getElementById('latent-label').textContent = LATENT_DIMS[li] + ' dimensions';
    document.getElementById('latent-comp').textContent = (64 / LATENT_DIMS[li]).toFixed(1) + 'x';

    const w = WINDOW_SIZES[si];
    document.getElementById('state-label').textContent = w + ' tokens' + (si === 0 ? ' (full)' : '');
    document.getElementById('state-comp').textContent = (SEQ_LEN / w).toFixed(1) + 'x';
}

function getSettings() {
    const bi = +document.getElementById('bits-slider').value;
    const li = +document.getElementById('latent-slider').value;
    const si = +document.getElementById('state-slider').value;
    return {
        bits: BIT_LEVELS[bi],
        latent_ratio: LATENT_DIMS[li] / 64,
        state_ratio: WINDOW_SIZES[si] / SEQ_LEN,
        mode,
    };
}

function debouncedEvaluate() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(evaluate, 100);
}

async function evaluate() {
    const settings = getSettings();
    const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
    });
    const data = await res.json();
    if (data.error) return;
    lastEval = data;
    renderAll(data);
}

function renderAll(data) {
    updateMetrics(data);
    updateClassBars(data);
    updateWeightChart(data);
    updateSamples(data);
    updateThesisStatus(data);
}

function updateMetrics(data) {
    const el = (id) => document.getElementById(id);
    el('accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
    el('accuracy').style.color = accColor(data.accuracy);
    el('accuracy-sub').textContent = 'baseline: ' + (data.baseline * 100).toFixed(1) + '%';

    el('retained').textContent = (data.retained * 100).toFixed(1) + '%';
    el('retained').style.color = accColor(data.retained);

    const p = data.prediction;
    el('pred-mse').textContent = p.mse.toFixed(3);
    el('pred-mse').style.color = mseColor(p.mse, p.baseline_mse);
    el('pred-latent-mse').textContent = (p.latent_mse !== undefined) ? p.latent_mse.toFixed(3) : '—';
    if (p.latent_mse !== undefined && p.latent_baseline_mse !== undefined) {
        el('pred-latent-mse').style.color = mseColor(p.latent_mse, p.latent_baseline_mse);
    }
    el('pred-mse-sub').textContent =
        'base raw / lat: ' + p.baseline_mse.toFixed(3)
        + ' / ' + (p.latent_baseline_mse !== undefined ? p.latent_baseline_mse.toFixed(3) : '—');

    el('pred-retained').textContent = (p.retained * 100).toFixed(1) + '%';
    el('pred-retained').style.color = accColor(Math.min(p.retained, 1));
    if (p.latent_retained !== undefined) {
        el('pred-latent-retained').textContent = (p.latent_retained * 100).toFixed(1) + '%';
        el('pred-latent-retained').style.color = accColor(Math.min(p.latent_retained, 1));
    } else {
        el('pred-latent-retained').textContent = '—';
    }

    if (p.divergence_index !== undefined) {
        const di = p.divergence_index;
        const sign = di >= 0 ? '+' : '';
        el('divergence').textContent = sign + (di * 100).toFixed(1) + '%';
        // Positive (latent better) = success green; negative (latent worse, H_D) = warning amber
        el('divergence').style.color = di >= 0.05 ? '#3fb950' : (di > -0.05 ? '#8b949e' : '#f0883e');
        const interp = di >= 0.1 ? 'JEPA wins (H_A)'
            : (di <= -0.1 ? 'latent worse (H_D)'
            : 'roughly equal');
        el('divergence-sub').textContent = interp;
    }

    el('compression').textContent = data.compression.total + 'x';
    el('comp-breakdown').textContent =
        'bits ' + data.compression.bits + 'x · latent ' + data.compression.latent + 'x · state ' + data.compression.state + 'x';

    el('memory').textContent = data.memory_kb < 10
        ? data.memory_kb.toFixed(2) + ' KB'
        : data.memory_kb.toFixed(1) + ' KB';
    el('memory-sub').textContent = 'baseline: ' + data.baseline_kb.toFixed(1) + ' KB';
}

function accColor(v) {
    if (v >= 0.85) return '#3fb950';
    if (v >= 0.65) return '#d29922';
    if (v >= 0.4) return '#f0883e';
    return '#f85149';
}

function mseColor(mse, baseline) {
    const ratio = mse / Math.max(baseline, 1e-6);
    if (ratio <= 1.2) return '#3fb950';
    if (ratio <= 2.0) return '#d29922';
    if (ratio <= 4.0) return '#f0883e';
    return '#f85149';
}

function updateThesisStatus(data) {
    const el = document.getElementById('thesis-status');
    const comp = data.compression.total;

    let metric;
    if (mode === 'classify') {
        metric = data.retained;
    } else if (mode === 'predict') {
        metric = data.prediction.retained;
    } else {
        metric = Math.min(data.retained, data.prediction.retained);
    }

    if (comp <= 1.1) {
        el.textContent = 'No compression applied';
        el.style.color = 'var(--text-dim)';
    } else if (metric >= 0.9) {
        el.textContent = comp + 'x compressed — structure fully intact';
        el.style.color = 'var(--success)';
    } else if (metric >= 0.7) {
        el.textContent = comp + 'x compressed — thesis holds';
        el.style.color = 'var(--warning)';
    } else if (metric >= 0.4) {
        el.textContent = comp + 'x compressed — approaching cliff';
        el.style.color = 'var(--bits-color)';
    } else {
        el.textContent = comp + 'x compressed — cliff reached';
        el.style.color = 'var(--danger)';
    }
}

function updateClassBars(data) {
    const container = document.getElementById('class-bars');
    container.innerHTML = '';
    const showClass = mode === 'classify' || mode === 'both';
    const showPred = mode === 'predict' || mode === 'both';
    const mseValues = PATTERNS.map(n => (data.prediction.per_class_mse || {})[n] || 0);
    const latValues = PATTERNS.map(n => (data.prediction.per_class_latent_mse || {})[n] || 0);
    const maxMse = Math.max(...mseValues, 0.001);
    const maxLat = Math.max(...latValues, 0.001);

    PATTERNS.forEach((name, i) => {
        const acc = (data.per_class || {})[name] || 0;
        const accPct = (acc * 100).toFixed(1);
        const mse = (data.prediction.per_class_mse || {})[name] || 0;
        const lat = (data.prediction.per_class_latent_mse || {})[name] || 0;
        const msePct = (mse / maxMse * 100).toFixed(1);
        const latPct = (lat / maxLat * 100).toFixed(1);

        const row = document.createElement('div');
        row.className = 'class-bar';

        let html = '<span class="class-name">' + name + '</span>';

        if (showClass) {
            html += '<div class="class-track"><div class="class-fill" style="width:' + accPct + '%;background:' + COLOR_CLASS + '"></div></div>'
                 + '<span class="class-pct">' + accPct + '%</span>';
        }
        if (showPred) {
            html += '<div class="class-track"><div class="class-fill" style="width:' + msePct + '%;background:' + COLOR_PRED + '"></div></div>'
                 + '<span class="class-pct">' + mse.toFixed(3) + '</span>'
                 + '<div class="class-track"><div class="class-fill" style="width:' + latPct + '%;background:' + COLOR_LATENT + '"></div></div>'
                 + '<span class="class-pct">' + lat.toFixed(3) + '</span>';
        }
        row.innerHTML = html;
        container.appendChild(row);
    });
}

function sparkline(values) {
    const chars = '▁▂▃▄▅▆▇█';
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    return values.map(v => chars[Math.min(7, Math.floor((v - min) / range * 7.99))]).join('');
}

function predBlocks(predicted, actual) {
    const all = [...predicted, ...actual];
    const min = Math.min(...all);
    const max = Math.max(...all);
    const range = max - min || 1;
    const chars = '▁▂▃▄▅▆▇█';
    const enc = v => chars[Math.min(7, Math.floor((v - min) / range * 7.99))];
    const p = predicted.map(enc).join('');
    const a = actual.map(enc).join('');
    return { p, a };
}

function updateSamples(data) {
    const container = document.getElementById('sample-list');
    container.innerHTML = '';
    const showClass = mode === 'classify' || mode === 'both';
    const showPred = mode === 'predict' || mode === 'both';

    data.samples.forEach(s => {
        const row = document.createElement('div');
        row.className = 'sample-row';

        const ctx = s.seq.slice(0, s.seq.length - 16);
        const tail = s.seq.slice(s.seq.length - 16);
        const ctxSpark = sparkline(ctx);
        const blocks = predBlocks(s.predicted_values, s.actual_values);

        let html = '<span class="sample-sparkline">' + ctxSpark + '<span class="sample-divider">│</span>'
                 + '<span class="sample-actual">' + blocks.a + '</span></span>';

        if (showClass) {
            const icon = s.ok ? '✓' : '✗';
            const cls = s.ok ? 'sample-ok' : 'sample-fail';
            const detail = s.ok ? s.pred : s.pred + ' (was ' + s.actual + ')';
            html += '<span class="sample-pred ' + cls + '">' + icon + ' ' + detail + '</span>'
                 + '<span class="sample-conf">' + (s.conf * 100).toFixed(0) + '%</span>';
        }

        if (showPred) {
            const latMse = (s.latent_mse !== undefined) ? s.latent_mse.toFixed(3) : '—';
            html += '<span class="sample-forecast">'
                 +    '<span class="forecast-label">forecast</span>'
                 +    '<span class="forecast-blocks">' + blocks.p + '</span>'
                 +    '<span class="forecast-mse">raw ' + s.pred_mse.toFixed(3) + ' · lat ' + latMse + '</span>'
                 + '</span>';
        }

        row.innerHTML = html;
        container.appendChild(row);
    });
}

async function loadSweeps() {
    const res = await fetch('/api/sweeps');
    sweepCache = await res.json();
    if (sweepCache.error) return;
    redrawSweeps();
}

function buildSweepData(axis) {
    const cls = sweepCache.classify[axis];
    const prd = sweepCache.predict[axis];
    const lat = sweepCache.latent ? sweepCache.latent[axis] : null;
    const labels = cls.map(d => {
        if (axis === 'bits') return d.x + 'b';
        if (axis === 'latent') return d.x + 'd';
        return 'w' + d.x;
    });
    return { labels, cls, prd, lat };
}

function redrawSweeps() {
    if (!sweepCache || sweepCache.error) return;
    ['bits', 'latent', 'state'].forEach(axis => {
        const id = 'chart-' + axis;
        if (charts[axis]) {
            charts[axis].destroy();
            charts[axis] = null;
        }
        charts[axis] = makeSweepChart(id, axis);
    });
}

function makeSweepChart(canvasId, axis) {
    const data = buildSweepData(axis);
    const showClass = mode === 'classify' || mode === 'both';
    const showPred = mode === 'predict' || mode === 'both';
    const dual = mode === 'both';
    const xTitle = axis === 'bits' ? 'Bits' : axis === 'latent' ? 'Dims' : 'Window';

    const datasets = [];
    if (showClass) {
        datasets.push({
            label: 'Accuracy',
            data: data.cls.map(d => d.acc),
            borderColor: COLOR_CLASS,
            backgroundColor: 'rgba(63,185,80,0.08)',
            fill: !dual,
            tension: 0.3,
            pointRadius: 3,
            pointBackgroundColor: COLOR_CLASS,
            yAxisID: 'y',
        });
    }
    if (showPred) {
        datasets.push({
            label: 'Raw MSE',
            data: data.prd.map(d => d.mse),
            borderColor: COLOR_PRED,
            backgroundColor: 'rgba(240,136,62,0.08)',
            fill: false,
            tension: 0.3,
            pointRadius: 3,
            pointBackgroundColor: COLOR_PRED,
            yAxisID: dual ? 'y1' : 'y',
            borderDash: dual ? [4, 3] : [],
        });
        if (data.lat) {
            datasets.push({
                label: 'Latent MSE',
                data: data.lat.map(d => d.mse),
                borderColor: COLOR_LATENT,
                backgroundColor: 'rgba(210,168,255,0.08)',
                fill: false,
                tension: 0.3,
                pointRadius: 3,
                pointBackgroundColor: COLOR_LATENT,
                yAxisID: dual ? 'y1' : 'y',
                borderDash: [2, 2],
            });
        }
    }

    const scales = {
        x: {
            title: { display: true, text: xTitle, color: '#8b949e', font: { size: 9 } },
            ticks: { color: '#8b949e', font: { size: 8 } },
            grid: { color: 'rgba(255,255,255,0.04)' },
        },
    };

    if (showClass && (!dual || !showPred)) {
        scales.y = {
            position: 'left',
            title: { display: true, text: 'Accuracy', color: COLOR_CLASS, font: { size: 9 } },
            min: 0, max: 1,
            ticks: { color: '#8b949e', font: { size: 8 }, callback: v => (v * 100) + '%' },
            grid: { color: 'rgba(255,255,255,0.04)' },
        };
    } else if (showClass && dual) {
        scales.y = {
            position: 'left',
            title: { display: true, text: 'Acc', color: COLOR_CLASS, font: { size: 9 } },
            min: 0, max: 1,
            ticks: { color: '#8b949e', font: { size: 8 }, callback: v => (v * 100) + '%' },
            grid: { color: 'rgba(255,255,255,0.04)' },
        };
    }

    if (showPred && dual) {
        scales.y1 = {
            position: 'right',
            title: { display: true, text: 'MSE', color: COLOR_PRED, font: { size: 9 } },
            min: 0,
            ticks: { color: '#8b949e', font: { size: 8 } },
            grid: { drawOnChartArea: false },
        };
    } else if (showPred && !showClass) {
        scales.y = {
            position: 'left',
            title: { display: true, text: 'Pred MSE', color: COLOR_PRED, font: { size: 9 } },
            min: 0,
            ticks: { color: '#8b949e', font: { size: 8 } },
            grid: { color: 'rgba(255,255,255,0.04)' },
        };
    }

    return new Chart(document.getElementById(canvasId), {
        type: 'line',
        data: { labels: data.labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: { duration: 0 },
            plugins: {
                legend: {
                    display: showPred || dual,
                    labels: { color: '#8b949e', font: { size: 9 }, boxWidth: 8 },
                },
            },
            scales,
        },
    });
}

function updateWeightChart(data) {
    const weights = data.weights;
    if (!weights || !weights.length) return;

    const nBins = 40;
    const min = Math.min(...weights);
    const max = Math.max(...weights);
    const range = max - min || 1;
    const step = range / nBins;
    const bins = new Array(nBins).fill(0);
    const labels = [];

    for (let i = 0; i < nBins; i++) {
        labels.push((min + step * (i + 0.5)).toFixed(2));
    }
    weights.forEach(w => {
        const idx = Math.min(nBins - 1, Math.floor((w - min) / step));
        bins[idx]++;
    });

    if (charts.weights) {
        charts.weights.data.labels = labels;
        charts.weights.data.datasets[0].data = bins;
        charts.weights.update('none');
    } else {
        charts.weights = new Chart(document.getElementById('chart-weights'), {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    data: bins,
                    backgroundColor: 'rgba(210,168,255,0.4)',
                    borderColor: '#d2a8ff',
                    borderWidth: 1,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                animation: { duration: 0 },
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: {
                        ticks: { color: '#8b949e', font: { size: 8 } },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                    },
                },
            },
        });
    }
}
