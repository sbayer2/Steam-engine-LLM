const BIT_LEVELS = [32, 16, 8, 4, 2, 1.58, 1];
const BIT_LABELS = ['32-bit FP32', '16-bit FP16', '8-bit INT8', '4-bit INT4', '2-bit INT2', '1.58-bit Ternary', '1-bit Binary'];
const LATENT_DIMS = [64, 48, 32, 16, 8, 4, 2, 1];
const WINDOW_SIZES = [16, 12, 8, 6, 4, 3, 2, 1];
const PATTERNS = ['Arithmetic', 'Geometric', 'Periodic', 'Fibonacci', 'Random'];
const PATTERN_COLORS = ['#58a6ff', '#3fb950', '#f0883e', '#d2a8ff', '#8b949e'];

let charts = {};
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
        document.getElementById('bits-slider').value = 6;
        document.getElementById('latent-slider').value = 7;
        document.getElementById('state-slider').value = 7;
        updateLabels();
        evaluate();
    });

    updateLabels();
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
    document.getElementById('state-comp').textContent = (16 / w).toFixed(1) + 'x';
}

function getSettings() {
    const bi = +document.getElementById('bits-slider').value;
    const li = +document.getElementById('latent-slider').value;
    const si = +document.getElementById('state-slider').value;
    return {
        bits: BIT_LEVELS[bi],
        latent_ratio: LATENT_DIMS[li] / 64,
        state_ratio: WINDOW_SIZES[si] / 16,
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

function updateThesisStatus(data) {
    const el = document.getElementById('thesis-status');
    const comp = data.compression.total;
    const ret = data.retained;

    if (comp <= 1.1) {
        el.textContent = 'No compression applied';
        el.style.color = 'var(--text-dim)';
    } else if (ret >= 0.9) {
        el.textContent = comp + 'x compressed — structure fully intact';
        el.style.color = 'var(--success)';
    } else if (ret >= 0.7) {
        el.textContent = comp + 'x compressed — thesis holds';
        el.style.color = 'var(--warning)';
    } else if (ret >= 0.4) {
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
    PATTERNS.forEach((name, i) => {
        const val = data.per_class[name] || 0;
        const pct = (val * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'class-bar';
        row.innerHTML =
            '<span class="class-name">' + name + '</span>' +
            '<div class="class-track"><div class="class-fill" style="width:' + pct + '%;background:' + PATTERN_COLORS[i] + '"></div></div>' +
            '<span class="class-pct">' + pct + '%</span>';
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

function updateSamples(data) {
    const container = document.getElementById('sample-list');
    container.innerHTML = '';
    data.samples.forEach(s => {
        const row = document.createElement('div');
        row.className = 'sample-row';
        const icon = s.ok ? '✓' : '✗';
        const cls = s.ok ? 'sample-ok' : 'sample-fail';
        const detail = s.ok ? s.pred : s.pred + ' (was ' + s.actual + ')';
        row.innerHTML =
            '<span class="sample-sparkline">' + sparkline(s.seq) + '</span>' +
            '<span class="sample-pred ' + cls + '">' + icon + ' ' + detail + '</span>' +
            '<span class="sample-conf">' + (s.conf * 100).toFixed(0) + '%</span>';
        container.appendChild(row);
    });
}

async function loadSweeps() {
    const res = await fetch('/api/sweeps');
    const sweeps = await res.json();
    if (sweeps.error) return;

    const baseOpts = (color, xLabel) => ({
        responsive: true,
        maintainAspectRatio: true,
        animation: { duration: 0 },
        plugins: { legend: { display: false } },
        scales: {
            x: {
                title: { display: true, text: xLabel, color: '#8b949e', font: { size: 9 } },
                ticks: { color: '#8b949e', font: { size: 8 } },
                grid: { color: 'rgba(255,255,255,0.04)' },
            },
            y: {
                title: { display: true, text: 'Accuracy', color: '#8b949e', font: { size: 9 } },
                min: 0, max: 1,
                ticks: { color: '#8b949e', font: { size: 8 }, callback: v => (v * 100) + '%' },
                grid: { color: 'rgba(255,255,255,0.04)' },
            },
        },
    });

    charts.bits = new Chart(document.getElementById('chart-bits'), {
        type: 'line',
        data: {
            labels: sweeps.bits.map(d => d.x <= 2 ? d.x + 'b' : d.x + 'b'),
            datasets: [{
                data: sweeps.bits.map(d => d.acc),
                borderColor: '#f0883e',
                backgroundColor: 'rgba(240,136,62,0.08)',
                fill: true, tension: 0.3, pointRadius: 3, pointBackgroundColor: '#f0883e',
            }],
        },
        options: baseOpts('#f0883e', 'Bits'),
    });

    charts.latent = new Chart(document.getElementById('chart-latent'), {
        type: 'line',
        data: {
            labels: sweeps.latent.map(d => d.x + 'd'),
            datasets: [{
                data: sweeps.latent.map(d => d.acc),
                borderColor: '#3fb950',
                backgroundColor: 'rgba(63,185,80,0.08)',
                fill: true, tension: 0.3, pointRadius: 3, pointBackgroundColor: '#3fb950',
            }],
        },
        options: baseOpts('#3fb950', 'Dims'),
    });

    charts.state = new Chart(document.getElementById('chart-state'), {
        type: 'line',
        data: {
            labels: sweeps.state.map(d => 'w' + d.x),
            datasets: [{
                data: sweeps.state.map(d => d.acc),
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88,166,255,0.08)',
                fill: true, tension: 0.3, pointRadius: 3, pointBackgroundColor: '#58a6ff',
            }],
        },
        options: baseOpts('#58a6ff', 'Window'),
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
