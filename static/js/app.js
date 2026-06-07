/* ── StockSense Frontend ─────────────────────────────────────────────────── */

let priceChart = null;

// ── Dataset registry for toggle control ──────────────────────────────────── //
const LAYER_COLORS = {
  actual:   { color: '#00e5a0', label: 'Actual Close' },
  predicted:{ color: '#5b8aff', label: 'Predicted Close' },
  ma100:    { color: '#ffb347', label: 'MA 100' },
  ma200:    { color: '#ff6b6b', label: 'MA 200' },
  ema100:   { color: '#c77dff', label: 'EMA 100' },
  ema200:   { color: '#f72585', label: 'EMA 200' },
};

// Map layer key → dataset index in Chart.js datasets array
const LAYER_INDEX = {
  actual: 0, predicted: 1, ma100: 2, ma200: 3, ema100: 4, ema200: 5,
};

// ── DOM Helpers ───────────────────────────────────────────────────────────── //
const $ = (id) => document.getElementById(id);

function showEl(id)  { $(id).classList.remove('hidden'); }
function hideEl(id)  { $(id).classList.add('hidden'); }
function setText(id, val) { $(id).textContent = val; }

function setLoaderStatus(msg) {
  setText('loader-status', msg);
}

// ── Quick Picks ───────────────────────────────────────────────────────────── //
document.querySelectorAll('.chip').forEach(btn => {
  btn.addEventListener('click', () => {
    $('ticker').value = btn.dataset.ticker;
    $('ticker').focus();
  });
});

// ── Enter key triggers prediction ─────────────────────────────────────────── //
$('ticker').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') runPrediction();
});

// ── Chart toggle buttons ──────────────────────────────────────────────────── //
document.querySelectorAll('.toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (!priceChart) return;
    const layer = btn.dataset.layer;
    const idx   = LAYER_INDEX[layer];
    const ds    = priceChart.data.datasets[idx];
    ds.hidden   = !ds.hidden;
    btn.classList.toggle('active', !ds.hidden);
    priceChart.update();
  });
});

// ── Prediction button ─────────────────────────────────────────────────────── //
$('predict-btn').addEventListener('click', runPrediction);

async function runPrediction() {
  const ticker = $('ticker').value.trim().toUpperCase();
  const years  = parseInt($('years').value, 10);

  // Validate client-side
  if (!ticker) {
    showError('Please enter a ticker symbol.');
    return;
  }

  hideEl('error-banner');
  hideEl('results');
  showEl('loader');

  const btn = $('predict-btn');
  btn.disabled = true;

  // Cycle loader messages
  const messages = [
    'Fetching market data…',
    'Preprocessing close prices…',
    'Running LSTM inference…',
    'Computing moving averages…',
    'Calculating metrics…',
  ];
  let msgIdx = 0;
  const msgInterval = setInterval(() => {
    msgIdx = (msgIdx + 1) % messages.length;
    setLoaderStatus(messages[msgIdx]);
  }, 1800);

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, years }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || 'Prediction failed.');
    }

    renderResults(data);

  } catch (err) {
    showError(err.message);
  } finally {
    clearInterval(msgInterval);
    hideEl('loader');
    btn.disabled = false;
  }
}

// ── Render results ────────────────────────────────────────────────────────── //
function renderResults(d) {
  // Header
  setText('res-ticker', d.ticker);
  setText('res-date', `Last data: ${d.last_date} · ${d.data_points.toLocaleString()} sessions loaded`);

  // Signal badge
  const badge = $('signal-badge');
  badge.textContent = d.signal;
  badge.className   = 'signal-badge ' + d.signal.toLowerCase();

  // KPIs
  setText('kpi-current',   fmt(d.current_price));
  setText('kpi-predicted', fmt(d.next_pred));
  const pctEl = $('kpi-pct');
  pctEl.textContent = `${d.pct_change >= 0 ? '+' : ''}${d.pct_change}%`;
  pctEl.className   = 'kpi-pct ' + (d.pct_change >= 0 ? 'up' : 'down');
  setText('kpi-mae',  fmt(d.metrics.mae));
  setText('kpi-rmse', fmt(d.metrics.rmse));
  setText('kpi-mape', d.metrics.mape.toFixed(2) + '%');

  // Chart
  renderChart(d.chart);

  showEl('results');
  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function fmt(n) {
  return typeof n === 'number'
    ? n.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : '—';
}

// ── Chart rendering ───────────────────────────────────────────────────────── //
function renderChart(chart) {
  const ctx = document.getElementById('price-chart').getContext('2d');

  if (priceChart) {
    priceChart.destroy();
    priceChart = null;
  }

  // Reset toggle buttons to match initial visibility
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    const visible = ['actual', 'predicted'].includes(btn.dataset.layer);
    btn.classList.toggle('active', visible);
  });

  const datasets = [
    buildDataset('actual',    chart.actual,              false),
    buildDataset('predicted', chart.predicted,           false),
    buildDataset('ma100',     chart.indicators.ma100,    true),
    buildDataset('ma200',     chart.indicators.ma200,    true),
    buildDataset('ema100',    chart.indicators.ema100,   true),
    buildDataset('ema200',    chart.indicators.ema200,   true),
  ];

  // Downsample dates for X axis readability (show ~12 labels)
  const labels = chart.dates;

  priceChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      animation:   { duration: 600, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(17,17,24,0.95)',
          borderColor:     'rgba(255,255,255,0.08)',
          borderWidth: 1,
          titleFont:  { family: "'JetBrains Mono', monospace", size: 11 },
          bodyFont:   { family: "'JetBrains Mono', monospace", size: 11 },
          callbacks: {
            label(ctx) {
              const v = ctx.raw;
              if (v === null || v === undefined) return null;
              return ` ${ctx.dataset.label}: ₹${v.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: {
            color: '#6b6b80',
            font: { family: "'JetBrains Mono', monospace", size: 10 },
            maxTicksLimit: 10,
            maxRotation: 0,
          },
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: {
            color: '#6b6b80',
            font: { family: "'JetBrains Mono', monospace", size: 10 },
            callback: (v) => '₹' + v.toLocaleString('en-IN'),
          },
        },
      },
    },
  });
}

function buildDataset(key, data, hidden) {
  const cfg = LAYER_COLORS[key];
  return {
    label:        cfg.label,
    data,
    hidden,
    borderColor:  cfg.color,
    borderWidth:  key === 'actual' || key === 'predicted' ? 1.8 : 1.2,
    pointRadius:  0,
    tension:      0.3,
    fill:         false,
    spanGaps:     true,
  };
}

// ── Error display ─────────────────────────────────────────────────────────── //
function showError(msg) {
  setText('error-msg', msg);
  showEl('error-banner');
  hideEl('results');
}
