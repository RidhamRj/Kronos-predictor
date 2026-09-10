/**
 * app.js
 * ------
 * Moneycontrol & Broker Live MCX Gold Mini (Goldm) Terminal & Kronos AI Predictor.
 * - Live real-time price tick polling loop (1.5s interval)
 * - Visual price flash animations (green/red) on live price changes
 * - Dynamic live candlestick body/wick movement in real time
 * - Moneycontrol 5-card commodity overview metrics:
 *     Open, Prev Close, Day Range (L-H), ATP, Traded Volume & Value, OI, OI Buildup, Market Depth
 * - MC Classic Floor Trader Pivot Levels (S3 to R3) & Technical Stance
 * - Multi-step Kronos Foundation Model (AAAI 2026) probabilistic forecast with 90% confidence bands
 */

// State Management
const state = {
  expiry: "2026-10-05",
  interval: "1h",
  horizon: 12,
  samples: 10,
  temperature: 0.2,
  liveData: null,
  forecastData: null,
  hoveredIndex: null,
  mousePos: { x: null, y: null },
  isHovering: false,
  // Navigation State (TradingView Style Smooth Pan & Zoom)
  visibleBars: 65,        // number of candles visible across default viewport
  panPixelOffset: 0,      // continuous pixel offset: 0 = live edge; > 0 = panned into history
  isDragging: false,
  dragStartX: 0,
  dragStartPanOffset: 0,
  // Prediction Comparison & Active Candle State
  compareMode: false,
  predictionsList: [],
  selectedPrediction: null,
  overlayPrediction: true,
};

let prevPrice = null;
const POLL_INTERVAL_SECONDS = 1;
let remainingSeconds = POLL_INTERVAL_SECONDS;
let countdownTimer = null;
let tickCount = 0;

function formatINR(val, decimals = 2) {
  if (val === null || val === undefined || isNaN(val)) return "--";
  return new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(val);
}

// DOM References
const spotPriceEl = document.getElementById("spot-price");
const changeAbsEl = document.getElementById("change-abs");
const changePctEl = document.getElementById("change-pct");
const changeWrapperEl = document.getElementById("price-change-wrapper");
const lastSyncEl = document.getElementById("last-sync-time");
const expirySelectEl = document.getElementById("expiry-select");
const lotSizeBadgeEl = document.getElementById("lot-size-badge");
const chartStatusEl = document.getElementById("chart-status-info");

// Moneycontrol Overview Cards
const ovOpenEl = document.getElementById("overview-open");
const ovPrevCloseEl = document.getElementById("overview-prev-close");
const ovLowEl = document.getElementById("overview-low");
const ovHighEl = document.getElementById("overview-high");
const rangePinEl = document.getElementById("range-bullet-pin");
const ovAtpEl = document.getElementById("overview-atp");
const ovVolValEl = document.getElementById("overview-vol-val");
const ovOiEl = document.getElementById("overview-oi");
const ovOiChgEl = document.getElementById("overview-oi-chg");
const ovOiBuildupEl = document.getElementById("overview-oi-buildup");
const ovBidEl = document.getElementById("overview-bid");
const ovAskEl = document.getElementById("overview-ask");

// Technicals & Pivots
const techStanceBadgeEl = document.getElementById("tech-stance-badge");
const pivotS3El = document.getElementById("pivot-s3");
const pivotS2El = document.getElementById("pivot-s2");
const pivotS1El = document.getElementById("pivot-s1");
const pivotPEl = document.getElementById("pivot-p");
const pivotR1El = document.getElementById("pivot-r1");
const pivotR2El = document.getElementById("pivot-r2");
const pivotR3El = document.getElementById("pivot-r3");

// Buttons & Controls
const btnRefresh = document.getElementById("btn-refresh-live");
const btnForecast = document.getElementById("btn-run-forecast");
const intervalSelector = document.getElementById("interval-selector");
const horizonSelect = document.getElementById("horizon-select");
const samplesSelect = document.getElementById("samples-select");
const tempSlider = document.getElementById("temp-slider");
const tempVal = document.getElementById("temp-val");

// Canvases & Navigation Controls
const mainCanvas = document.getElementById("main-candlestick-canvas");
const mainCtx = mainCanvas.getContext("2d");
const subCanvas = document.getElementById("sub-oscillator-canvas");
const subCtx = subCanvas.getContext("2d");
const chartTooltip = document.getElementById("chart-tooltip");
const chartWrapper = document.getElementById("chart-wrapper");
const btnZoomIn = document.getElementById("btn-chart-zoom-in");
const btnZoomOut = document.getElementById("btn-chart-zoom-out");
const btnReset = document.getElementById("btn-chart-reset");
const quickRangeSelector = document.getElementById("quick-range-selector");
const visibleRangeInfoEl = document.getElementById("visible-range-info");

// Mode & Comparison Controls
const tabModeLive = document.getElementById("tab-mode-live");
const tabModeCompare = document.getElementById("tab-mode-compare");
const comparePanel = document.getElementById("compare-panel");
const pastPredictionSelect = document.getElementById("past-prediction-select");
const chkOverlayPrediction = document.getElementById("chk-overlay-prediction");
const btnSaveCurrentPred = document.getElementById("btn-save-current-pred");
const compareHistoryCountEl = document.getElementById("compare-history-count");
const formingTimerTextEl = document.getElementById("forming-timer-text");
const legendOverlayItem = document.getElementById("legend-overlay-item");

// Initialization
window.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  resizeCanvases();
  window.addEventListener("resize", () => {
    resizeCanvases();
    renderCharts();
  });
  loadInitialData();
  startLiveTickStream();
});

function setupEventListeners() {
  // Expiry Dropdown
  expirySelectEl.addEventListener("change", (e) => {
    state.expiry = e.target.value;
    prevPrice = null;
    remainingSeconds = POLL_INTERVAL_SECONDS;
    updateCountdownUI();
    loadInitialData();
  });

  // Timeframe Interval Selection
  intervalSelector.querySelectorAll(".pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      intervalSelector.querySelectorAll(".pill").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.interval = btn.dataset.interval;
      state.panPixelOffset = 0; // reset to latest when changing timeframe
      updateActiveCandleTimer();
      loadInitialData();
    });
  });

  // Horizon & Samples
  horizonSelect.addEventListener("change", (e) => (state.horizon = parseInt(e.target.value)));
  samplesSelect.addEventListener("change", (e) => (state.samples = parseInt(e.target.value)));

  // Temperature Slider
  tempSlider.addEventListener("input", (e) => {
    state.temperature = parseFloat(e.target.value);
    tempVal.textContent = state.temperature.toFixed(1);
  });

  // Action Buttons
  btnRefresh.addEventListener("click", () => {
    prevPrice = null;
    remainingSeconds = POLL_INTERVAL_SECONDS;
    updateCountdownUI();
    loadInitialData();
  });
  btnForecast.addEventListener("click", runForecast);

  // Mode Switcher (Live Forecast vs Prediction Comparison)
  if (tabModeLive && tabModeCompare) {
    tabModeLive.addEventListener("click", () => {
      tabModeLive.classList.add("active");
      tabModeCompare.classList.remove("active");
      if (comparePanel) comparePanel.classList.add("hidden");
      state.compareMode = false;
      if (legendOverlayItem) legendOverlayItem.style.display = "none";
      renderCharts();
    });

    tabModeCompare.addEventListener("click", async () => {
      tabModeCompare.classList.add("active");
      tabModeLive.classList.remove("active");
      if (comparePanel) comparePanel.classList.remove("hidden");
      state.compareMode = true;
      if (legendOverlayItem) legendOverlayItem.style.display = state.overlayPrediction ? "inline-flex" : "none";
      await fetchPredictionsList();
      renderCharts();
    });
  }

  // Comparison Select & Toggles
  if (pastPredictionSelect) {
    pastPredictionSelect.addEventListener("change", (e) => selectPrediction(e.target.value));
  }
  if (chkOverlayPrediction) {
    chkOverlayPrediction.addEventListener("change", (e) => {
      state.overlayPrediction = e.target.checked;
      if (legendOverlayItem) legendOverlayItem.style.display = state.overlayPrediction ? "inline-flex" : "none";
      renderCharts();
    });
  }
  if (btnSaveCurrentPred) {
    btnSaveCurrentPred.addEventListener("click", saveCurrentPrediction);
  }

  // Chart Navigation Buttons (Zoom & Live Reset)
  if (btnZoomIn) btnZoomIn.addEventListener("click", () => zoomChart(-1));
  if (btnZoomOut) btnZoomOut.addEventListener("click", () => zoomChart(1));
  if (btnReset) btnReset.addEventListener("click", () => resetChartToLive());

  // Quick Range Selector Buttons (TradingView style at bottom)
  if (quickRangeSelector) {
    quickRangeSelector.querySelectorAll(".range-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        quickRangeSelector.querySelectorAll(".range-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const bars = parseInt(btn.dataset.bars, 10);
        state.visibleBars = bars;
        state.panPixelOffset = 0; // jump to latest
        renderCharts();
      });
    });
  }

  // Interactive Pan / Drag & Zoom Event Listeners (TradingView style)
  mainCanvas.addEventListener("mousedown", handleCanvasMouseDown);
  window.addEventListener("mousemove", handleCanvasMouseMove);
  window.addEventListener("mouseup", handleCanvasMouseUp);
  mainCanvas.addEventListener("wheel", handleCanvasWheel, { passive: false });
  mainCanvas.addEventListener("dblclick", () => resetChartToLive());
  mainCanvas.addEventListener("mouseleave", () => {
    if (!state.isDragging) {
      state.isHovering = false;
      state.hoveredIndex = null;
      chartTooltip.classList.add("hidden");
      renderCharts();
    }
  });
}

function resizeCanvases() {
  const dpr = window.devicePixelRatio || 1;
  const mainRect = chartWrapper.getBoundingClientRect();
  mainCanvas.width = mainRect.width * dpr;
  mainCanvas.height = mainRect.height * dpr;
  mainCtx.scale(dpr, dpr);

  const subRect = subCanvas.parentElement.getBoundingClientRect();
  subCanvas.width = (subRect.width - 24) * dpr;
  subCanvas.height = 110 * dpr;
  subCtx.scale(dpr, dpr);
}

// --------------------------------------------------------------------------
// Real-Time Live Ticker Streaming
// --------------------------------------------------------------------------
function startLiveTickStream() {
  if (countdownTimer) clearInterval(countdownTimer);

  remainingSeconds = POLL_INTERVAL_SECONDS;
  updateCountdownUI();

  countdownTimer = setInterval(() => {
    remainingSeconds--;
    if (remainingSeconds <= 0) {
      remainingSeconds = POLL_INTERVAL_SECONDS;
      fetchTick();
      if (tickCount % 15 === 0) {
        refreshHistoricalBars();
      }
    }
    updateCountdownUI();
    updateActiveCandleTimer();
  }, 1000);
}

function updateCountdownUI() {
  const el = document.getElementById("countdown-label");
  if (el) {
    el.textContent = `${remainingSeconds}s`;
  }
}

async function fetchTick() {
  try {
    const res = await fetch(`/api/tick?expiry=${state.expiry}`);
    if (!res.ok) return;
    const tick = await res.json();
    if (!tick || tick.price === undefined) return;

    applyLiveTick(tick);
  } catch (err) {
    console.warn("Tick fetch hiccup:", err);
  }
}

function applyLiveTick(tick) {
  tickCount++;

  // 1. Visual Price Flash
  if (prevPrice !== null && tick.price !== undefined) {
    if (tick.price > prevPrice) {
      spotPriceEl.classList.remove("flash-down");
      spotPriceEl.classList.add("flash-up");
      setTimeout(() => spotPriceEl.classList.remove("flash-up"), 450);
    } else if (tick.price < prevPrice) {
      spotPriceEl.classList.remove("flash-up");
      spotPriceEl.classList.add("flash-down");
      setTimeout(() => spotPriceEl.classList.remove("flash-down"), 450);
    }
  }
  prevPrice = tick.price;

  // 2. Update Header Price & Change
  spotPriceEl.textContent = formatINR(tick.price);
  const cVal = tick.change_val !== undefined ? tick.change_val : 547.0;
  const cPct = tick.change_pct !== undefined ? tick.change_pct : 0.36;

  changeAbsEl.textContent = formatINR(Math.abs(cVal));
  changePctEl.textContent = `(${cVal >= 0 ? "+" : ""}${cPct.toFixed(2)}%)`;

  if (cVal >= 0) {
    changeWrapperEl.className = "quote-change positive";
    changeWrapperEl.querySelector(".arrow-icon").textContent = "▲";
  } else {
    changeWrapperEl.className = "quote-change negative";
    changeWrapperEl.querySelector(".arrow-icon").textContent = "▼";
  }

  // 3. Update As-of Time
  if (tick.time_formatted) {
    lastSyncEl.textContent = tick.time_formatted;
  }

  // 4. Update Moneycontrol Overview Grid
  if (tick.open_price) ovOpenEl.textContent = `₹ ${formatINR(tick.open_price)}`;
  if (tick.prev_close) ovPrevCloseEl.textContent = `₹ ${formatINR(tick.prev_close)}`;
  if (tick.day_low) ovLowEl.textContent = `₹ ${formatINR(tick.day_low)}`;
  if (tick.day_high) ovHighEl.textContent = `₹ ${formatINR(tick.day_high)}`;

  // Range pin percentage
  if (tick.day_high && tick.day_low && tick.day_high > tick.day_low && rangePinEl) {
    const pct = Math.max(0, Math.min(100, ((tick.price - tick.day_low) / (tick.day_high - tick.day_low)) * 100));
    rangePinEl.style.left = `${pct.toFixed(1)}%`;
  }

  if (tick.avg_price) ovAtpEl.textContent = `₹ ${formatINR(tick.avg_price)}`;

  // Volume & Value formatting
  if (tick.volume !== undefined && tick.traded_val_lacs !== undefined) {
    const volLacs = (tick.volume / 100000.0).toFixed(2);
    const valLacs = (tick.traded_val_lacs / 100000.0).toFixed(2);
    ovVolValEl.textContent = `${volLacs}L / ₹ ${valLacs}L`;
  }

  if (tick.open_interest !== undefined) {
    ovOiEl.textContent = formatINR(tick.open_interest, 0);
  }
  if (tick.oi_change !== undefined && tick.oi_change_pct !== undefined) {
    const chgSign = tick.oi_change >= 0 ? "+" : "";
    ovOiChgEl.textContent = `${chgSign}${tick.oi_change} (${chgSign}${tick.oi_change_pct.toFixed(2)}%)`;
    ovOiChgEl.style.color = tick.oi_change >= 0 ? "var(--bull-green)" : "var(--bear-red)";
  }
  if (tick.oi_buildup) {
    ovOiBuildupEl.textContent = tick.oi_buildup.toUpperCase();
    ovOiBuildupEl.className = `oi-buildup-tag ${tick.oi_buildup.toLowerCase().includes("long") ? "green" : "red"}`;
  }

  if (tick.bid_price && tick.bid_qty) {
    ovBidEl.textContent = `₹ ${formatINR(tick.bid_price)} (${tick.bid_qty})`;
  }
  if (tick.ask_price && tick.ask_qty) {
    ovAskEl.textContent = `₹ ${formatINR(tick.ask_price)} (${tick.ask_qty})`;
  }

  if (tick.lot_size) {
    lotSizeBadgeEl.textContent = `Lot Size: ${tick.lot_size}`;
  }

  // Update active forming candle timer
  updateActiveCandleTimer();

  // 5. Dynamic Real-Time Candlestick Movement on Canvas & Period Rollover
  const activeData = state.forecastData || state.liveData;
  if (activeData) {
    const history = activeData.history || activeData.candles;
    if (history && history.length > 0) {
      const bucketInfo = getActiveBucketInfo(state.interval);
      const lastCandle = history[history.length - 1];
      const lastTimeMs = new Date(lastCandle.time).getTime();

      // Check if real-world time crossed into a new period bucket
      if (!isNaN(lastTimeMs) && lastTimeMs < bucketInfo.bucketDate.getTime() && (bucketInfo.bucketDate.getTime() - lastTimeMs) >= 60000) {
        // Roll over: seal previous candle and spawn new active forming candle immediately
        lastCandle.is_forming = false;
        const newCandle = {
          time: bucketInfo.bucketIso,
          time_formatted: formatCustomDateIST(bucketInfo.bucketDate),
          period_end: bucketInfo.closeDate.toISOString(),
          period_end_formatted: formatCustomTimeIST(bucketInfo.closeDate),
          is_forming: true,
          open: lastCandle.close,
          high: Math.max(lastCandle.close, tick.price),
          low: Math.min(lastCandle.close, tick.price),
          close: tick.price,
          volume: 10,
        };
        history.push(newCandle);
      } else {
        lastCandle.close = tick.price;
        if (tick.price > lastCandle.high) lastCandle.high = tick.price;
        if (tick.price < lastCandle.low) lastCandle.low = tick.price;
        lastCandle.is_forming = true;
      }

      if (state.forecastData) {
        state.forecastData.current_price = tick.price;
        // Check for 90% Confidence Band Invalidation / Volatility Breakout
        checkAutoReevaluation(tick);
      }
      renderCharts();
    }
  }
}

// --------------------------------------------------------------------------
// Automated Re-evaluation Watchdog (Event-Driven Trigger)
// --------------------------------------------------------------------------
let lastAutoEvalTime = 0;
let isAutoEvaluating = false;

function showAutoReevalToast(title, message) {
  const toast = document.getElementById("auto-reeval-toast");
  const titleEl = document.getElementById("toast-title");
  const msgEl = document.getElementById("toast-message");
  if (!toast) return;

  if (titleEl) titleEl.textContent = title;
  if (msgEl) msgEl.textContent = message;

  toast.classList.add("visible");
  setTimeout(() => {
    toast.classList.remove("visible");
  }, 6500);
}

function checkAutoReevaluation(tick) {
  if (isAutoEvaluating || !state.forecastData || !state.forecastData.forecast) return;
  const fCandles = state.forecastData.forecast;
  if (fCandles.length === 0) return;

  const now = Date.now();
  // 30-second cooldown so it doesn't spam inference on rapid volatility ticks
  if (now - lastAutoEvalTime < 30000) return;

  const lastF = fCandles[fCandles.length - 1];
  const p = tick.price;

  if (p > lastF.upper_95) {
    triggerAutoReevaluation(`Price pierced 90% Upper Band (₹${formatINR(lastF.upper_95, 0)}). Volatility breakout detected.`);
  } else if (p < lastF.lower_5) {
    triggerAutoReevaluation(`Price broke 90% Lower Band (₹${formatINR(lastF.lower_5, 0)}). Downward expansion detected.`);
  }
}

async function triggerAutoReevaluation(reason) {
  if (isAutoEvaluating) return;
  const now = Date.now();
  if (now - lastAutoEvalTime < 25000) return;

  isAutoEvaluating = true;
  lastAutoEvalTime = now;

  showAutoReevalToast("⚡ Auto-Reevaluation Triggered", reason);
  console.log("[Auto-Watchdog]", reason);

  try {
    await runForecast();
  } catch (err) {
    console.warn("Auto-reevaluation run failed:", err);
  } finally {
    isAutoEvaluating = false;
  }
}

// --------------------------------------------------------------------------
// Real-Time Bucket Timing Helpers
// --------------------------------------------------------------------------
function getActiveBucketInfo(interval) {
  const now = new Date();
  const bucketDate = new Date(now);
  const closeDate = new Date(now);

  if (interval === "15m") {
    const min = now.getMinutes();
    const bMin = Math.floor(min / 15) * 15;
    bucketDate.setMinutes(bMin, 0, 0);
    closeDate.setMinutes(bMin + 15, 0, 0);
  } else if (interval === "1h") {
    bucketDate.setMinutes(0, 0, 0);
    closeDate.setHours(now.getHours() + 1, 0, 0, 0);
  } else if (interval === "4h") {
    const hr = Math.floor(now.getHours() / 4) * 4;
    bucketDate.setHours(hr, 0, 0, 0);
    closeDate.setHours(hr + 4, 0, 0, 0);
  } else {
    // 1d
    bucketDate.setHours(9, 30, 0, 0);
    closeDate.setDate(now.getDate() + 1);
    closeDate.setHours(9, 30, 0, 0);
  }

  const msRemaining = Math.max(0, closeDate.getTime() - now.getTime());
  const totalSec = Math.floor(msRemaining / 1000);
  const mRem = Math.floor(totalSec / 60);
  const sRem = totalSec % 60;

  return {
    bucketDate,
    closeDate,
    bucketIso: bucketDate.toISOString(),
    minRemaining: mRem,
    secRemaining: sRem,
    formattedRemaining: `${String(mRem).padStart(2, "0")}m ${String(sRem).padStart(2, "0")}s`,
  };
}

function updateActiveCandleTimer() {
  if (!formingTimerTextEl) return;
  const info = getActiveBucketInfo(state.interval);
  formingTimerTextEl.textContent = `Active ${state.interval.toUpperCase()} Candle: Closes in ${info.formattedRemaining}`;
}

function formatCustomDateIST(d) {
  return d.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
  }) + " " + d.toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  }) + " IST";
}

function formatCustomTimeIST(d) {
  return d.toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  }) + " IST";
}

// --------------------------------------------------------------------------
// API Calls & Initial Load
// --------------------------------------------------------------------------
async function loadInitialData() {
  chartStatusEl.textContent = `Connecting to Moneycontrol MCX Goldm (${state.expiry}) feed...`;
  btnRefresh.disabled = true;

  try {
    const res = await fetch(`/api/live?interval=${state.interval}&limit=400&expiry=${state.expiry}`);
    const data = await res.json();
    state.liveData = data;

    // Update Expiry Dropdown options if provided
    if (data.expiries && data.expiries.length > 0) {
      populateExpiryDropdown(data.expiries);
    }

    // Apply Live Ticker Data from payload
    if (data.live_ticker) {
      applyLiveTick(data.live_ticker);
    }

    // Apply Technical Stance & Pivots
    updateTechnicalsAndPivots(data);

    await runForecast();
    await fetchPredictionsList();
  } catch (err) {
    console.error("Live fetch error:", err);
    chartStatusEl.textContent = "MCX feed connection error.";
  } finally {
    btnRefresh.disabled = false;
  }
}

async function refreshHistoricalBars() {
  if (state.isHovering || state.isDragging || Math.abs(state.panPixelOffset || 0) > 10) return; // Do not interrupt active panning or reviewing past history
  try {
    const res = await fetch(`/api/live?interval=${state.interval}&limit=400&expiry=${state.expiry}`);
    if (!res.ok) return;
    const data = await res.json();
    state.liveData = data;
    if (data.live_ticker) {
      applyLiveTick(data.live_ticker);
    }
    updateTechnicalsAndPivots(data);
  } catch (err) {
    console.warn("Background klines refresh error:", err);
  }
}

async function runForecast() {
  btnForecast.disabled = true;
  chartStatusEl.textContent = `Running Kronos AI Engine on Goldm (${state.horizon} steps, ${state.samples} Monte Carlo rollouts)...`;

  try {
    const url = `/api/forecast?interval=${state.interval}&lookback=350&horizon=${state.horizon}&samples=${state.samples}&temperature=${state.temperature}&expiry=${state.expiry}`;
    const res = await fetch(url);
    const data = await res.json();
    state.forecastData = data;

    updateUI();
    chartStatusEl.textContent = `Kronos Goldm Projections Ready (T+${state.horizon} Ahead, 90% Confidence)`;
    await fetchPredictionsList();
  } catch (err) {
    console.error("Forecast error:", err);
    chartStatusEl.textContent = "Forecast computation error.";
  } finally {
    btnForecast.disabled = false;
  }
}

function populateExpiryDropdown(expiries) {
  const currentVal = state.expiry;
  expirySelectEl.innerHTML = "";
  expiries.forEach(([val, label]) => {
    const opt = document.createElement("option");
    opt.value = val;
    opt.textContent = label;
    if (val === currentVal) opt.selected = true;
    expirySelectEl.appendChild(opt);
  });
}

function updateTechnicalsAndPivots(data) {
  if (data.tech_stance) {
    techStanceBadgeEl.textContent = data.tech_stance;
    techStanceBadgeEl.className = `tech-stance-pill ${data.tech_stance.includes("BULLISH") ? "bullish" : "bearish"}`;
  }

  if (data.pivots) {
    const p = data.pivots;
    pivotS3El.textContent = `₹ ${formatINR(p.s3, 0)}`;
    pivotS2El.textContent = `₹ ${formatINR(p.s2, 0)}`;
    pivotS1El.textContent = `₹ ${formatINR(p.s1, 0)}`;
    pivotPEl.textContent = `₹ ${formatINR(p.pivot, 0)}`;
    pivotR1El.textContent = `₹ ${formatINR(p.r1, 0)}`;
    pivotR2El.textContent = `₹ ${formatINR(p.r2, 0)}`;
    pivotR3El.textContent = `₹ ${formatINR(p.r3, 0)}`;
  }
}

// --------------------------------------------------------------------------
// UI & State Renderers
// --------------------------------------------------------------------------
function updateUI() {
  const data = state.forecastData || state.liveData;
  if (!data) return;

  if (data.live_ticker) {
    applyLiveTick(data.live_ticker);
  }
  updateTechnicalsAndPivots(data);

  // Alpha Signal Card
  if (state.forecastData) {
    const fData = state.forecastData;
    const sigBadge = document.getElementById("signal-badge");
    const sigTitle = document.getElementById("signal-title");
    const targetPriceEl = document.getElementById("target-price");
    const expRetEl = document.getElementById("expected-return");
    const bandLowEl = document.getElementById("band-low");
    const bandHighEl = document.getElementById("band-high");

    const modelBadge = document.getElementById("model-badge-info");
    if (modelBadge && fData.ai_model) {
      modelBadge.textContent = fData.ai_model.toUpperCase();
    }

    sigTitle.textContent = fData.signal || "BUY / LONG BIAS";
    const ret = fData.expected_return_pct;
    expRetEl.textContent = `${ret >= 0 ? "+" : ""}${ret.toFixed(2)}%`;
    expRetEl.className = `stat-number ${ret >= 0 ? "positive" : "negative"}`;

    if (ret > 0.2) {
      sigBadge.textContent = "BULLISH BIAS";
      sigBadge.className = "status-badge bullish";
    } else if (ret < -0.2) {
      sigBadge.textContent = "BEARISH BIAS";
      sigBadge.className = "status-badge bearish";
    } else {
      sigBadge.textContent = "NEUTRAL BIAS";
      sigBadge.className = "status-badge neutral";
    }

    targetPriceEl.textContent = formatINR(fData.target_price);

    const lastForecast = fData.forecast[fData.forecast.length - 1];
    bandLowEl.textContent = `₹ ${formatINR(lastForecast.lower_5, 0)}`;
    bandHighEl.textContent = `₹ ${formatINR(lastForecast.upper_95, 0)}`;

    // Market Regime & Indicators
    document.getElementById("market-regime-text").textContent = fData.oi_analysis || fData.indicators.market_regime;

    const ind = fData.indicators;
    document.getElementById("ind-ema20").textContent = `₹ ${formatINR(ind.ema_20)}`;
    document.getElementById("ind-ema50").textContent = `₹ ${formatINR(ind.ema_50)}`;
    document.getElementById("ind-rsi").textContent = ind.rsi_14.toFixed(1);
    document.getElementById("current-rsi-val").textContent = `RSI: ${ind.rsi_14.toFixed(1)}`;
    document.getElementById("ind-atr").textContent = `₹ ${formatINR(ind.atr_14)}`;
    document.getElementById("ind-vol").textContent = `${ind.realized_volatility_pct.toFixed(2)}%`;
    document.getElementById("ind-bb-up").textContent = `₹ ${formatINR(ind.bollinger_upper)}`;
    document.getElementById("ind-bb-low").textContent = `₹ ${formatINR(ind.bollinger_lower)}`;
    document.getElementById("ind-macd").textContent = `${ind.macd_hist >= 0 ? "+" : ""}${ind.macd_hist.toFixed(2)}`;

    // Update Synthesized Confluence Card (No manual calculations needed)
    updateConfluenceCard(fData, ind, data.live_ticker);
  }

  // Update Forecast Table
  updateForecastTable();

  // Render Canvas Charts
  renderCharts();
}

function updateConfluenceCard(fData, ind, liveTicker) {
  if (!fData || !ind) return;

  const ret = fData.expected_return_pct;
  const currPrice = fData.current_price || (liveTicker ? liveTicker.price : ind.ema_20);
  const oiBuildup = (liveTicker && liveTicker.oi_buildup) ? liveTicker.oi_buildup.toLowerCase() : "long buildup";

  // Factor 1: AI Direction
  let aiDirection = "NEUTRAL";
  if (ret > 0.15) aiDirection = "BULLISH";
  else if (ret < -0.15) aiDirection = "BEARISH";

  // Factor 2: Order Flow (OI)
  let oiDirection = "NEUTRAL";
  if (oiBuildup.includes("long buildup") || oiBuildup.includes("short covering")) {
    oiDirection = "BULLISH";
  } else if (oiBuildup.includes("short buildup") || oiBuildup.includes("long unwinding")) {
    oiDirection = "BEARISH";
  }

  // Factor 3: Technical Trend
  let trendDirection = "NEUTRAL";
  if (currPrice > ind.ema_20 && ind.ema_20 > ind.ema_50) {
    trendDirection = "BULLISH";
  } else if (currPrice < ind.ema_20 && ind.ema_20 < ind.ema_50) {
    trendDirection = "BEARISH";
  } else if (currPrice > ind.ema_50) {
    trendDirection = "BULLISH";
  } else {
    trendDirection = "BEARISH";
  }

  let bullVotes = (aiDirection === "BULLISH" ? 1 : 0) + (oiDirection === "BULLISH" ? 1 : 0) + (trendDirection === "BULLISH" ? 1 : 0);
  let bearVotes = (aiDirection === "BEARISH" ? 1 : 0) + (oiDirection === "BEARISH" ? 1 : 0) + (trendDirection === "BEARISH" ? 1 : 0);

  const headlineEl = document.getElementById("verdict-action-title");
  const badgeEl = document.getElementById("verdict-confluence-badge");
  const reasonEl = document.getElementById("verdict-reason-text");
  const actionTagEl = document.getElementById("v-action-tag");
  const stopLossEl = document.getElementById("v-stop-loss");
  const targetZoneEl = document.getElementById("v-target-zone");

  const chkKronosVal = document.getElementById("chk-kronos-val");
  const chkOiVal = document.getElementById("chk-oi-val");
  const chkTrendVal = document.getElementById("chk-trend-val");

  if (chkKronosVal) {
    chkKronosVal.textContent = `${aiDirection} (${ret >= 0 ? "+" : ""}${ret.toFixed(2)}%)`;
    chkKronosVal.className = `chk-status ${aiDirection === "BULLISH" ? "pass" : (aiDirection === "BEARISH" ? "fail" : "warn")}`;
  }
  if (chkOiVal) {
    chkOiVal.textContent = (liveTicker && liveTicker.oi_buildup) ? liveTicker.oi_buildup.toUpperCase() : "LIVE OI";
    chkOiVal.className = `chk-status ${oiDirection === "BULLISH" ? "pass" : (oiDirection === "BEARISH" ? "fail" : "warn")}`;
  }
  if (chkTrendVal) {
    chkTrendVal.textContent = currPrice >= ind.ema_50 ? "ABOVE 50 EMA" : "BELOW 50 EMA";
    chkTrendVal.className = `chk-status ${trendDirection === "BULLISH" ? "pass" : "fail"}`;
  }

  const lastForecast = fData.forecast && fData.forecast.length > 0 ? fData.forecast[fData.forecast.length - 1] : null;
  const stopLoss = lastForecast ? (aiDirection === "BULLISH" ? lastForecast.lower_5 : lastForecast.upper_95) : currPrice;
  const targetZone = fData.target_price;

  if (stopLossEl) stopLossEl.textContent = `₹ ${formatINR(stopLoss, 0)}`;
  if (targetZoneEl) targetZoneEl.textContent = `₹ ${formatINR(targetZone, 0)}`;

  if (bullVotes === 3) {
    if (headlineEl) { headlineEl.textContent = "STRONG BUY / LONG (CONFIRMED)"; headlineEl.className = "verdict-headline buy"; }
    if (badgeEl) { badgeEl.textContent = "3 / 3 FULL ALIGNMENT"; badgeEl.className = "status-badge bullish"; }
    if (actionTagEl) { actionTagEl.textContent = "ENTER LONG"; actionTagEl.className = "v-val buy"; }
    if (reasonEl) reasonEl.textContent = "Kronos AI, MCX Order Flow (Long Buildup), and EMAs are in 100% bullish confluence. Favorable statistical edge for upside.";
  } else if (bearVotes === 3) {
    if (headlineEl) { headlineEl.textContent = "STRONG SELL / SHORT (CONFIRMED)"; headlineEl.className = "verdict-headline sell"; }
    if (badgeEl) { badgeEl.textContent = "3 / 3 FULL ALIGNMENT"; badgeEl.className = "status-badge bearish"; }
    if (actionTagEl) { actionTagEl.textContent = "ENTER SHORT"; actionTagEl.className = "v-val sell"; }
    if (reasonEl) reasonEl.textContent = "Kronos AI, MCX Order Flow (Short Buildup/Unwinding), and EMAs are in 100% bearish confluence. Downside breakdown favored.";
  } else {
    // Conflict / Divergence
    if (headlineEl) { headlineEl.textContent = "WAIT / NO TRADE (CONFLICT)"; headlineEl.className = "verdict-headline wait"; }
    if (badgeEl) { badgeEl.textContent = `${Math.max(bullVotes, bearVotes)} / 3 DIVERGENT`; badgeEl.className = "status-badge neutral"; }
    if (actionTagEl) { actionTagEl.textContent = "STAY IN CASH"; actionTagEl.className = "v-val wait"; }

    let conflictDetail = "";
    if (aiDirection === "BEARISH" && oiDirection === "BULLISH") {
      conflictDetail = "Kronos-base predicts Downward, but MCX Order Flow has Long Buildup. Order flow conflicts with neural forecast.";
    } else if (aiDirection === "BULLISH" && trendDirection === "BEARISH") {
      conflictDetail = "Kronos-base predicts Upward, but price is below the 50 EMA baseline. Counter-trend risk is elevated.";
    } else {
      conflictDetail = "Mixed signals across AI momentum and Indian domestic order flow. High probability of chop or false breakdown.";
    }
    if (reasonEl) reasonEl.textContent = `${conflictDetail} Recommended: Protect capital and wait for 3/3 confirmation.`;
  }
}

function updateForecastTable() {
  const tbody = document.getElementById("forecast-table-body");
  if (!state.forecastData || !state.forecastData.forecast) return;

  tbody.innerHTML = "";
  const fCandles = state.forecastData.forecast;
  const currPrice = state.forecastData.current_price;

  fCandles.forEach((c, idx) => {
    const diffPct = ((c.close - currPrice) / currPrice) * 100.0;
    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td><strong>T+${idx + 1}</strong></td>
      <td>${c.time_formatted || c.time}</td>
      <td><span class="status-badge ${diffPct >= 0 ? "bullish" : "bearish"}">Goldm Future</span></td>
      <td>₹ ${formatINR(c.open)}</td>
      <td>₹ ${formatINR(c.high)}</td>
      <td>₹ ${formatINR(c.low)}</td>
      <td style="color: ${diffPct >= 0 ? "var(--bull-green)" : "var(--bear-red)"}; font-weight: 700;">₹ ${formatINR(c.close)}</td>
      <td style="color: ${diffPct >= 0 ? "var(--bull-green)" : "var(--bear-red)"}; font-weight: 600;">${diffPct >= 0 ? "+" : ""}${diffPct.toFixed(2)}%</td>
      <td>₹ ${formatINR(c.lower_5, 0)} - ₹ ${formatINR(c.upper_95, 0)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// --------------------------------------------------------------------------
// Prediction vs Reality Comparison Engine
// --------------------------------------------------------------------------
async function fetchPredictionsList() {
  try {
    const res = await fetch(`/api/predictions?interval=${state.interval}&expiry=${state.expiry}`);
    if (!res.ok) return;
    const data = await res.json();
    state.predictionsList = data.predictions || [];

    if (compareHistoryCountEl) {
      compareHistoryCountEl.textContent = state.predictionsList.length;
    }

    populatePastPredictionsDropdown();
    if (state.predictionsList.length > 0 && !state.selectedPrediction) {
      selectPrediction(state.predictionsList[0].id);
    }
  } catch (err) {
    console.warn("Error fetching predictions list:", err);
  }
}

function populatePastPredictionsDropdown() {
  if (!pastPredictionSelect) return;
  pastPredictionSelect.innerHTML = "";
  if (state.predictionsList.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "No predictions recorded yet";
    pastPredictionSelect.appendChild(opt);
    return;
  }

  state.predictionsList.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.id;
    const accText = p.metrics && p.metrics.evaluated_steps > 0
      ? ` [${p.metrics.directional_accuracy_pct}% Acc, ₹${p.metrics.mae_inr} MAE]`
      : " [Pending]";
    opt.textContent = `${p.created_at_formatted} (${p.interval}, T+${p.horizon})${accText}`;
    if (state.selectedPrediction && state.selectedPrediction.id === p.id) {
      opt.selected = true;
    }
    pastPredictionSelect.appendChild(opt);
  });
}

function selectPrediction(id) {
  const pred = state.predictionsList.find((p) => p.id === id);
  if (!pred) return;
  state.selectedPrediction = pred;

  // Update Scorecard
  const m = pred.metrics || {};
  const dirEl = document.getElementById("metric-dir-acc");
  const dirSub = document.getElementById("metric-dir-sub");
  const maeEl = document.getElementById("metric-mae-inr");
  const maePct = document.getElementById("metric-mae-pct");
  const covEl = document.getElementById("metric-band-cov");
  const statusBadge = document.getElementById("metric-status-badge");
  const targetInfo = document.getElementById("metric-target-info");

  if (dirEl) {
    dirEl.textContent = m.evaluated_steps > 0 ? `${m.directional_accuracy_pct}%` : "--%";
    dirEl.className = `scorecard-val ${m.directional_accuracy_pct >= 70 ? "positive" : (m.directional_accuracy_pct >= 50 ? "neutral" : "negative")}`;
  }
  if (dirSub) {
    dirSub.textContent = `${Math.round((m.directional_accuracy_pct * m.evaluated_steps) / 100)} of ${m.evaluated_steps} steps correct`;
  }
  if (maeEl) maeEl.textContent = m.evaluated_steps > 0 ? `±₹ ${formatINR(m.mae_inr, 2)}` : "₹ --";
  if (maePct) maePct.textContent = m.evaluated_steps > 0 ? `${m.mae_pct.toFixed(3)}% avg error` : "0.00% avg error";
  if (covEl) {
    covEl.textContent = m.evaluated_steps > 0 ? `${m.band_coverage_pct}%` : "--%";
    covEl.className = `scorecard-val ${m.band_coverage_pct >= 80 ? "positive" : "negative"}`;
  }
  if (statusBadge) {
    statusBadge.textContent = pred.status || "IN PROGRESS";
    statusBadge.className = `status-badge ${pred.is_complete ? "bullish" : "neutral"}`;
  }
  if (targetInfo) {
    targetInfo.textContent = `Target: ₹ ${formatINR(pred.target_price)} (${pred.signal})`;
  }

  // Update Table
  updateComparisonTable(pred);

  if (legendOverlayItem) {
    legendOverlayItem.style.display = state.overlayPrediction ? "inline-flex" : "none";
  }

  renderCharts();
}

function updateComparisonTable(pred) {
  const tbody = document.getElementById("compare-table-body");
  if (!tbody || !pred.comparison_steps) return;
  tbody.innerHTML = "";

  pred.comparison_steps.forEach((step) => {
    const tr = document.createElement("tr");
    const isEvaluated = step.status === "EVALUATED";

    const dirBadge = isEvaluated
      ? (step.dir_correct ? `<span style="color:var(--bull-green); font-weight:700;">Matched ✔</span>` : `<span style="color:var(--bear-red); font-weight:700;">Diverged ✖</span>`)
      : `<span style="color:var(--text-muted);">Pending</span>`;

    const bandBadge = isEvaluated
      ? (step.in_band ? `<span style="color:#00c087;">Inside Band</span>` : `<span style="color:#f43f5e;">Outside Band</span>`)
      : `<span style="color:var(--text-muted);">Pending</span>`;

    const errInrText = isEvaluated
      ? `<span style="color:${step.error_inr >= 0 ? "var(--bull-green)" : "var(--bear-red)"}; font-weight:600;">${step.error_inr >= 0 ? "+" : ""}₹ ${formatINR(step.error_inr, 2)}</span>`
      : `--`;

    const errPctText = isEvaluated ? `${step.pct_error.toFixed(3)}%` : `--`;
    const actualText = isEvaluated ? `<strong>₹ ${formatINR(step.actual_close)}</strong>` : `<em style="color:var(--text-muted);">Awaiting Candle</em>`;

    tr.innerHTML = `
      <td><strong>T+${step.step}</strong></td>
      <td>${step.time_formatted || step.time}</td>
      <td style="color:#fbbf24; font-weight:600;">₹ ${formatINR(step.predicted_close)}</td>
      <td>${actualText}</td>
      <td>${errInrText}</td>
      <td>${errPctText}</td>
      <td>${dirBadge}</td>
      <td>${bandBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function saveCurrentPrediction() {
  if (!state.forecastData || !state.forecastData.forecast) {
    alert("Please run a Kronos forecast first to save a snapshot.");
    return;
  }
  const fData = state.forecastData;
  const payload = {
    id: `pred_${Date.now()}`,
    created_at: new Date().toISOString(),
    created_at_formatted: new Date().toLocaleString("en-IN", { timeZone: "Asia/Kolkata" }),
    interval: state.interval,
    horizon: state.horizon,
    base_price: fData.current_price,
    target_price: fData.target_price,
    expected_return_pct: fData.expected_return_pct,
    signal: fData.signal,
    forecast: fData.forecast,
  };

  try {
    const res = await fetch("/api/predictions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      await fetchPredictionsList();
      alert("Forecast snapshot saved successfully! You can now track its ongoing accuracy.");
    }
  } catch (err) {
    console.error("Save error:", err);
  }
}

// --------------------------------------------------------------------------
// High-Precision Candlestick & Forecast Canvas Renderer (TradingView Style)
// --------------------------------------------------------------------------
function renderCharts() {
  const data = state.forecastData || state.liveData;
  if (!data) return;

  const history = data.history || data.candles || [];
  const forecast = data.forecast || [];
  const allBars = [...history, ...forecast];
  if (allBars.length === 0) return;

  const width = chartWrapper.clientWidth;
  const height = chartWrapper.clientHeight;

  mainCtx.clearRect(0, 0, width, height);

  // Clamp visibleBars
  state.visibleBars = Math.max(15, Math.min(allBars.length, state.visibleBars));

  const plotW = width - 95;
  const plotH = height - 26;
  const barSpacing = plotW / state.visibleBars;
  const barWidth = Math.max(barSpacing * 0.72, 2.5);
  const DEFAULT_RIGHT_OFFSET_BARS = 5;

  if (state.panPixelOffset === undefined) state.panPixelOffset = 0;

  // Clamp panPixelOffset: allows panning into history and small future margin
  const minPan = -(DEFAULT_RIGHT_OFFSET_BARS + 20) * barSpacing;
  const maxPan = Math.max(0, (allBars.length - 4)) * barSpacing;
  state.panPixelOffset = Math.max(minPan, Math.min(maxPan, state.panPixelOffset));

  // Pure TradingView world coordinate transform for bar i (0 <= i < allBars.length)
  const getBarX = (i) => {
    return plotW - (DEFAULT_RIGHT_OFFSET_BARS + 0.5 + (allBars.length - 1 - i)) * barSpacing + state.panPixelOffset;
  };

  // Determine visible index range [startIdx, endIdx] based on screen coordinates
  const leftI = Math.floor(allBars.length - 1 + DEFAULT_RIGHT_OFFSET_BARS + 0.5 - (plotW + state.panPixelOffset + barSpacing) / barSpacing);
  const rightI = Math.ceil(allBars.length - 1 + DEFAULT_RIGHT_OFFSET_BARS + 0.5 - (state.panPixelOffset - barSpacing) / barSpacing);

  const startIdx = Math.max(0, Math.min(allBars.length - 1, leftI));
  const endIdx = Math.max(startIdx + 1, Math.min(allBars.length, rightI + 1));
  const visibleBarsList = allBars.slice(startIdx, endIdx);
  if (visibleBarsList.length === 0) return;

  // Update visible range info text
  if (visibleRangeInfoEl) {
    if (Math.abs(state.panPixelOffset) < 5) {
      visibleRangeInfoEl.textContent = `Viewing latest ${visibleBarsList.length} candles (Live)`;
    } else {
      const firstT = visibleBarsList[0].time_formatted || visibleBarsList[0].time;
      const lastT = visibleBarsList[visibleBarsList.length - 1].time_formatted || visibleBarsList[visibleBarsList.length - 1].time;
      visibleRangeInfoEl.textContent = `Historical View (${firstT} → ${lastT})`;
    }
  }

  // Dynamic Price Extents strictly for visible candles (TradingView auto-scaling)
  let minPrice = Infinity;
  let maxPrice = -Infinity;

  visibleBarsList.forEach((b) => {
    const l = b.lower_5 !== undefined ? b.lower_5 : b.low;
    const h = b.upper_95 !== undefined ? b.upper_95 : b.high;
    if (l < minPrice) minPrice = l;
    if (h > maxPrice) maxPrice = h;
  });

  // Include overlay prediction prices if active
  if (state.selectedPrediction && state.overlayPrediction) {
    const predSteps = state.selectedPrediction.comparison_steps || state.selectedPrediction.forecast || [];
    predSteps.forEach((s) => {
      const pTime = s.time;
      const isVis = visibleBarsList.some((b) => b.time === pTime || (b.time && pTime && b.time.slice(0, 16) === pTime.slice(0, 16)));
      if (isVis) {
        const pClose = Number(s.predicted_close || s.close || 0);
        const pLow = Number(s.lower_5 || pClose);
        const pHigh = Number(s.upper_95 || pClose);
        if (pLow > 0 && pLow < minPrice) minPrice = pLow;
        if (pHigh > 0 && pHigh > maxPrice) maxPrice = pHigh;
      }
    });
  }

  const pricePadding = (maxPrice - minPrice) * 0.08 || 100;
  minPrice -= pricePadding;
  maxPrice += pricePadding;

  const getY = (val) => plotH - ((val - minPrice) / (maxPrice - minPrice)) * plotH;

  // 1. Draw Grid Lines & Price Labels
  mainCtx.strokeStyle = "#131f36";
  mainCtx.lineWidth = 1;
  const gridRows = 6;
  mainCtx.font = "10px 'JetBrains Mono', monospace";
  mainCtx.fillStyle = "#64748b";
  mainCtx.textAlign = "left";

  for (let i = 0; i <= gridRows; i++) {
    const pY = (plotH / gridRows) * i;
    mainCtx.beginPath();
    mainCtx.moveTo(0, pY);
    mainCtx.lineTo(plotW, pY);
    mainCtx.stroke();

    const pVal = maxPrice - (i / gridRows) * (maxPrice - minPrice);
    mainCtx.fillText(`₹ ${formatINR(pVal, 0)}`, plotW + 8, pY + 3);
  }

  // 2. Draw 90% Confidence Band Fill for Forecast Zone if visible
  if (forecast.length > 0) {
    const fStartIdx = history.length;
    const fEndIdx = allBars.length;
    const fVisibleStart = Math.max(fStartIdx, startIdx);
    const fVisibleEnd = Math.min(fEndIdx, endIdx);

    if (fVisibleStart < fVisibleEnd) {
      mainCtx.beginPath();
      // Upper boundary
      for (let i = fVisibleStart; i < fVisibleEnd; i++) {
        const bar = allBars[i];
        if (!bar.is_predicted) continue;
        const x = getBarX(i);
        const yUpper = getY(bar.upper_95);
        if (i === fVisibleStart) mainCtx.moveTo(x, yUpper);
        else mainCtx.lineTo(x, yUpper);
      }
      // Lower boundary back
      for (let i = fVisibleEnd - 1; i >= fVisibleStart; i--) {
        const bar = allBars[i];
        if (!bar.is_predicted) continue;
        const x = getBarX(i);
        const yLower = getY(bar.lower_5);
        mainCtx.lineTo(x, yLower);
      }
      mainCtx.closePath();
      mainCtx.fillStyle = "rgba(6, 182, 212, 0.12)";
      mainCtx.fill();

      // Dotted boundary line
      mainCtx.setLineDash([3, 3]);
      mainCtx.strokeStyle = "rgba(6, 182, 212, 0.4)";
      mainCtx.stroke();
      mainCtx.setLineDash([]);
    }
  }

  // 3. Draw Vertical Demarcation Line (Separating History and Forecast) if in view
  const histBoundaryIdx = history.length - 1;
  if (forecast.length > 0 && histBoundaryIdx >= 0 && histBoundaryIdx < allBars.length) {
    const demX = getBarX(histBoundaryIdx) + barSpacing / 2;
    if (demX >= 0 && demX <= plotW) {
      mainCtx.beginPath();
      mainCtx.moveTo(demX, 0);
      mainCtx.lineTo(demX, plotH);
      mainCtx.setLineDash([4, 4]);
      mainCtx.strokeStyle = "#38bdf8";
      mainCtx.lineWidth = 1.5;
      mainCtx.stroke();
      mainCtx.setLineDash([]);

      mainCtx.fillStyle = "#38bdf8";
      mainCtx.font = "bold 9px 'Inter', sans-serif";
      mainCtx.fillText("LIVE REALITY ◀", demX - 85, 16);
      mainCtx.fillText("▶ KRONOS FORECAST", demX + 8, 16);
    }
  }

  // 4. Render Candlesticks
  for (let i = startIdx; i < endIdx; i++) {
    const bar = allBars[i];
    const x = getBarX(i);
    if (x < -barSpacing || x > plotW + barSpacing) continue;

    const isForecast = bar.is_predicted;
    const yOpen = getY(bar.open);
    const yClose = getY(bar.close);
    const yHigh = getY(bar.high);
    const yLow = getY(bar.low);

    const isBull = bar.close >= bar.open;

    if (isForecast) {
      mainCtx.strokeStyle = isBull ? "#22d3ee" : "#f43f5e";
      mainCtx.fillStyle = isBull ? "rgba(34, 211, 238, 0.75)" : "rgba(244, 63, 94, 0.75)";
    } else {
      mainCtx.strokeStyle = isBull ? "var(--bull-green)" : "var(--bear-red)";
      mainCtx.fillStyle = isBull ? "#00c087" : "#f43f5e";
    }

    // Candle Wick
    mainCtx.lineWidth = 1.5;
    mainCtx.beginPath();
    mainCtx.moveTo(x, yHigh);
    mainCtx.lineTo(x, yLow);
    mainCtx.stroke();

    // Candle Body
    const bodyY = Math.min(yOpen, yClose);
    const bodyH = Math.max(Math.abs(yClose - yOpen), 1.5);
    mainCtx.fillRect(x - barWidth / 2, bodyY, barWidth, bodyH);
    mainCtx.strokeRect(x - barWidth / 2, bodyY, barWidth, bodyH);
  }

  // 4b. Render Past Prediction Comparison Overlay (Amber/Golden Trajectory)
  if (state.selectedPrediction && state.overlayPrediction) {
    const pred = state.selectedPrediction;
    const predSteps = pred.comparison_steps || pred.forecast || [];

    // Map timestamps to global bar indices
    const timeToIndexMap = new Map();
    allBars.forEach((bar, idx) => {
      if (bar.time) {
        timeToIndexMap.set(bar.time, idx);
        timeToIndexMap.set(bar.time.slice(0, 16), idx);
      }
    });

    const overlayPoints = [];

    // Origin point if present in view
    if (pred.base_price && pred.created_at) {
      const origIdx = timeToIndexMap.get(pred.created_at) || timeToIndexMap.get(pred.created_at.slice(0, 16));
      if (origIdx !== undefined) {
        const ox = getBarX(origIdx);
        overlayPoints.push({
          x: ox,
          y: getY(pred.base_price),
          yLow: getY(pred.base_price),
          yHigh: getY(pred.base_price),
          isOrigin: true,
          label: `Origin: ₹ ${formatINR(pred.base_price)}`,
        });
      }
    }

    predSteps.forEach((s, idx) => {
      const pTime = s.time;
      if (!pTime) return;
      const bIdx = timeToIndexMap.get(pTime) || timeToIndexMap.get(pTime.slice(0, 16));
      if (bIdx !== undefined) {
        const px = getBarX(bIdx);
        const pClose = Number(s.predicted_close || s.close || 0);
        const pLow = Number(s.lower_5 || (pClose * 0.998));
        const pHigh = Number(s.upper_95 || (pClose * 1.002));
        overlayPoints.push({
          x: px,
          y: getY(pClose),
          yLow: getY(pLow),
          yHigh: getY(pHigh),
          val: pClose,
          isEvaluated: s.status === "EVALUATED",
          step: s.step || (idx + 1),
        });
      }
    });

    if (overlayPoints.length > 0) {
      // 1. Draw Confidence Cloud Fill
      if (overlayPoints.length > 1) {
        mainCtx.save();
        mainCtx.beginPath();
        overlayPoints.forEach((pt, i) => {
          if (i === 0) mainCtx.moveTo(pt.x, pt.yHigh);
          else mainCtx.lineTo(pt.x, pt.yHigh);
        });
        for (let i = overlayPoints.length - 1; i >= 0; i--) {
          mainCtx.lineTo(overlayPoints[i].x, overlayPoints[i].yLow);
        }
        mainCtx.closePath();
        mainCtx.fillStyle = "rgba(245, 158, 11, 0.14)";
        mainCtx.fill();

        mainCtx.setLineDash([2, 3]);
        mainCtx.strokeStyle = "rgba(245, 158, 11, 0.45)";
        mainCtx.lineWidth = 1;
        mainCtx.stroke();
        mainCtx.setLineDash([]);
        mainCtx.restore();
      }

      // 2. Draw Golden Dashed Prediction Path
      mainCtx.save();
      mainCtx.beginPath();
      mainCtx.setLineDash([5, 4]);
      mainCtx.strokeStyle = "#f59e0b";
      mainCtx.lineWidth = 2.5;
      overlayPoints.forEach((pt, i) => {
        if (i === 0) mainCtx.moveTo(pt.x, pt.y);
        else mainCtx.lineTo(pt.x, pt.y);
      });
      mainCtx.stroke();
      mainCtx.setLineDash([]);

      // 3. Draw Points & Key Markers
      overlayPoints.forEach((pt) => {
        if (pt.x < -20 || pt.x > plotW + 20) return;
        if (pt.isOrigin) {
          mainCtx.fillStyle = "#f59e0b";
          mainCtx.beginPath();
          mainCtx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
          mainCtx.fill();
          mainCtx.strokeStyle = "#ffffff";
          mainCtx.lineWidth = 1.5;
          mainCtx.stroke();

          mainCtx.font = "bold 9px 'Inter', sans-serif";
          mainCtx.fillStyle = "#fbbf24";
          mainCtx.fillText("📍 PREDICTION ORIGIN", pt.x - 45, pt.y - 10);
        } else {
          mainCtx.fillStyle = "#0f172a";
          mainCtx.beginPath();
          mainCtx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
          mainCtx.fill();

          mainCtx.strokeStyle = "#f59e0b";
          mainCtx.lineWidth = 2;
          mainCtx.stroke();

          mainCtx.fillStyle = "rgba(245, 158, 11, 0.85)";
          mainCtx.font = "bold 8px 'JetBrains Mono', monospace";
          mainCtx.fillText(`T+${pt.step}`, pt.x - 8, pt.y - 7);
        }
      });
      mainCtx.restore();
    }
  }

  // 5. Draw Time/Date Ticks along Bottom X-Axis (TradingView style)
  // CRITICAL: Ticks physically drag along with the graph in 1:1 sync
  mainCtx.fillStyle = "#64748b";
  mainCtx.font = "9px 'JetBrains Mono', monospace";
  mainCtx.textAlign = "center";

  const minLabelSpacingPx = 85;
  const tickStep = Math.max(1, Math.ceil(minLabelSpacingPx / barSpacing));
  const firstTickIdx = Math.floor(startIdx / tickStep) * tickStep;

  for (let i = firstTickIdx; i < endIdx + tickStep; i += tickStep) {
    if (i < 0 || i >= allBars.length) continue;
    const bar = allBars[i];
    const x = getBarX(i);

    if (x < 15 || x > plotW - 10) continue;

    // Tick mark moving with bar
    mainCtx.strokeStyle = "#1e2c4a";
    mainCtx.beginPath();
    mainCtx.moveTo(x, plotH);
    mainCtx.lineTo(x, plotH + 5);
    mainCtx.stroke();

    let label = "";
    if (bar.time_formatted) {
      const parts = bar.time_formatted.split(" ");
      if (state.interval === "1d") {
        label = `${parts[0]} ${parts[1]}`;
      } else {
        label = `${parts[0]} ${parts[1]} ${parts[2]}`;
      }
    } else if (bar.time) {
      label = bar.time.slice(5, 16).replace("T", " ");
    }
    mainCtx.fillText(label, x, plotH + 16);
  }

  // 6. Draw Crosshair on Hover
  if (state.isHovering && state.hoveredIndex !== null && state.hoveredIndex >= 0 && state.hoveredIndex < allBars.length) {
    const hoveredBar = allBars[state.hoveredIndex];
    const hoverX = getBarX(state.hoveredIndex);
    const hoverY = getY(hoveredBar.close);

    if (hoverX >= 0 && hoverX <= plotW) {
      mainCtx.setLineDash([2, 2]);
      mainCtx.strokeStyle = "rgba(255, 255, 255, 0.4)";
      mainCtx.lineWidth = 1;

      // Vertical line
      mainCtx.beginPath();
      mainCtx.moveTo(hoverX, 0);
      mainCtx.lineTo(hoverX, plotH);
      mainCtx.stroke();

      // Horizontal line
      mainCtx.beginPath();
      mainCtx.moveTo(0, hoverY);
      mainCtx.lineTo(plotW, hoverY);
      mainCtx.stroke();
      mainCtx.setLineDash([]);

      // Time Tag on Bottom X-Axis
      let timeTag = hoveredBar.time_formatted || hoveredBar.time;
      if (timeTag) {
        mainCtx.fillStyle = "#1e293b";
        mainCtx.fillRect(hoverX - 55, plotH + 2, 110, 18);
        mainCtx.strokeStyle = "#38bdf8";
        mainCtx.lineWidth = 1;
        mainCtx.strokeRect(hoverX - 55, plotH + 2, 110, 18);
        mainCtx.fillStyle = "#f8fafc";
        mainCtx.font = "bold 9px 'JetBrains Mono', monospace";
        mainCtx.textAlign = "center";
        mainCtx.fillText(timeTag.replace(" IST", ""), hoverX, plotH + 14);
      }

      // Price Tag on Y-Axis
      mainCtx.fillStyle = hoveredBar.close >= hoveredBar.open ? "#00c087" : "#f43f5e";
      mainCtx.fillRect(plotW + 2, hoverY - 9, 88, 18);
      mainCtx.fillStyle = "#ffffff";
      mainCtx.font = "bold 10px 'JetBrains Mono', monospace";
      mainCtx.textAlign = "left";
      mainCtx.fillText(`₹ ${formatINR(hoveredBar.close, 2)}`, plotW + 6, hoverY + 4);
    }
  }

  // 7. Sub-Chart (Volume & RSI)
  renderSubChart(allBars, startIdx, endIdx, getBarX, barSpacing, plotW);
}

function renderSubChart(allBars, startIdx, endIdx, getBarX, barSpacing, plotW) {
  const width = subCanvas.clientWidth;
  const height = subCanvas.clientHeight;
  subCtx.clearRect(0, 0, width, height);

  const plotH = height - 10;
  const visibleBars = allBars.slice(startIdx, endIdx);
  if (visibleBars.length === 0) return;
  const maxVol = Math.max(...visibleBars.map((b) => b.volume || 10), 10);

  // Draw RSI Levels (30 Oversold, 70 Overbought)
  subCtx.strokeStyle = "#1e2c4a";
  subCtx.setLineDash([2, 2]);
  subCtx.beginPath();
  subCtx.moveTo(0, plotH * 0.3);
  subCtx.lineTo(plotW, plotH * 0.3);
  subCtx.moveTo(0, plotH * 0.7);
  subCtx.lineTo(plotW, plotH * 0.7);
  subCtx.stroke();
  subCtx.setLineDash([]);

  for (let i = startIdx; i < endIdx; i++) {
    const bar = allBars[i];
    const x = getBarX(i);
    if (x < -barSpacing || x > plotW + barSpacing) continue;

    const vol = bar.volume || 1;
    const vH = (vol / maxVol) * (plotH * 0.85);
    const isBull = bar.close >= bar.open;

    subCtx.fillStyle = bar.is_predicted
      ? "rgba(6, 182, 212, 0.4)"
      : isBull
        ? "rgba(0, 192, 135, 0.45)"
        : "rgba(244, 63, 94, 0.45)";

    const barW = Math.max(barSpacing * 0.6, 2);
    subCtx.fillRect(x - barW / 2, plotH - vH, barW, vH);
  }
}

// --------------------------------------------------------------------------
// Pan, Drag & Zoom Handlers (TradingView Style)
// --------------------------------------------------------------------------
function handleCanvasMouseDown(e) {
  state.isDragging = true;
  state.dragStartX = e.clientX;
  state.dragStartPanOffset = state.panPixelOffset || 0;
  mainCanvas.style.cursor = "grabbing";
  chartTooltip.classList.add("hidden");
}

function handleCanvasMouseMove(e) {
  const rect = mainCanvas.getBoundingClientRect();
  const plotW = rect.width - 95;
  const barSpacing = plotW / state.visibleBars;
  const DEFAULT_RIGHT_OFFSET_BARS = 5;

  const data = state.forecastData || state.liveData;
  if (!data) return;
  const allBars = [...(data.history || data.candles || []), ...(data.forecast || [])];
  if (allBars.length === 0) return;

  if (state.isDragging) {
    const deltaX = e.clientX - state.dragStartX;
    state.panPixelOffset = state.dragStartPanOffset + deltaX;

    const minPan = -(DEFAULT_RIGHT_OFFSET_BARS + 20) * barSpacing;
    const maxPan = Math.max(0, (allBars.length - 4)) * barSpacing;
    state.panPixelOffset = Math.max(minPan, Math.min(maxPan, state.panPixelOffset));

    renderCharts();
    return;
  }

  // Crosshair Hovering
  if (
    e.clientX < rect.left ||
    e.clientX > rect.right ||
    e.clientY < rect.top ||
    e.clientY > rect.bottom
  ) {
    return;
  }

  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const g = Math.round(allBars.length - 1 + DEFAULT_RIGHT_OFFSET_BARS + 0.5 - (plotW + (state.panPixelOffset || 0) - mouseX) / barSpacing);

  if (g >= 0 && g < allBars.length) {
    state.isHovering = true;
    state.hoveredIndex = g;
    state.mousePos = { x: mouseX, y: mouseY };

    const bar = allBars[g];

    chartTooltip.classList.remove("hidden");
    chartTooltip.style.left = `${Math.min(mouseX + 16, rect.width - 210)}px`;
    chartTooltip.style.top = `${Math.max(mouseY - 90, 10)}px`;

    let statusTag = "";
    if (bar.is_predicted) {
      statusTag = `<span style="color:#06b6d4; font-weight:700;">● Kronos Projected Bar</span>`;
    } else if (bar.is_forming) {
      const bucket = getActiveBucketInfo(state.interval);
      statusTag = `<span style="color:#f59e0b; font-weight:700;">● Active Forming Candle (Closes in ${bucket.formattedRemaining})</span>`;
    } else {
      statusTag = `<span style="color:#00c087; font-weight:700;">● Realized Market Candle</span>`;
    }

    let timeHeader = bar.time_formatted || bar.time;
    if (bar.is_forming && bar.period_end_formatted) {
      timeHeader = `${timeHeader.replace(" IST", "")} → ${bar.period_end_formatted}`;
    }

    chartTooltip.innerHTML = `
      <div style="color:#94a3b8; font-size:10px; margin-bottom:2px;">${timeHeader}</div>
      <div style="margin-bottom:4px;">${statusTag}</div>
      <div>O: <strong>₹ ${formatINR(bar.open)}</strong></div>
      <div>H: <strong>₹ ${formatINR(bar.high)}</strong></div>
      <div>L: <strong>₹ ${formatINR(bar.low)}</strong></div>
      <div>C: <strong style="color:${bar.close >= bar.open ? '#00c087' : '#f43f5e'};">₹ ${formatINR(bar.close)} ${bar.is_forming ? '(LIVE)' : ''}</strong></div>
      <div>Vol: <strong>${formatINR(bar.volume, 0)}</strong></div>
      ${bar.lower_5 ? `<div style="color:#06b6d4; font-size:10px; margin-top:2px;">90% Band: ₹ ${formatINR(bar.lower_5, 0)} - ₹ ${formatINR(bar.upper_95, 0)}</div>` : ''}
    `;

    renderCharts();
  } else {
    state.isHovering = false;
    state.hoveredIndex = null;
    chartTooltip.classList.add("hidden");
    renderCharts();
  }
}

function handleCanvasMouseUp() {
  if (state.isDragging) {
    state.isDragging = false;
    mainCanvas.style.cursor = "crosshair";
  }
}

function handleCanvasWheel(e) {
  e.preventDefault();
  if (e.deltaY < 0) {
    zoomChart(-1); // Zoom in
  } else {
    zoomChart(1);  // Zoom out
  }
}

function zoomChart(direction) {
  const data = state.forecastData || state.liveData;
  if (!data) return;
  const allBars = [...(data.history || data.candles || []), ...(data.forecast || [])];

  const oldVisible = state.visibleBars;
  let newVisible;
  if (direction < 0) {
    newVisible = Math.max(15, Math.round(state.visibleBars * 0.85));
  } else {
    newVisible = Math.min(Math.min(allBars.length, 450), Math.round(state.visibleBars * 1.18));
  }
  if (newVisible === oldVisible) return;

  const scaleRatio = oldVisible / newVisible;
  state.panPixelOffset = (state.panPixelOffset || 0) * scaleRatio;

  state.visibleBars = newVisible;
  updateRangeButtonHighlight();
  renderCharts();
}

function resetChartToLive() {
  state.panPixelOffset = 0;
  state.visibleBars = 65;
  renderCharts();
}

function updateRangeButtonHighlight() {
  if (!quickRangeSelector) return;
  const buttons = quickRangeSelector.querySelectorAll(".range-btn");
  let closestBtn = null;
  let minDiff = Infinity;
  buttons.forEach((btn) => {
    const b = parseInt(btn.dataset.bars, 10);
    const diff = Math.abs(b - state.visibleBars);
    if (diff < minDiff) {
      minDiff = diff;
      closestBtn = btn;
    }
    btn.classList.remove("active");
  });
  if (closestBtn && minDiff < 25) closestBtn.classList.add("active");
}
