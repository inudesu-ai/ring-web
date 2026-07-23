const elements = {
  connection: document.querySelector('#connection'),
  connectionLabel: document.querySelector('#connection-label'),
  gesture: document.querySelector('#gesture-name'),
  rawGesture: document.querySelector('#raw-gesture'),
  timestamp: document.querySelector('#timestamp'),
  confidence: document.querySelector('#confidence'),
  confidenceRing: document.querySelector('#confidence-ring'),
  modelType: document.querySelector('#model-type'),
  source: document.querySelector('#source'),
  probabilities: document.querySelector('#probabilities'),
};

const local = ['localhost', '127.0.0.1'].includes(window.location.hostname);
const apiBase = local ? 'http://127.0.0.1:3000' : 'https://api.inudesu.xyz';
const webSocketUrl = local ? 'ws://127.0.0.1:3000/ws' : 'wss://api.inudesu.xyz/ws';
let reconnectDelay = 1000;
let lastEventAt = null;

function readableLabel(value) {
  return String(value || 'unknown').replaceAll('_', ' ');
}

function setConnection(state, label) {
  elements.connection.className = `connection ${state}`;
  elements.connectionLabel.textContent = label;
}

function renderProbabilities(probabilities) {
  const entries = Object.entries(probabilities || {})
    .filter(([, value]) => Number.isFinite(Number(value)))
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  elements.probabilities.replaceChildren();

  if (!entries.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-state';
    empty.textContent = 'No class probability data in this event.';
    elements.probabilities.append(empty);
    return;
  }

  for (const [label, rawValue] of entries) {
    const value = Math.max(0, Math.min(1, Number(rawValue)));
    const item = document.createElement('div');
    item.className = 'probability-item';

    const title = document.createElement('div');
    title.className = 'probability-title';
    const name = document.createElement('span');
    name.textContent = readableLabel(label);
    const percentage = document.createElement('span');
    percentage.textContent = `${Math.round(value * 100)}%`;
    title.append(name, percentage);

    const track = document.createElement('div');
    track.className = 'probability-track';
    const bar = document.createElement('div');
    bar.className = 'probability-value';
    bar.style.width = `${value * 100}%`;
    track.append(bar);
    item.append(title, track);
    elements.probabilities.append(item);
  }
}

function renderGesture(event) {
  if (!event || event.type !== 'gesture') return;
  const confidence = Math.max(0, Math.min(1, Number(event.confidence) || 0));
  const date = new Date(event.received_at);
  lastEventAt = Number.isNaN(date.getTime()) ? new Date() : date;

  elements.gesture.textContent = readableLabel(event.gesture);
  elements.rawGesture.textContent =
    event.gesture === event.raw_gesture
      ? 'prediction accepted'
      : `candidate · ${readableLabel(event.raw_gesture)}`;
  elements.timestamp.textContent = `Updated ${lastEventAt.toLocaleTimeString()}`;
  elements.confidence.textContent = String(Math.round(confidence * 100));
  elements.confidenceRing.style.background =
    `conic-gradient(var(--acid) ${confidence * 360}deg, rgba(255,255,255,.07) 0deg)`;
  elements.modelType.textContent = event.model_type || 'unknown';
  elements.source.textContent = event.source || 'ring-bridge';
  renderProbabilities(event.probabilities);
}

async function loadLatest() {
  try {
    const response = await fetch(`${apiBase}/v1/gesture/latest`);
    if (!response.ok) return;
    const payload = await response.json();
    renderGesture(payload.event);
  } catch {
    // WebSocket reconnect handles temporary API downtime.
  }
}

function connect() {
  setConnection('', 'Connecting');
  const socket = new WebSocket(webSocketUrl);

  socket.addEventListener('open', () => {
    reconnectDelay = 1000;
    setConnection('connected', 'API connected');
  });

  socket.addEventListener('message', (message) => {
    try {
      renderGesture(JSON.parse(message.data));
    } catch {
      // Ignore malformed non-JSON server messages.
    }
  });

  socket.addEventListener('close', () => {
    setConnection('disconnected', 'Reconnecting');
    window.setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 1.8, 15000);
  });

  socket.addEventListener('error', () => socket.close());
}

window.setInterval(() => {
  if (!lastEventAt) return;
  const seconds = Math.round((Date.now() - lastEventAt.getTime()) / 1000);
  if (seconds > 5) {
    elements.timestamp.textContent = `Last prediction ${seconds}s ago`;
  }
}, 1000);

loadLatest();
connect();
