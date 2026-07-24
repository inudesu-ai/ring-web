const elements = {
  apiDot: document.querySelector('#api-dot'),
  apiStatus: document.querySelector('#api-status'),
  streamDot: document.querySelector('#stream-dot'),
  streamStatus: document.querySelector('#stream-status'),
  sampleRate: document.querySelector('#sample-rate'),
  motionState: document.querySelector('#motion-state'),
  motionCanvas: document.querySelector('#motion-canvas'),
  signalCanvas: document.querySelector('#signal-canvas'),
  gesture: document.querySelector('#gesture-name'),
  gestureDetail: document.querySelector('#gesture-detail'),
  confidence: document.querySelector('#confidence'),
  confidenceBar: document.querySelector('#confidence-bar'),
  probabilities: document.querySelector('#probabilities'),
  modelType: document.querySelector('#model-type'),
  sequence: document.querySelector('#sequence'),
  quaternion: document.querySelector('#quaternion-value'),
  roll: document.querySelector('#roll-value'),
  pitch: document.querySelector('#pitch-value'),
  yaw: document.querySelector('#yaw-value'),
  rollBar: document.querySelector('#roll-bar'),
  pitchBar: document.querySelector('#pitch-bar'),
  yawBar: document.querySelector('#yaw-bar'),
  accel: ['x', 'y', 'z'].map((axis) => document.querySelector(`#accel-${axis}`)),
  gyro: ['x', 'y', 'z'].map((axis) => document.querySelector(`#gyro-${axis}`)),
  signalWindow: document.querySelector('#signal-window'),
  eventList: document.querySelector('#event-list'),
  eventCount: document.querySelector('#event-count'),
  lastUpdate: document.querySelector('#last-update'),
  zeroPose: document.querySelector('#zero-pose'),
  demoToggle: document.querySelector('#demo-toggle'),
};

const local = ['localhost', '127.0.0.1'].includes(window.location.hostname);
const query = new URLSearchParams(window.location.search);
const apiBase = query.get('api') || (local ? 'http://127.0.0.1:3000' : 'https://api.inudesu.xyz');
const socketBase = apiBase.replace(/^http/, 'ws');
const webSocketUrl = `${socketBase}/ws`;

const gestureNames = {
  idle: '静止',
  wave: '挥手',
  rotate_back: '向后旋转',
  rotate_front: '向前旋转',
  left: '向左',
  right: '向右',
  up: '向上',
  down: '向下',
  circle: '画圆',
  double_tap: '双击',
  uncertain: '不确定',
};

const axisColors = ['#ff9b50', '#67e8a5', '#67b7ff'];
const identityQuaternion = [1, 0, 0, 0];
const state = {
  socket: null,
  reconnectDelay: 1000,
  lastTelemetryAt: null,
  lastGestureAt: null,
  rawTargetQuaternion: [...identityQuaternion],
  displayQuaternion: [...identityQuaternion],
  zeroQuaternion: [...identityQuaternion],
  acceleration: [0, 0, 1],
  gyro: [0, 0, 0],
  linearAcceleration: [0, 0, 0],
  motionPosition: [0, 0, 0],
  motionVelocity: [0, 0, 0],
  motionTrail: [],
  lastMotionAt: null,
  history: [],
  sampleRate: 0,
  stationary: true,
  events: [],
  demo: false,
  demoTimer: null,
  demoGestureIndex: -1,
};

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function number(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function vector(object, axes) {
  return axes.map((axis) => number(object?.[axis]));
}

function normalizeQuaternion(quaternion) {
  const length = Math.hypot(...quaternion);
  return length > 1e-9
    ? quaternion.map((value) => value / length)
    : [...identityQuaternion];
}

function quaternionConjugate([w, x, y, z]) {
  return [w, -x, -y, -z];
}

function quaternionMultiply([aw, ax, ay, az], [bw, bx, by, bz]) {
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

function quaternionFromEuler(roll, pitch, yaw) {
  const cr = Math.cos(roll / 2);
  const sr = Math.sin(roll / 2);
  const cp = Math.cos(pitch / 2);
  const sp = Math.sin(pitch / 2);
  const cy = Math.cos(yaw / 2);
  const sy = Math.sin(yaw / 2);
  return normalizeQuaternion([
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ]);
}

function quaternionToEuler([w, x, y, z]) {
  return [
    Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
    Math.asin(clamp(2 * (w * y - z * x), -1, 1)),
    Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
  ].map((value) => value * 180 / Math.PI);
}

function rotateVector(quaternion, [vx, vy, vz]) {
  const vectorQuaternion = [0, vx, vy, vz];
  return quaternionMultiply(
    quaternionMultiply(quaternion, vectorQuaternion),
    quaternionConjugate(quaternion),
  ).slice(1);
}

function smoothQuaternion(current, target, amount) {
  let destination = target;
  const dot = current.reduce((sum, value, index) => sum + value * target[index], 0);
  if (dot < 0) destination = target.map((value) => -value);
  return normalizeQuaternion(
    current.map((value, index) => value + (destination[index] - value) * amount),
  );
}

function readableGesture(value) {
  const key = String(value || 'unknown');
  return gestureNames[key] || key.replaceAll('_', ' ');
}

function formatSigned(value, digits) {
  const normalized = Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value;
  return `${normalized >= 0 ? '+' : ''}${normalized.toFixed(digits)}`;
}

function eventDate(event) {
  const parsed = new Date(event?.received_at);
  return Number.isNaN(parsed.getTime()) ? new Date() : parsed;
}

function setApiState(status, label) {
  elements.apiDot.className = `status-dot ${status}`;
  elements.apiStatus.textContent = label;
}

function setStreamState(status, label) {
  elements.streamDot.className = `status-dot ${status}`;
  elements.streamStatus.textContent = label;
}

function renderProbabilities(probabilities) {
  const entries = Object.entries(probabilities || {})
    .filter(([, value]) => Number.isFinite(Number(value)))
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 5);
  elements.probabilities.replaceChildren();

  if (!entries.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-state';
    empty.textContent = '等待第一组分类概率';
    elements.probabilities.append(empty);
    return;
  }

  for (const [label, rawValue] of entries) {
    const probability = clamp(Number(rawValue), 0, 1);
    const row = document.createElement('div');
    row.className = 'probability-row';

    const name = document.createElement('span');
    name.textContent = readableGesture(label);
    const track = document.createElement('i');
    const bar = document.createElement('b');
    bar.style.width = `${probability * 100}%`;
    track.append(bar);
    const value = document.createElement('span');
    value.textContent = `${Math.round(probability * 100)}%`;
    row.append(name, track, value);
    elements.probabilities.append(row);
  }
}

function renderEventLog() {
  elements.eventList.replaceChildren();
  elements.eventCount.textContent = String(state.events.length).padStart(2, '0');
  if (!state.events.length) {
    const empty = document.createElement('li');
    empty.className = 'empty-event';
    empty.textContent = '暂无手势事件';
    elements.eventList.append(empty);
    return;
  }

  for (const event of state.events.slice(0, 6)) {
    const item = document.createElement('li');
    const time = document.createElement('time');
    time.textContent = event.at.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
    const gesture = document.createElement('strong');
    gesture.textContent = readableGesture(event.gesture);
    const confidence = document.createElement('span');
    confidence.textContent = `${Math.round(event.confidence * 100)}%`;
    item.append(time, gesture, confidence);
    elements.eventList.append(item);
  }
}

function renderGesture(event) {
  if (!event || event.type !== 'gesture' || (state.demo && !event.demo)) return;
  const confidence = clamp(number(event.confidence), 0, 1);
  const at = eventDate(event);
  state.lastGestureAt = at;

  elements.gesture.textContent = readableGesture(event.gesture);
  elements.gestureDetail.textContent =
    event.gesture === event.raw_gesture
      ? `${event.source || 'ring-bridge'} / 识别通过`
      : `候选：${readableGesture(event.raw_gesture)} / 低于阈值`;
  elements.confidence.textContent = String(Math.round(confidence * 100));
  elements.confidenceBar.style.width = `${confidence * 100}%`;
  elements.modelType.textContent = event.model_type || 'unknown';
  renderProbabilities(event.probabilities);

  state.events.unshift({
    gesture: event.gesture,
    confidence,
    at,
  });
  state.events = state.events.slice(0, 12);
  renderEventLog();
}

function integrateRelativeMotion(linearAcceleration, stationary, receivedAt) {
  const current = receivedAt.getTime() / 1000;
  const dt = state.lastMotionAt === null
    ? 1 / Math.max(state.sampleRate, 10)
    : clamp(current - state.lastMotionAt, 0.01, 0.2);
  state.lastMotionAt = current;

  if (stationary) {
    state.motionVelocity = state.motionVelocity.map((value) => value * 0.25);
    state.motionPosition = state.motionPosition.map((value) => value * 0.9);
  } else {
    state.motionVelocity = state.motionVelocity.map(
      (value, index) => (value + linearAcceleration[index] * 9.80665 * dt) * 0.76,
    );
    state.motionPosition = state.motionPosition.map(
      (value, index) => clamp((value + state.motionVelocity[index] * dt) * 0.995, -1.25, 1.25),
    );
  }
  state.motionTrail.push([...state.motionPosition]);
  state.motionTrail = state.motionTrail.slice(-36);
}

function renderTelemetry(event) {
  if (!event || event.type !== 'telemetry' || (state.demo && !event.demo)) return;
  const at = eventDate(event);
  state.lastTelemetryAt = at;
  state.rawTargetQuaternion = normalizeQuaternion(
    vector(event.quaternion, ['w', 'x', 'y', 'z']),
  );
  state.acceleration = vector(event.accel_g, ['x', 'y', 'z']);
  state.gyro = vector(event.gyro_dps, ['x', 'y', 'z']);
  state.linearAcceleration = vector(event.linear_accel_g, ['x', 'y', 'z']);
  state.sampleRate = number(event.sample_rate_hz);
  state.stationary = event.stationary === true;

  integrateRelativeMotion(state.linearAcceleration, state.stationary, at);
  state.history.push({
    at,
    accel: [...state.acceleration],
    gyro: [...state.gyro],
  });
  state.history = state.history.slice(-240);

  elements.sampleRate.textContent = `${state.sampleRate.toFixed(0)} Hz`;
  elements.motionState.textContent = state.stationary ? 'STATIONARY' : 'IN MOTION';
  elements.motionState.classList.toggle('active', !state.stationary);
  elements.sequence.textContent = Number.isFinite(Number(event.sequence))
    ? `SEQ ${Number(event.sequence)}`
    : 'SEQ —';

  state.acceleration.forEach((value, index) => {
    elements.accel[index].textContent = formatSigned(value, 3);
  });
  state.gyro.forEach((value, index) => {
    elements.gyro[index].textContent = formatSigned(value, 2);
  });
  elements.quaternion.textContent = state.rawTargetQuaternion
    .map((value) => value.toFixed(3))
    .join(' ');
  elements.lastUpdate.textContent = `TELEMETRY ${at.toLocaleTimeString()} / ${event.source || 'RING'}`;
}

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

function cameraRotate([x, y, z]) {
  const yaw = -0.58;
  const pitch = 0.32;
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  return [x1, cp * y - sp * z1, sp * y + cp * z1];
}

function project(point, width, height) {
  const [x, y, z] = cameraRotate(point);
  const depth = 5.8;
  const perspective = depth / Math.max(2.2, depth - z);
  const scale = Math.min(width, height) * 0.135;
  return {
    x: width * 0.55 + x * scale * perspective,
    y: height * 0.43 - y * scale * perspective,
    z,
    perspective,
  };
}

function transformedPoint(local, quaternion) {
  const rotated = rotateVector(quaternion, local);
  return [
    rotated[0] + state.motionPosition[0] * 0.45,
    rotated[1] + state.motionPosition[2] * 0.45,
    rotated[2] + state.motionPosition[1] * 0.45,
  ];
}

function drawGrid(context, width, height) {
  context.save();
  context.strokeStyle = 'rgba(255,255,255,.045)';
  context.lineWidth = 1;
  const horizon = height * 0.66;
  for (let index = -6; index <= 6; index += 1) {
    context.beginPath();
    context.moveTo(width * 0.5 + index * 18, horizon);
    context.lineTo(width * 0.5 + index * 72, height * 0.96);
    context.stroke();
  }
  for (let index = 0; index < 6; index += 1) {
    const t = index / 5;
    const y = horizon + t * t * (height * 0.29);
    context.beginPath();
    context.moveTo(width * 0.08, y);
    context.lineTo(width * 0.92, y);
    context.stroke();
  }
  context.restore();
}

function drawMotionTrail(context, width, height) {
  if (state.motionTrail.length < 2) return;
  context.save();
  context.lineWidth = 1.2;
  context.beginPath();
  state.motionTrail.forEach((position, index) => {
    const projected = project(
      [position[0] * 0.45, position[2] * 0.45, position[1] * 0.45],
      width,
      height,
    );
    if (index === 0) context.moveTo(projected.x, projected.y);
    else context.lineTo(projected.x, projected.y);
  });
  const gradient = context.createLinearGradient(width * 0.3, 0, width * 0.7, 0);
  gradient.addColorStop(0, 'rgba(200,255,61,0)');
  gradient.addColorStop(1, 'rgba(200,255,61,.48)');
  context.strokeStyle = gradient;
  context.stroke();
  context.restore();
}

function drawAxes(context, quaternion, width, height) {
  const origin = project(transformedPoint([0, 0, 0], quaternion), width, height);
  const axes = [
    { vector: [1.85, 0, 0], color: '#ff9b50', label: 'X' },
    { vector: [0, 1.85, 0], color: '#67e8a5', label: 'Y' },
    { vector: [0, 0, 1.85], color: '#67b7ff', label: 'Z' },
  ];
  context.save();
  context.font = '9px SFMono-Regular, monospace';
  for (const axis of axes) {
    const end = project(transformedPoint(axis.vector, quaternion), width, height);
    context.beginPath();
    context.moveTo(origin.x, origin.y);
    context.lineTo(end.x, end.y);
    context.strokeStyle = axis.color;
    context.globalAlpha = 0.72;
    context.lineWidth = 1.1;
    context.stroke();
    context.fillStyle = axis.color;
    context.fillText(axis.label, end.x + 5, end.y - 3);
  }
  context.restore();
}

function drawRing(context, quaternion, width, height) {
  const majorRadius = 1.28;
  const minorRadius = 0.28;
  const uCount = 42;
  const vCount = 10;
  const points = [];
  for (let v = 0; v < vCount; v += 1) {
    points[v] = [];
    const vAngle = v / vCount * Math.PI * 2;
    for (let u = 0; u < uCount; u += 1) {
      const uAngle = u / uCount * Math.PI * 2;
      const radius = majorRadius + minorRadius * Math.cos(vAngle);
      const local = [
        radius * Math.cos(uAngle),
        radius * Math.sin(uAngle),
        minorRadius * Math.sin(vAngle),
      ];
      points[v][u] = project(transformedPoint(local, quaternion), width, height);
    }
  }

  const segments = [];
  for (let v = 0; v < vCount; v += 1) {
    for (let u = 0; u < uCount; u += 1) {
      const next = (u + 1) % uCount;
      segments.push([points[v][u], points[v][next]]);
    }
  }
  for (let u = 0; u < uCount; u += 6) {
    for (let v = 0; v < vCount; v += 1) {
      const next = (v + 1) % vCount;
      segments.push([points[v][u], points[next][u]]);
    }
  }
  segments.sort((a, b) => (a[0].z + a[1].z) - (b[0].z + b[1].z));

  context.save();
  context.lineCap = 'round';
  context.shadowColor = 'rgba(200,255,61,.24)';
  context.shadowBlur = 7;
  for (const [start, end] of segments) {
    const depth = clamp((start.z + end.z + 4) / 8, 0, 1);
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.strokeStyle = `rgba(200,255,61,${0.14 + depth * 0.58})`;
    context.lineWidth = 0.65 + depth * 0.75;
    context.stroke();
  }
  context.restore();

  drawSensorModule(context, quaternion, width, height);
}

function drawSensorModule(context, quaternion, width, height) {
  const center = [0, -1.43, 0.12];
  const half = [0.45, 0.24, 0.25];
  const vertices = [];
  for (const x of [-1, 1]) {
    for (const y of [-1, 1]) {
      for (const z of [-1, 1]) {
        vertices.push([
          center[0] + x * half[0],
          center[1] + y * half[1],
          center[2] + z * half[2],
        ]);
      }
    }
  }
  const edges = [
    [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
    [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7],
  ];
  const projected = vertices.map((point) =>
    project(transformedPoint(point, quaternion), width, height));
  context.save();
  context.strokeStyle = 'rgba(255,155,80,.8)';
  context.fillStyle = 'rgba(255,155,80,.08)';
  context.shadowColor = 'rgba(255,155,80,.45)';
  context.shadowBlur = 8;
  context.lineWidth = 1;
  context.beginPath();
  for (const [from, to] of edges) {
    context.moveTo(projected[from].x, projected[from].y);
    context.lineTo(projected[to].x, projected[to].y);
  }
  context.stroke();
  context.restore();
}

function updatePoseReadout(quaternion) {
  const [roll, pitch, yaw] = quaternionToEuler(quaternion);
  const values = [roll, pitch, yaw];
  const outputs = [elements.roll, elements.pitch, elements.yaw];
  const bars = [elements.rollBar, elements.pitchBar, elements.yawBar];
  values.forEach((value, index) => {
    outputs[index].textContent = `${formatSigned(value, 1)}°`;
    bars[index].style.width = `${clamp((value + 180) / 360 * 100, 0, 100)}%`;
  });
}

function drawMotionCanvas() {
  const { context, width, height } = fitCanvas(elements.motionCanvas);
  context.clearRect(0, 0, width, height);
  drawGrid(context, width, height);
  drawMotionTrail(context, width, height);

  const target = normalizeQuaternion(
    quaternionMultiply(state.zeroQuaternion, state.rawTargetQuaternion),
  );
  state.displayQuaternion = smoothQuaternion(state.displayQuaternion, target, 0.11);
  drawAxes(context, state.displayQuaternion, width, height);
  drawRing(context, state.displayQuaternion, width, height);
  updatePoseReadout(state.displayQuaternion);
}

function drawSeries(context, samples, key, region, scale) {
  if (samples.length < 2) return;
  const width = region.right - region.left;
  const height = region.bottom - region.top;
  for (let axis = 0; axis < 3; axis += 1) {
    context.beginPath();
    samples.forEach((sample, index) => {
      const x = region.left + index / Math.max(1, samples.length - 1) * width;
      const raw = sample[key][axis];
      const y = region.top + height / 2 - clamp(raw / scale, -1, 1) * height * 0.42;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.strokeStyle = axisColors[axis];
    context.globalAlpha = 0.82;
    context.lineWidth = 1.15;
    context.stroke();
  }
}

function drawSignalCanvas() {
  const { context, width, height } = fitCanvas(elements.signalCanvas);
  context.clearRect(0, 0, width, height);
  const left = 58;
  const right = width - 14;
  const midpoint = height / 2;
  const regions = [
    { top: 8, bottom: midpoint - 7, label: 'ACC / g' },
    { top: midpoint + 7, bottom: height - 8, label: 'GYRO / °s' },
  ];

  context.save();
  context.font = '8px SFMono-Regular, monospace';
  for (const region of regions) {
    context.fillStyle = 'rgba(133,140,132,.72)';
    context.fillText(region.label, 8, region.top + 8);
    for (let row = 0; row <= 4; row += 1) {
      const y = region.top + (region.bottom - region.top) * row / 4;
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(right, y);
      context.strokeStyle = row === 2
        ? 'rgba(255,255,255,.11)'
        : 'rgba(255,255,255,.045)';
      context.lineWidth = 1;
      context.stroke();
    }
  }
  for (let column = 0; column <= 8; column += 1) {
    const x = left + (right - left) * column / 8;
    context.beginPath();
    context.moveTo(x, 8);
    context.lineTo(x, height - 8);
    context.strokeStyle = 'rgba(255,255,255,.035)';
    context.stroke();
  }

  const maxGyro = Math.max(
    40,
    ...state.history.flatMap((sample) => sample.gyro.map((value) => Math.abs(value))),
  );
  drawSeries(context, state.history, 'accel', regions[0], 2);
  drawSeries(context, state.history, 'gyro', regions[1], Math.min(maxGyro * 1.15, 2000));
  context.restore();

  if (state.history.length > 1) {
    const seconds = (
      state.history.at(-1).at.getTime() - state.history[0].at.getTime()
    ) / 1000;
    elements.signalWindow.textContent = `最近 ${Math.max(0, seconds).toFixed(1)}s`;
  }
}

function animationFrame() {
  drawMotionCanvas();
  drawSignalCanvas();
  window.requestAnimationFrame(animationFrame);
}

async function loadLatest() {
  try {
    const [gestureResponse, telemetryResponse] = await Promise.all([
      fetch(`${apiBase}/v1/gesture/latest`),
      fetch(`${apiBase}/v1/telemetry/latest`),
    ]);
    if (gestureResponse.ok) {
      const payload = await gestureResponse.json();
      renderGesture(payload.event);
    }
    if (telemetryResponse.ok) {
      const payload = await telemetryResponse.json();
      renderTelemetry(payload.event);
    }
  } catch {
    // WebSocket reconnection owns temporary API recovery.
  }
}

function connect() {
  setApiState('', '正在连接');
  const socket = new WebSocket(webSocketUrl);
  state.socket = socket;

  socket.addEventListener('open', () => {
    state.reconnectDelay = 1000;
    setApiState('connected', '在线');
  });
  socket.addEventListener('message', (message) => {
    try {
      const event = JSON.parse(message.data);
      if (event.type === 'gesture') renderGesture(event);
      if (event.type === 'telemetry') renderTelemetry(event);
    } catch {
      // Ignore malformed non-JSON messages.
    }
  });
  socket.addEventListener('close', () => {
    setApiState('error', '重新连接');
    window.setTimeout(connect, state.reconnectDelay);
    state.reconnectDelay = Math.min(state.reconnectDelay * 1.8, 15000);
  });
  socket.addEventListener('error', () => socket.close());
}

function demoTelemetry(elapsed) {
  const roll = Math.sin(elapsed * 1.3) * 0.45;
  const pitch = Math.sin(elapsed * 0.9 + 0.7) * 0.32;
  const yaw = Math.sin(elapsed * 0.42) * 0.9;
  const quaternion = quaternionFromEuler(roll, pitch, yaw);
  const linear = [
    Math.sin(elapsed * 2.4) * 0.08,
    Math.cos(elapsed * 1.8) * 0.06,
    Math.sin(elapsed * 2.1 + 0.8) * 0.045,
  ];
  const gravityBody = rotateVector(quaternionConjugate(quaternion), [0, 0, 1]);
  const accel = gravityBody.map((value, index) => value + linear[index]);
  renderTelemetry({
    type: 'telemetry',
    demo: true,
    quaternion: Object.fromEntries(['w', 'x', 'y', 'z'].map((axis, index) => [axis, quaternion[index]])),
    accel_g: Object.fromEntries(['x', 'y', 'z'].map((axis, index) => [axis, accel[index]])),
    gyro_dps: {
      x: Math.cos(elapsed * 1.3) * 33,
      y: Math.cos(elapsed * 0.9 + 0.7) * 17,
      z: Math.cos(elapsed * 0.42) * 22,
    },
    linear_accel_g: Object.fromEntries(
      ['x', 'y', 'z'].map((axis, index) => [axis, linear[index]]),
    ),
    sample_rate_hz: 100,
    sequence: Math.floor(elapsed * 100),
    stationary: false,
    received_at: new Date().toISOString(),
    source: 'browser-demo',
  });
}

function demoGesture(elapsed) {
  const gestures = ['wave', 'rotate_front', 'circle', 'left', 'double_tap', 'idle'];
  const index = Math.floor(elapsed / 2.8) % gestures.length;
  if (index === state.demoGestureIndex) return;
  state.demoGestureIndex = index;
  const gesture = gestures[index];
  const probabilities = Object.fromEntries(
    gestures.map((label) => [label, label === gesture ? 0.91 : 0.018]),
  );
  renderGesture({
    type: 'gesture',
    demo: true,
    gesture,
    raw_gesture: gesture,
    confidence: 0.91,
    probabilities,
    model_type: 'ring-mlp-v1 / DEMO',
    source: 'browser-demo',
    received_at: new Date().toISOString(),
  });
}

function startDemo() {
  state.demo = true;
  state.demoGestureIndex = -1;
  state.history = [];
  state.motionTrail = [];
  state.lastMotionAt = null;
  elements.demoToggle.classList.add('active');
  elements.demoToggle.lastChild.textContent = ' 返回实时';
  setStreamState('connected', '模拟信号');
  const started = performance.now();
  state.demoTimer = window.setInterval(() => {
    const elapsed = (performance.now() - started) / 1000;
    demoTelemetry(elapsed);
    demoGesture(elapsed);
  }, 50);
}

function stopDemo() {
  state.demo = false;
  window.clearInterval(state.demoTimer);
  state.demoTimer = null;
  state.history = [];
  state.motionTrail = [];
  state.lastMotionAt = null;
  state.demoToggle.classList.remove('active');
  elements.demoToggle.lastChild.textContent = ' 模拟信号';
  setStreamState('muted', '等待实时数据');
  loadLatest();
}

elements.demoToggle.addEventListener('click', () => {
  if (state.demo) stopDemo();
  else startDemo();
});

elements.zeroPose.addEventListener('click', () => {
  state.zeroQuaternion = quaternionConjugate(state.rawTargetQuaternion);
  state.motionPosition = [0, 0, 0];
  state.motionVelocity = [0, 0, 0];
  state.motionTrail = [];
  const original = elements.zeroPose.lastChild.textContent;
  elements.zeroPose.lastChild.textContent = ' 已归零';
  window.setTimeout(() => {
    elements.zeroPose.lastChild.textContent = original;
  }, 900);
});

window.setInterval(() => {
  if (state.demo) return;
  if (!state.lastTelemetryAt) {
    setStreamState('muted', '等待数据');
    return;
  }
  const age = (Date.now() - state.lastTelemetryAt.getTime()) / 1000;
  if (age < 2.5) {
    setStreamState('connected', '实时');
  } else if (age < 10) {
    setStreamState('', `${Math.round(age)}s 前`);
  } else {
    setStreamState('error', '数据中断');
  }
}, 1000);

renderEventLog();
loadLatest();
connect();
window.requestAnimationFrame(animationFrame);
