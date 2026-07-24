import 'dotenv/config';
import { randomUUID, timingSafeEqual } from 'node:crypto';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import cors from 'cors';
import express from 'express';
import helmet from 'helmet';
import { WebSocket, WebSocketServer } from 'ws';

const API_DIR = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.resolve(API_DIR, '..', 'public');

function safeTokenEquals(expected, supplied) {
  const expectedBytes = Buffer.from(expected);
  const suppliedBytes = Buffer.from(supplied);
  return (
    expectedBytes.length === suppliedBytes.length &&
    timingSafeEqual(expectedBytes, suppliedBytes)
  );
}

function cleanText(value, maximumLength) {
  return typeof value === 'string' ? value.trim().slice(0, maximumLength) : '';
}

function cleanNumber(value, maximumAbsolute) {
  const number = Number(value);
  return Number.isFinite(number) && Math.abs(number) <= maximumAbsolute
    ? number
    : null;
}

function normalizeVector(value, axes, maximumAbsolute) {
  if (!value || typeof value !== 'object') return null;
  const vector = {};
  for (const [index, axis] of axes.entries()) {
    const number = cleanNumber(
      Array.isArray(value) ? value[index] : value[axis],
      maximumAbsolute,
    );
    if (number === null) return null;
    vector[axis] = number;
  }
  return vector;
}

function normalizeMotion(value) {
  if (!value || typeof value !== 'object') return null;
  const position = normalizeVector(value.position_m, ['x', 'y', 'z'], 1000);
  const velocity = normalizeVector(value.velocity_mps, ['x', 'y', 'z'], 100);
  const linearAccel = normalizeVector(
    value.linear_accel_world_g,
    ['x', 'y', 'z'],
    64,
  );
  const correctedAccel = normalizeVector(
    value.corrected_accel_world_g,
    ['x', 'y', 'z'],
    64,
  );
  const accelBias = normalizeVector(
    value.accel_bias_world_g,
    ['x', 'y', 'z'],
    4,
  );
  if (!position || !velocity || !linearAccel || !correctedAccel || !accelBias) {
    return null;
  }
  const numeric = (key, maximum, fallback = 0) => {
    const parsed = cleanNumber(value[key], maximum);
    return parsed === null ? fallback : parsed;
  };
  return {
    armed: value.armed === true,
    moving: value.moving === true,
    rotating_only: value.rotating_only === true,
    translation_candidate: value.translation_candidate === true,
    position_m: position,
    velocity_mps: velocity,
    linear_accel_world_g: linearAccel,
    corrected_accel_world_g: correctedAccel,
    accel_bias_world_g: accelBias,
    accel_threshold_g: numeric('accel_threshold_g', 64),
    noise_sigma_g: numeric('noise_sigma_g', 64),
    speed_mps: Math.max(0, numeric('speed_mps', 100)),
    distance_m: Math.max(0, numeric('distance_m', 10000)),
    segment_id: Math.max(0, Math.trunc(numeric('segment_id', 1_000_000))),
    segment_elapsed_s: Math.max(0, numeric('segment_elapsed_s', 3600)),
    zupt_count: Math.max(0, Math.trunc(numeric('zupt_count', 1_000_000))),
    zupt_confidence: Math.min(1, Math.max(0, numeric('zupt_confidence', 1))),
    confidence: Math.min(1, Math.max(0, numeric('confidence', 1))),
  };
}

function producerAuthorized(request, bridgeToken) {
  if (!bridgeToken) return true;
  const authorization = request.get('authorization') || '';
  const supplied = authorization.startsWith('Bearer ')
    ? authorization.slice('Bearer '.length)
    : '';
  return Boolean(supplied && safeTokenEquals(bridgeToken, supplied));
}

function normalizeCircleMetrics(value) {
  if (!value || typeof value !== 'object') return null;
  const pathLength = cleanNumber(value.path_length_m, 10000);
  const radius = cleanNumber(value.radius_m, 1000);
  const turn = cleanNumber(value.turn_radians, 100);
  const closure = cleanNumber(value.closure_ratio, 100);
  const plane = cleanNumber(value.plane_score, 1);
  const roundness = cleanNumber(value.roundness, 1);
  if ([pathLength, radius, turn, closure, plane, roundness].includes(null)) {
    return null;
  }
  return {
    label: cleanText(value.label, 32) || 'circle',
    confidence: Math.min(1, Math.max(0, Number(value.confidence) || 0)),
    path_length_m: Math.max(0, pathLength),
    radius_m: Math.max(0, radius),
    turn_radians: Math.max(0, turn),
    closure_ratio: Math.max(0, closure),
    plane_score: Math.min(1, Math.max(0, plane)),
    roundness: Math.min(1, Math.max(0, roundness)),
  };
}

function normalizeRobotCommand(value) {
  if (!value || typeof value !== 'object') return null;
  const confidence = cleanNumber(value.confidence, 1);
  const confirmations = cleanNumber(value.confirmations, 100);
  if (confidence === null || confirmations === null) return null;
  return {
    command: cleanText(value.command, 64) || null,
    emitted: value.emitted === true,
    reason: cleanText(value.reason, 64) || 'unknown',
    armed: value.armed === true,
    confirmations: Math.max(0, Math.trunc(confirmations)),
    source_gesture: cleanText(value.source_gesture, 64),
    confidence: Math.min(1, Math.max(0, confidence)),
  };
}

function normalizeGesture(body) {
  const gesture = cleanText(body?.gesture, 64);
  const rawGesture = cleanText(body?.raw_gesture, 64) || gesture;
  const confidence = Number(body?.confidence);
  if (!gesture || !Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    return null;
  }

  const probabilities = {};
  if (body.probabilities && typeof body.probabilities === 'object') {
    for (const [label, value] of Object.entries(body.probabilities).slice(0, 32)) {
      const probability = Number(value);
      const cleanLabel = cleanText(label, 64);
      if (cleanLabel && Number.isFinite(probability) && probability >= 0 && probability <= 1) {
        probabilities[cleanLabel] = probability;
      }
    }
  }

  return {
    type: 'gesture',
    event_id: randomUUID(),
    gesture,
    raw_gesture: rawGesture,
    confidence,
    probabilities,
    model_type: cleanText(body.model_type, 128) || 'unknown',
    model_file: cleanText(body.model_file, 128),
    recognition_source: cleanText(body.recognition_source, 64) || 'mlp',
    direction_displacement_m:
      normalizeVector(body?.direction_displacement_m, ['x', 'y', 'z'], 1000),
    circle_metrics: normalizeCircleMetrics(body?.circle_metrics),
    robot_command: normalizeRobotCommand(body?.robot_command),
    source: cleanText(body.source, 128) || 'ring-bridge',
    device_timestamp_ms: Number.isFinite(Number(body.device_timestamp_ms))
      ? Number(body.device_timestamp_ms)
      : null,
    received_at: new Date().toISOString(),
  };
}

function normalizeTelemetry(body) {
  const quaternion = normalizeVector(body?.quaternion, ['w', 'x', 'y', 'z'], 1.1);
  const euler = normalizeVector(body?.euler_deg, ['roll', 'pitch', 'yaw'], 10000);
  const accel = normalizeVector(body?.accel_g, ['x', 'y', 'z'], 64);
  const gyro = normalizeVector(body?.gyro_dps, ['x', 'y', 'z'], 10000);
  const linearAccel = normalizeVector(body?.linear_accel_g, ['x', 'y', 'z'], 64);
  if (!quaternion || !euler || !accel || !gyro || !linearAccel) return null;

  const quaternionNorm = Math.hypot(
    quaternion.w,
    quaternion.x,
    quaternion.y,
    quaternion.z,
  );
  const sampleRate = Number(body?.sample_rate_hz);
  if (
    quaternionNorm < 0.5 ||
    !Number.isFinite(sampleRate) ||
    sampleRate <= 0 ||
    sampleRate > 2000
  ) {
    return null;
  }
  for (const axis of ['w', 'x', 'y', 'z']) {
    quaternion[axis] /= quaternionNorm;
  }

  return {
    type: 'telemetry',
    event_id: randomUUID(),
    quaternion,
    euler_deg: euler,
    accel_g: accel,
    gyro_dps: gyro,
    linear_accel_g: linearAccel,
    gyro_bias_dps:
      normalizeVector(body?.gyro_bias_dps, ['x', 'y', 'z'], 1000) ?? {
        x: 0,
        y: 0,
        z: 0,
      },
    gyro_raw_dps:
      normalizeVector(body?.gyro_raw_dps, ['x', 'y', 'z'], 10000) ?? gyro,
    stationary: body?.stationary === true,
    stationary_confidence: Math.min(
      1,
      Math.max(0, cleanNumber(body?.stationary_confidence, 1) ?? 0),
    ),
    calibrated: body?.calibrated === true,
    motion: normalizeMotion(body?.motion),
    sample_rate_hz: sampleRate,
    sequence: Number.isFinite(Number(body?.sequence))
      ? Math.max(0, Math.trunc(Number(body.sequence)))
      : null,
    device_timestamp_ms: Number.isFinite(Number(body?.device_timestamp_ms))
      ? Number(body.device_timestamp_ms)
      : null,
    source: cleanText(body?.source, 128) || 'ring-bridge',
    received_at: new Date().toISOString(),
  };
}

export function createRingServer({ bridgeToken = process.env.RING_BRIDGE_TOKEN || '' } = {}) {
  const app = express();
  const server = http.createServer(app);
  const webSocketServer = new WebSocketServer({ server, path: '/ws' });
  let latestGesture = null;
  let latestTelemetry = null;

  app.use(helmet());
  app.use(cors());
  app.use(express.json({ limit: '64kb' }));

  app.get('/health', (_req, res) => {
    res.json({
      ok: true,
      service: 'ring-api',
      viewers: webSocketServer.clients.size,
      latest_gesture_at: latestGesture?.received_at ?? null,
      latest_telemetry_at: latestTelemetry?.received_at ?? null,
      time: new Date().toISOString(),
    });
  });

  app.get('/hello', (_req, res) => {
    res.json({ message: 'hello from api.inudesu.xyz' });
  });

  app.post('/v1/echo', (req, res) => {
    res.json({ received: req.body ?? null });
  });

  app.get('/v1/gesture/latest', (_req, res) => {
    res.json({ ok: true, event: latestGesture });
  });

  app.post('/v1/gesture', (req, res) => {
    if (!producerAuthorized(req, bridgeToken)) {
      return res.status(401).json({ ok: false, error: 'invalid producer token' });
    }

    const event = normalizeGesture(req.body);
    if (!event) {
      return res.status(400).json({
        ok: false,
        error: 'gesture and confidence between 0 and 1 are required',
      });
    }
    latestGesture = event;
    const message = JSON.stringify(event);
    for (const client of webSocketServer.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    }
    return res.status(202).json({ ok: true, event_id: event.event_id });
  });

  app.get('/v1/telemetry/latest', (_req, res) => {
    res.json({ ok: true, event: latestTelemetry });
  });

  app.post('/v1/telemetry', (req, res) => {
    if (!producerAuthorized(req, bridgeToken)) {
      return res.status(401).json({ ok: false, error: 'invalid producer token' });
    }

    const event = normalizeTelemetry(req.body);
    if (!event) {
      return res.status(400).json({
        ok: false,
        error: 'valid quaternion, euler, acceleration, gyro and sample rate are required',
      });
    }
    latestTelemetry = event;
    const message = JSON.stringify(event);
    for (const client of webSocketServer.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    }
    return res.status(202).json({ ok: true, event_id: event.event_id });
  });

  // Serve the dashboard from the same process and origin as the API. Keeping
  // this after all API routes prevents a static file from shadowing /v1/*.
  app.use(express.static(PUBLIC_DIR, { extensions: ['html'] }));

  webSocketServer.on('connection', (socket) => {
    socket.isAlive = true;
    socket.on('pong', () => {
      socket.isAlive = true;
    });
    socket.send(
      JSON.stringify({
        type: 'hello',
        service: 'ring-api',
        connected_at: new Date().toISOString(),
      }),
    );
    if (latestGesture) {
      socket.send(JSON.stringify(latestGesture));
    }
    if (latestTelemetry) {
      socket.send(JSON.stringify(latestTelemetry));
    }
  });

  const heartbeat = setInterval(() => {
    for (const socket of webSocketServer.clients) {
      if (socket.isAlive === false) {
        socket.terminate();
        continue;
      }
      socket.isAlive = false;
      socket.ping();
    }
  }, 30_000);
  heartbeat.unref();

  server.on('close', () => {
    clearInterval(heartbeat);
    webSocketServer.close();
  });

  return { app, server, webSocketServer };
}

export function startRingServer() {
  const port = Number(process.env.PORT || 3000);
  const host = process.env.HOST || '0.0.0.0';
  const instance = createRingServer();
  instance.server.listen(port, host, () => {
    console.log(`ring-api listening on http://${host}:${port}`);
  });
  return instance;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const instance = startRingServer();
  const shutdown = () => {
    for (const socket of instance.webSocketServer.clients) {
      socket.close(1001, 'server shutdown');
    }
    instance.server.close(() => process.exit(0));
  };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}
