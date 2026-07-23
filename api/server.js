import 'dotenv/config';
import { randomUUID, timingSafeEqual } from 'node:crypto';
import http from 'node:http';
import { pathToFileURL } from 'node:url';
import cors from 'cors';
import express from 'express';
import helmet from 'helmet';
import { WebSocket, WebSocketServer } from 'ws';

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
    model_type: cleanText(body.model_type, 64) || 'unknown',
    model_file: cleanText(body.model_file, 128),
    source: cleanText(body.source, 128) || 'ring-bridge',
    device_timestamp_ms: Number.isFinite(Number(body.device_timestamp_ms))
      ? Number(body.device_timestamp_ms)
      : null,
    received_at: new Date().toISOString(),
  };
}

export function createRingServer({ bridgeToken = process.env.RING_BRIDGE_TOKEN || '' } = {}) {
  const app = express();
  const server = http.createServer(app);
  const webSocketServer = new WebSocketServer({ server, path: '/ws' });
  let latestGesture = null;

  app.use(helmet());
  app.use(cors());
  app.use(express.json({ limit: '64kb' }));

  app.get('/health', (_req, res) => {
    res.json({
      ok: true,
      service: 'ring-api',
      viewers: webSocketServer.clients.size,
      latest_gesture_at: latestGesture?.received_at ?? null,
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
    if (bridgeToken) {
      const authorization = req.get('authorization') || '';
      const supplied = authorization.startsWith('Bearer ')
        ? authorization.slice('Bearer '.length)
        : '';
      if (!supplied || !safeTokenEquals(bridgeToken, supplied)) {
        return res.status(401).json({ ok: false, error: 'invalid producer token' });
      }
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
  const host = process.env.HOST || '127.0.0.1';
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
