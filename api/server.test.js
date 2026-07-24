import assert from 'node:assert/strict';
import { once } from 'node:events';
import test from 'node:test';
import { WebSocket } from 'ws';
import { createRingServer } from './server.js';

function nextJson(socket) {
  return new Promise((resolve, reject) => {
    socket.once('message', (data) => {
      try {
        resolve(JSON.parse(data.toString()));
      } catch (error) {
        reject(error);
      }
    });
    socket.once('error', reject);
  });
}

test('authenticated gesture events are broadcast and retained', async (context) => {
  const instance = createRingServer({ bridgeToken: 'test-secret' });
  instance.server.listen(0, '127.0.0.1');
  await once(instance.server, 'listening');
  const { port } = instance.server.address();
  const baseUrl = `http://127.0.0.1:${port}`;
  const socket = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  const helloPromise = nextJson(socket);
  context.after(async () => {
    socket.close();
    await new Promise((resolve) => instance.server.close(resolve));
  });

  await once(socket, 'open');
  const hello = await helloPromise;
  assert.equal(hello.type, 'hello');

  const rejected = await fetch(`${baseUrl}/v1/gesture`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ gesture: 'wave', confidence: 0.9 }),
  });
  assert.equal(rejected.status, 401);

  const broadcastPromise = nextJson(socket);
  const accepted = await fetch(`${baseUrl}/v1/gesture`, {
    method: 'POST',
    headers: {
      authorization: 'Bearer test-secret',
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      gesture: 'wave',
      raw_gesture: 'wave',
      confidence: 0.93,
      probabilities: { wave: 0.93, idle: 0.07 },
      model_type: 'ring-mlp-v1',
    }),
  });
  assert.equal(accepted.status, 202);

  const event = await broadcastPromise;
  assert.equal(event.type, 'gesture');
  assert.equal(event.gesture, 'wave');
  assert.equal(event.confidence, 0.93);

  const latestResponse = await fetch(`${baseUrl}/v1/gesture/latest`);
  const latest = await latestResponse.json();
  assert.equal(latest.ok, true);
  assert.equal(latest.event.event_id, event.event_id);

  const telemetryBroadcast = nextJson(socket);
  const telemetryResponse = await fetch(`${baseUrl}/v1/telemetry`, {
    method: 'POST',
    headers: {
      authorization: 'Bearer test-secret',
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      quaternion: { w: 1, x: 0, y: 0, z: 0 },
      euler_deg: { roll: 2.1, pitch: -3.2, yaw: 8.4 },
      accel_g: { x: 0.01, y: -0.02, z: 1.01 },
      gyro_dps: { x: 1.2, y: -0.8, z: 2.4 },
      linear_accel_g: { x: 0.01, y: -0.02, z: 0.01 },
      gyro_bias_dps: { x: 0.1, y: -0.1, z: 0.2 },
      stationary: false,
      sample_rate_hz: 100,
      sequence: 42,
      device_timestamp_ms: 1234,
      source: 'test-ring',
    }),
  });
  assert.equal(telemetryResponse.status, 202);

  const telemetry = await telemetryBroadcast;
  assert.equal(telemetry.type, 'telemetry');
  assert.equal(telemetry.sample_rate_hz, 100);
  assert.equal(telemetry.sequence, 42);
  assert.deepEqual(telemetry.euler_deg, { roll: 2.1, pitch: -3.2, yaw: 8.4 });

  const latestTelemetryResponse = await fetch(`${baseUrl}/v1/telemetry/latest`);
  const latestTelemetry = await latestTelemetryResponse.json();
  assert.equal(latestTelemetry.ok, true);
  assert.equal(latestTelemetry.event.event_id, telemetry.event_id);
});
