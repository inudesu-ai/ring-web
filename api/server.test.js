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
      recognition_source: 'mlp',
      circle_metrics: {
        label: 'circle',
        confidence: 0.9,
        path_length_m: 0.52,
        radius_m: 0.08,
        turn_radians: 6.1,
        closure_ratio: 0.12,
        plane_score: 0.96,
        roundness: 0.82,
      },
    }),
  });
  assert.equal(accepted.status, 202);

  const event = await broadcastPromise;
  assert.equal(event.type, 'gesture');
  assert.equal(event.gesture, 'wave');
  assert.equal(event.confidence, 0.93);
  assert.equal(event.recognition_source, 'mlp');
  assert.equal(event.circle_metrics.turn_radians, 6.1);
  assert.equal(event.circle_metrics.roundness, 0.82);

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
      gyro_raw_dps: { x: 1.3, y: -0.9, z: 2.6 },
      stationary: false,
      stationary_confidence: 0.42,
      calibrated: true,
      motion: {
        armed: true,
        moving: true,
        rotating_only: false,
        translation_candidate: false,
        position_m: [0.02, -0.01, 0.03],
        velocity_mps: [0.1, 0.0, 0.05],
        linear_accel_world_g: [0.1, 0.0, 0.05],
        corrected_accel_world_g: [0.08, 0.0, 0.04],
        accel_bias_world_g: [0.002, -0.001, 0.003],
        accel_threshold_g: 0.02,
        noise_sigma_g: 0.003,
        speed_mps: 0.112,
        distance_m: 0.04,
        segment_id: 2,
        segment_elapsed_s: 0.3,
        zupt_count: 1,
        zupt_confidence: 0.42,
        confidence: 0.93,
      },
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
  assert.equal(telemetry.calibrated, true);
  assert.equal(telemetry.motion.segment_id, 2);
  assert.deepEqual(telemetry.motion.position_m, { x: 0.02, y: -0.01, z: 0.03 });
  assert.deepEqual(telemetry.euler_deg, { roll: 2.1, pitch: -3.2, yaw: 8.4 });

  const latestTelemetryResponse = await fetch(`${baseUrl}/v1/telemetry/latest`);
  const latestTelemetry = await latestTelemetryResponse.json();
  assert.equal(latestTelemetry.ok, true);
  assert.equal(latestTelemetry.event.event_id, telemetry.event_id);
});
