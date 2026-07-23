# Ring Sound SDK integration

## Imported release

- SDK: `0.4.1`
- Protocol: voice ring v4
- Firmware documentation baseline: `V2.000.0001.0015`
- Source archive and checksums: [`SOURCE.json`](SOURCE.json)

Only the Python SDK and its protocol documentation are kept in this Web/API
repository. The Android demo APK and mechanical STEP/CAD files remain outside
the repository.

## 0.3.x to 0.4.1

The imported SDK changes the default BLE scan timeout to 25 seconds and always
writes NUS RX data in 20-byte chunks. `NusClient(write_chunk_size=...)` is no
longer supported:

```python
# 0.3.x
# transport = sdk.NusClient(address=address, write_chunk_size=20)

# 0.4.1
transport = sdk.NusClient(address=address)
```

The receive side still accepts notifications of arbitrary lengths and rebuilds
v4 packets through `PacketStream`.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r sdk/requirements.txt
python sdk/ring_sound.py scan
```

Audio decoding additionally requires `ffmpeg`. BLE access runs on the machine
physically near the ring; an AWS EC2 instance normally has no Bluetooth radio.
The planned production data path is:

```text
Ring -> local Python BLE bridge -> WebSocket/API -> AWS ring-web
```

## IMU dataset bridge

Put the ring into gesture mode before starting real-time IMU reporting:

```bash
python sdk/examples/capture_imu_jsonl.py \
  --address YOUR_RING_ADDRESS \
  --label rotate_front \
  --duration 30 \
  --output sdk/captures/rotate-front-001.jsonl
```

Each JSONL row uses schema `ring-imu/v1` and stores one raw six-axis sample,
its device timestamp, sequence number, reported sensor ranges, SDK version,
label, and session ID. Do not mix multiple people or recording sessions into a
single train/test split; split by `session_id` to avoid data leakage.

The firmware HMM event (`0x0702`) can be monitored without enabling real-time
`0x0605` reports:

```bash
python sdk/examples/hmm_gesture_monitor.py --address YOUR_RING_ADDRESS
```
