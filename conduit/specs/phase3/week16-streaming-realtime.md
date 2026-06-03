# Week 16: Streaming and Real-Time Patterns

## Context

**Where it fits:** Phase 3, Week 16 — Platform Maturity + Portfolio
**Prerequisites:** Phases 1+2 complete (batch pipelines, model serving, monitoring operational). Week 15 CLI/SDK available for pipeline creation.
**What it builds on:** All prior work is batch-oriented — data arrives, pipelines run on schedule, models retrain periodically. This week adds event-driven processing for use cases requiring sub-second responses: real-time feature computation, online prediction, and streaming aggregations.

**Hardware:** ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu

---

## Learning Goals

- [ ] Understand event-driven architecture vs. batch processing tradeoffs
- [ ] Learn Redis Streams: consumer groups, acknowledgment, pending entries
- [ ] Study stream processing patterns: windowed aggregations, sessionization, watermarks
- [ ] Explore online feature stores and real-time feature computation
- [ ] Understand backpressure mechanisms and flow control
- [ ] Learn Lambda architecture: merge batch accuracy with streaming speed
- [ ] Study sub-100ms inference optimization: model warmup, connection pooling, caching

---

## Implementation Goals

- [ ] Deploy Redis Streams as event ingestion layer with consumer groups
- [ ] Build event-driven pipeline framework: source → transform → sink
- [ ] Implement real-time feature computation with sliding window aggregations
- [ ] Create online prediction service with feature lookup and sub-100ms latency
- [ ] Build windowed aggregation engine (tumbling, sliding, session windows)
- [ ] Implement backpressure handling with adaptive rate limiting
- [ ] Design Lambda architecture combining batch and streaming paths
- [ ] Build end-to-end demo: real-time fraud detection system
- [ ] Add stream monitoring: lag, throughput, error rates

---

## Acceptance Criteria

1. Events published to Redis Streams are consumed and processed within 50ms by at least one consumer in the group
2. Real-time feature computation updates user-level features (e.g., transaction count in last 5 minutes) within 100ms of event arrival
3. Online prediction endpoint returns fraud score in <100ms including feature lookup from the online feature store
4. Windowed aggregations correctly compute tumbling (fixed), sliding (overlapping), and session windows with <1% error vs. batch recomputation
5. Backpressure system throttles producers when consumers fall behind by >1000 messages, resuming when lag drops below 500
6. Lambda architecture produces consistent results: streaming approximation within 5% of batch ground truth for all aggregation queries
7. System handles 10,000 events/second sustained throughput on local hardware without message loss
8. Consumer group rebalancing completes within 5 seconds when a consumer crashes and rejoins
9. Stream monitoring dashboard shows real-time lag, throughput, and error rate with <5-second refresh
10. End-to-end fraud detection demo processes a transaction, computes features, scores the model, and returns a decision in under 200ms total

---

## Validation Commands

```bash
# Start Redis with Streams support
docker run -d --name conduit-redis -p 6379:6379 redis:7-alpine

# Verify Redis Streams
redis-cli PING
redis-cli XADD test-stream '*' key value
redis-cli XLEN test-stream

# Start the streaming pipeline
cd ~/conduit && python -m conduit.streaming.runner --config configs/streaming.yaml &
RUNNER_PID=$!

# Publish test events
python -m conduit.streaming.producer \
  --stream transactions \
  --rate 1000 \
  --duration 10

# Check consumer lag
redis-cli XINFO GROUPS transactions

# Verify real-time features updated
python -c "
from conduit.features.online import OnlineFeatureStore
store = OnlineFeatureStore()
features = store.get_features('user_123', ['tx_count_5m', 'tx_amount_avg_1h'])
print(f'Features: {features}')
assert features['tx_count_5m'] > 0
"

# Test online prediction latency
python -m conduit.streaming.benchmark \
  --endpoint http://localhost:8080/predict/fraud \
  --requests 1000 \
  --concurrency 10 \
  | grep "p99_latency"

# Verify windowed aggregations
python -m pytest tests/streaming/test_windows.py -v

# Test backpressure
python -m conduit.streaming.producer \
  --stream overload-test \
  --rate 50000 \
  --duration 5 2>&1 | grep "backpressure"

# Lambda architecture consistency check
python -m conduit.streaming.lambda_check \
  --batch-result data/batch_agg.parquet \
  --stream-result data/stream_agg.json \
  --tolerance 0.05

# Cleanup
kill $RUNNER_PID
docker stop conduit-redis
```

---

## Technical Implementation Details

### Event-Driven Pipeline Framework

```python
# src/conduit/streaming/pipeline.py
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Callable
from conduit.streaming.sources import RedisStreamSource
from conduit.streaming.sinks import Sink

@dataclass
class StreamEvent:
    stream: str
    event_id: str
    timestamp: float
    payload: dict

class StreamingPipeline:
    def __init__(self, name: str, source: RedisStreamSource):
        self.name = name
        self.source = source
        self._transforms: list[Callable] = []
        self._sinks: list[Sink] = []

    def transform(self, fn: Callable) -> "StreamingPipeline":
        self._transforms.append(fn)
        return self

    def sink(self, sink: Sink) -> "StreamingPipeline":
        self._sinks.append(sink)
        return self

    async def run(self):
        async for event in self.source.consume():
            try:
                result = event
                for transform in self._transforms:
                    result = await transform(result) if asyncio.iscoroutinefunction(transform) else transform(result)
                    if result is None:
                        break
                if result is not None:
                    for sink in self._sinks:
                        await sink.write(result)
                await self.source.ack(event)
            except Exception as e:
                await self._handle_error(event, e)

    async def _handle_error(self, event: StreamEvent, error: Exception):
        from conduit.streaming.dlq import DeadLetterQueue
        dlq = DeadLetterQueue(f"{self.name}-dlq")
        await dlq.send(event, error)
```

### Redis Streams Consumer with Consumer Groups

```python
# src/conduit/streaming/sources/redis_stream.py
import redis.asyncio as redis
from typing import AsyncIterator
from conduit.streaming.pipeline import StreamEvent

class RedisStreamSource:
    def __init__(self, stream: str, group: str, consumer: str, redis_url: str = "redis://localhost:6379"):
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self._redis = redis.from_url(redis_url)
        self._batch_size = 100
        self._block_ms = 1000

    async def initialize(self):
        try:
            await self._redis.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def consume(self) -> AsyncIterator[StreamEvent]:
        await self.initialize()
        while True:
            entries = await self._redis.xreadgroup(
                groupname=self.group,
                consumername=self.consumer,
                streams={self.stream: ">"},
                count=self._batch_size,
                block=self._block_ms,
            )
            for stream_name, messages in entries:
                for msg_id, fields in messages:
                    yield StreamEvent(
                        stream=stream_name.decode(),
                        event_id=msg_id.decode(),
                        timestamp=float(msg_id.decode().split("-")[0]) / 1000,
                        payload={k.decode(): v.decode() for k, v in fields.items()},
                    )

    async def ack(self, event: StreamEvent):
        await self._redis.xack(self.stream, self.group, event.event_id)
```

### Real-Time Feature Computation

```python
# src/conduit/features/online.py
import time
import redis
from typing import Optional

class OnlineFeatureStore:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis = redis.from_url(redis_url)
        self._ttl = 86400  # 24h default TTL

    def update_feature(self, entity_id: str, feature_name: str, value: float, timestamp: float = None):
        ts = timestamp or time.time()
        key = f"features:{entity_id}:{feature_name}"
        self._redis.zadd(key, {f"{ts}:{value}": ts})
        self._redis.expire(key, self._ttl)

    def get_windowed_count(self, entity_id: str, feature_name: str, window_seconds: int) -> int:
        key = f"features:{entity_id}:{feature_name}"
        now = time.time()
        start = now - window_seconds
        return self._redis.zcount(key, start, now)

    def get_windowed_avg(self, entity_id: str, feature_name: str, window_seconds: int) -> Optional[float]:
        key = f"features:{entity_id}:{feature_name}"
        now = time.time()
        start = now - window_seconds
        entries = self._redis.zrangebyscore(key, start, now)
        if not entries:
            return None
        values = [float(e.decode().split(":")[1]) for e in entries]
        return sum(values) / len(values)

    def get_features(self, entity_id: str, feature_names: list[str]) -> dict:
        result = {}
        pipe = self._redis.pipeline()
        for name in feature_names:
            if name.endswith("_5m"):
                result[name] = self.get_windowed_count(entity_id, name.replace("_5m", ""), 300)
            elif name.endswith("_1h"):
                result[name] = self.get_windowed_avg(entity_id, name.replace("_avg_1h", ""), 3600)
        return result
```

### Windowed Aggregation Engine

```python
# src/conduit/streaming/windows.py
import time
from dataclasses import dataclass, field
from typing import Callable
from collections import defaultdict

@dataclass
class WindowConfig:
    window_type: str  # "tumbling", "sliding", "session"
    size_seconds: float
    slide_seconds: float = 0  # for sliding windows
    gap_seconds: float = 0   # for session windows

@dataclass
class WindowState:
    start: float
    end: float
    values: list = field(default_factory=list)

class WindowAggregator:
    def __init__(self, config: WindowConfig, agg_fn: Callable):
        self.config = config
        self.agg_fn = agg_fn
        self._windows: dict[str, list[WindowState]] = defaultdict(list)

    def process(self, key: str, value: float, event_time: float) -> list[tuple[str, float, float]]:
        """Returns list of (key, result, window_end) for any closed windows."""
        results = []
        if self.config.window_type == "tumbling":
            results = self._process_tumbling(key, value, event_time)
        elif self.config.window_type == "sliding":
            results = self._process_sliding(key, value, event_time)
        elif self.config.window_type == "session":
            results = self._process_session(key, value, event_time)
        return results

    def _process_tumbling(self, key: str, value: float, event_time: float):
        window_start = (event_time // self.config.size_seconds) * self.config.size_seconds
        window_end = window_start + self.config.size_seconds
        windows = self._windows[key]
        current = next((w for w in windows if w.start == window_start), None)
        if current is None:
            current = WindowState(start=window_start, end=window_end)
            windows.append(current)
        current.values.append(value)

        closed = []
        for w in windows:
            if event_time >= w.end:
                closed.append((key, self.agg_fn(w.values), w.end))
        self._windows[key] = [w for w in windows if event_time < w.end]
        return closed
```

### Backpressure Handler

```python
# src/conduit/streaming/backpressure.py
import asyncio
import redis.asyncio as redis

class BackpressureController:
    def __init__(self, stream: str, group: str, high_watermark: int = 1000, low_watermark: int = 500):
        self.stream = stream
        self.group = group
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self._throttled = False
        self._redis = redis.from_url("redis://localhost:6379")

    async def check_lag(self) -> int:
        info = await self._redis.xinfo_groups(self.stream)
        for group_info in info:
            if group_info["name"].decode() == self.group:
                return group_info["lag"] or 0
        return 0

    async def should_throttle(self) -> bool:
        lag = await self.check_lag()
        if lag > self.high_watermark and not self._throttled:
            self._throttled = True
        elif lag < self.low_watermark and self._throttled:
            self._throttled = False
        return self._throttled

    async def wait_if_throttled(self):
        while await self.should_throttle():
            await asyncio.sleep(0.1)
```

### Project file structure:
```
~/conduit/src/conduit/streaming/
├── __init__.py
├── pipeline.py
├── runner.py
├── producer.py
├── benchmark.py
├── lambda_check.py
├── sources/
│   ├── redis_stream.py
│   └── kafka_source.py
├── sinks/
│   ├── redis_sink.py
│   ├── feature_store_sink.py
│   └── prediction_sink.py
├── windows.py
├── backpressure.py
└── dlq.py
~/conduit/src/conduit/features/
├── online.py
├── offline.py
└── sync.py
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Redis XREADGROUP returns empty despite messages in stream | Consumer group ID might be set to `$` (only new messages). Delete and recreate group with ID `0` to read from beginning |
| Consumer group not rebalancing after crash | Pending entries stay claimed. Use `XAUTOCLAIM` with a min-idle-time to reclaim stuck messages |
| Windowed aggregation results drift from batch | Check event-time vs processing-time semantics. Use event timestamps, not `time.time()` for window assignment |
| Backpressure causing producer timeouts | Implement circuit breaker pattern: after N throttle events, buffer to disk and replay later |
| Feature store reads too slow for <100ms target | Pipeline Redis commands with `pipeline()` for batch reads. Consider Redis Cluster for key sharding |
| Memory grows unbounded with session windows | Add max session duration and evict windows older than 2x the gap timeout |

---

## Agent Handoff Template

```
I'm building Week 16 of the Conduit ML platform: Streaming and Real-Time Patterns.

Current state:
- Batch ML pipelines fully operational from Phases 1+2
- CLI/SDK from Week 15 available for pipeline management
- Need to add event-driven processing for real-time use cases

What I need help with:
- [specific task: e.g., "implementing windowed aggregations with correct watermark handling"]

Key files:
- Streaming pipeline: src/conduit/streaming/pipeline.py
- Redis consumer: src/conduit/streaming/sources/redis_stream.py
- Online features: src/conduit/features/online.py
- Window engine: src/conduit/streaming/windows.py
- Backpressure: src/conduit/streaming/backpressure.py

Tech stack: Python 3.11, Redis Streams, asyncio, redis-py async
Hardware: RTX 5080 16GB, 32GB RAM, Ubuntu

The goal is event-driven ML pipelines with sub-100ms prediction latency
and real-time feature computation using streaming aggregations.
```

---

## Out of Scope

- Apache Kafka deployment (Redis Streams is simpler for local; Kafka adapter is a future plugin)
- Exactly-once semantics (at-least-once with idempotent consumers is sufficient)
- Multi-region streaming replication
- Complex event processing (CEP) engine
- Stream SQL (like ksqlDB or Flink SQL)
- Schema registry (Avro/Protobuf schemas for events)
- Production Kubernetes autoscaling for stream consumers
