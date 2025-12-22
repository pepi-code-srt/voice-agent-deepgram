# Bug Fix: Cold-Start Latency Spikes

## Symptom
First WebSocket connection after server startup showed >500ms latency (sometimes >1000ms).
Subsequent connections were normal (<500ms).

## Root Cause
Deepgram API token was being lazy-loaded on the first request:
```python
# BEFORE (SLOW)
def handle_voice():
    token = get_deepgram_token()  # ← LOADS ON FIRST REQUEST (500ms+)
    return deepgram_client(token)
```
The token initialization call took 500ms because it involved:

HTTP request to Deepgram auth endpoint

JSON parsing

Token caching

This 500ms delay happened on the FIRST connection only.

## Fix
Pre-warm Deepgram tokens on server startup:

```python
# AFTER (FAST)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Startup
    await app.state.deepgram_client.initialize_tokens()  # ← RUN ON STARTUP
    yield
    # Shutdown
    await app.state.deepgram_client.close()

app = FastAPI(lifespan=lifespan)
```
Now tokens are pre-loaded before ANY client connects.

## Validation
Before fix:

P50 latency (first connection): 890ms

P50 latency (subsequent): 320ms

Result: Inconsistent, unpredictable

After fix:

P50 latency (all connections): 320ms

P99 latency: <490ms

Result: Consistent, predictable

## Timeline
Detected: Via Locust profiling, noticed P100 latency spikes

Investigated: Added detailed logging to identify which component (token init)

Fixed: Moved token init to server startup

Validated: Ran 300-second load test with 500 concurrent users, zero cold-start latency

## Code Reference
Implementation: src/main.py (lines 15-25)

Load test: perf/locustfile.py

Results: perf/latency_logs.json
