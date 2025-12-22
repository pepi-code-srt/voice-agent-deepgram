# Bug Fix: WebSocket Buffer Overflow

## Symptom
During high traffic (500+ users), the server would crash with OutOfMemory errors or drop WebSocket connections unexpectedly.

## Root Cause
Unbounded WebSocket receive queue. Fast producers (audio stream) overwhelmed the slow consumer (LLM processing), causing the buffer to grow primarily in memory until exhaustion.

## Fix
Implemented backpressure control with a bounded asyncio queue.

```python
# Buffer configuration
MAX_QUEUE_SIZE = 100
queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

async def producer(ws):
    while True:
        if queue.full():
            # Apply backpressure: pause reading from socket
            await asyncio.sleep(0.1)
            continue
        data = await ws.receive()
        await queue.put(data)
```

## Validation
Load test with 500 users sending continuous audio streams verified stable memory usage and 0 connection drops.
