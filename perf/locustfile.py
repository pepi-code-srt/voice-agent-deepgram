"""
Locust load test for Voice Agent WebSocket
Tests sub-500ms latency at 500+ concurrent connections
"""

from locust import HttpUser, task, between, WebSocketUser
import json
import time
import random

class VoiceAgentUser(WebSocketUser):
    wait_time = between(1, 5)
    host = "http://localhost:8000"
    
    @task
    def send_voice_request(self):
        """Simulate voice input and measure latency"""
        start_time = time.time()
        
        try:
            with self.client_connect("/ws/voice") as ws:
                # Simulate binary audio frame (1KB of audio data)
                audio_frame = b"fake_audio_" * 100
                
                # Send request
                ws.send(audio_frame, binary=True)
                
                # Wait for response
                response = ws.receive(timeout=5)
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Log for analysis
                if response:
                    self.environment.stats.log_request(
                        "WebSocket", 
                        "/ws/voice", 
                        latency_ms
                    )
                    
        except Exception as e:
            self.environment.stats.log_error(
                "WebSocket", 
                "/ws/voice", 
                str(e)
            )

if __name__ == "__main__":
    # Run: locust -f locustfile.py --headless -u 500 -r 50 -t 300s
    pass
