# Speech to Speech

Real-time conversational AI powered by Deepgram's Voice Agent API. Talk naturally with an AI agent using ultra-low latency speech-to-speech technology.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-SocketIO-green.svg)
![Deepgram](https://img.shields.io/badge/Deepgram-Voice%20AI-purple.svg)

## ✨ Features

- **🎤 Real-time Voice Conversation** - Speak naturally and get instant AI responses
- **⚡ Ultra-Low Latency** - Sub-300ms response times with streaming audio
- **🗣️ 30+ Voice Models** - Choose from various AI voices and accents
- **📊 Live Latency Display** - Monitor STT, LLM, and TTS timing in real-time
- **🎵 Audio Visualizer** - Circular waveform that reacts to your voice
- **📥 Conversation Export** - Download chat history as JSON or TXT
- **🌓 Dark/Light Theme** - Beautiful UI with theme toggle
- **👥 Multi-User Support** - Session-based voice agents for concurrent users
- **🚦 Rate Limiting** - Built-in protection against API abuse

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Deepgram API Key](https://console.deepgram.com/)
- Microphone access

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Speech-to-Speech.git
cd Speech-to-Speech

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
DEEPGRAM_API_KEY=your_api_key_here
```

### Run the Application

```bash
python client.py
```

Open your browser to `http://127.0.0.1:5000` and click the microphone to start talking!

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │────▶│  Flask App  │────▶│  Deepgram   │
│  (WebAudio) │◀────│  (SocketIO) │◀────│  Voice API  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │
       ▼                   ▼
  Audio Capture      Voice Agent
  Audio Playback     Function Calling
```

## 📁 Project Structure

```
Speech-to-Speech/
├── client.py              # Main Flask application
├── static/
│   ├── style.css          # UI styles
│   └── favicon.svg        # Site favicon
├── templates/
│   └── index.html         # Frontend UI
├── common/
│   ├── agent_functions.py # Function calling handlers
│   ├── agent_templates.py # Voice agent configurations
│   └── business_logic.py  # Business logic & data
├── src/                   # Additional modules
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables
```

## 🎛️ Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `DEEPGRAM_API_KEY` | Your Deepgram API key | Required |
| `SECRET_KEY` | Flask session secret | Auto-generated |

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application UI |
| `/tts-models` | GET | Available voice models |
| `/industries` | GET | Available agent personas |

## 📡 Socket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `start_voice_agent` | Client → Server | Start voice session |
| `stop_voice_agent` | Client → Server | Stop voice session |
| `audio_data` | Client → Server | Stream audio chunks |
| `conversation_update` | Server → Client | New message |
| `audio_output` | Server → Client | TTS audio chunks |
| `latency_update` | Server → Client | Performance metrics |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Deepgram](https://deepgram.com/) - Voice AI Platform
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/) - WebSocket support
