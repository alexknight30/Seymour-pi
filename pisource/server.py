"""
Flask-based MJPEG streaming server.

Provides:
  GET /            - HTML page showing the video stream
  GET /stream.mjpg - Multipart MJPEG stream
  GET /health      - JSON health check endpoint
"""

import time
from flask import Flask, Response, jsonify, render_template_string

from . import config
from .camera import create_camera, CameraError

app = Flask(__name__)

# Global camera instance (initialized in run_server)
_camera = None


# HTML template for the index page
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Pi Camera Stream</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: monospace;
            background: #000;
            color: #fff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            margin-bottom: 20px;
            font-weight: normal;
            letter-spacing: 2px;
        }
        .stream-container {
            border: 1px solid #333;
            background: #111;
            padding: 10px;
        }
        img {
            display: block;
            max-width: 100%;
            height: auto;
        }
        .info {
            margin-top: 20px;
            color: #666;
            font-size: 12px;
        }
        .info a {
            color: #888;
        }
        .status {
            margin-top: 10px;
            padding: 10px;
            background: #111;
            border: 1px solid #333;
            font-size: 12px;
        }
        .ok { color: #0f0; }
        .error { color: #f00; }
    </style>
</head>
<body>
    <h1>PISOURCE STREAM</h1>
    <div class="stream-container">
        <img src="/stream.mjpg" alt="Camera Stream" id="stream">
    </div>
    <div class="info">
        <p>Resolution: {{ width }}x{{ height }} @ {{ fps }}fps</p>
        <p>Stream URL: <a href="/stream.mjpg">/stream.mjpg</a></p>
        <p>Health: <a href="/health">/health</a></p>
    </div>
    <div class="status" id="status">
        Connecting...
    </div>
    <script>
        const img = document.getElementById('stream');
        const status = document.getElementById('status');
        
        img.onload = function() {
            status.innerHTML = '<span class="ok">STREAMING</span>';
        };
        
        img.onerror = function() {
            status.innerHTML = '<span class="error">STREAM ERROR - Retrying...</span>';
            setTimeout(() => {
                img.src = '/stream.mjpg?' + Date.now();
            }, 2000);
        };
        
        // Periodically check health
        setInterval(async () => {
            try {
                const resp = await fetch('/health');
                const data = await resp.json();
                if (data.status === 'ok') {
                    status.innerHTML = '<span class="ok">STREAMING</span> | Frames: ' + (data.frames || 'N/A');
                }
            } catch (e) {
                // Ignore fetch errors
            }
        }, 5000);
    </script>
</body>
</html>
"""


def generate_frames():
    """
    Generator that yields MJPEG frames for streaming.
    
    Yields frames in multipart/x-mixed-replace format.
    """
    global _camera
    
    frame_interval = config.FRAME_INTERVAL
    last_frame_time = 0
    
    while True:
        # Rate limiting
        now = time.time()
        elapsed = now - last_frame_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        
        # Get frame from camera
        if _camera is None:
            print("[server] Camera not initialized")
            time.sleep(1)
            continue
        
        frame = _camera.get_frame(timeout=2.0)
        
        if frame is None:
            print("[server] No frame available")
            time.sleep(0.1)
            continue
        
        last_frame_time = time.time()
        
        # Yield frame in MJPEG format
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n'
            b'Content-Length: ' + str(len(frame)).encode() + b'\r\n'
            b'\r\n' + frame + b'\r\n'
        )


@app.route('/')
def index():
    """Serve the main page with embedded video stream."""
    return render_template_string(
        INDEX_HTML,
        width=config.WIDTH,
        height=config.HEIGHT,
        fps=config.FPS
    )


@app.route('/stream.mjpg')
def stream():
    """
    MJPEG video stream endpoint.
    
    Returns a multipart/x-mixed-replace response with continuous JPEG frames.
    """
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'keep-alive'
        }
    )


@app.route('/health')
def health():
    """
    Health check endpoint.
    
    Returns JSON with server and camera status.
    """
    global _camera
    
    camera_status = "not_initialized"
    camera_stats = {}
    
    if _camera is not None:
        try:
            camera_stats = _camera.get_stats()
            camera_status = "ok" if _camera.is_running() else "stopped"
        except Exception as e:
            camera_status = f"error: {e}"
    
    return jsonify({
        "status": "ok",
        "camera": camera_status,
        "config": {
            "width": config.WIDTH,
            "height": config.HEIGHT,
            "fps": config.FPS,
            "quality": config.JPEG_QUALITY
        },
        "stats": camera_stats
    })


def run_server(use_dummy_camera: bool = False):
    """
    Initialize camera and start the Flask server.
    
    Args:
        use_dummy_camera: If True, use dummy camera for testing
    """
    global _camera
    
    print("=" * 50)
    print("  PISOURCE - Raspberry Pi Camera Stream")
    print("=" * 50)
    
    config.print_config()
    print()
    
    # Initialize camera
    print("[server] Initializing camera...")
    try:
        _camera = create_camera(use_dummy=use_dummy_camera)
        _camera.start()
    except CameraError as e:
        print(f"[server] Camera error: {e}")
        print("[server] Starting with dummy camera for testing...")
        _camera = create_camera(use_dummy=True)
        _camera.start()
    
    print()
    print(f"[server] Starting HTTP server on http://{config.HOST}:{config.PORT}")
    print(f"[server] Stream URL: http://<PI_IP>:{config.PORT}/stream.mjpg")
    print(f"[server] Health URL: http://<PI_IP>:{config.PORT}/health")
    print()
    print("[server] Press Ctrl+C to stop")
    print()
    
    try:
        # Run Flask server
        # Using threaded=True for handling multiple stream connections
        app.run(
            host=config.HOST,
            port=config.PORT,
            threaded=True,
            debug=False
        )
    except KeyboardInterrupt:
        print("\n[server] Shutting down...")
    finally:
        if _camera is not None:
            _camera.stop()
        print("[server] Stopped")

