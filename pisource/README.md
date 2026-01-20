# pisource - Raspberry Pi Camera MJPEG Stream

A low-latency MJPEG HTTP streaming server for Raspberry Pi camera modules.

## Features

- MJPEG streaming over HTTP on port 8000
- Simple web interface to view stream
- Health check endpoint
- Configurable resolution, FPS, and quality via environment variables
- Graceful error handling

## Requirements

- Raspberry Pi (3B+, 4, or 5 recommended)
- Raspberry Pi Camera Module (v2 or v3)
- Raspberry Pi OS (Bullseye or later, 64-bit recommended)
- Python 3.10+

## Quick Setup

### 1. Update System and Enable Camera

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Enable camera interface (if not already enabled)
# For Pi 5 / newer OS, this is often auto-enabled
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable
# Reboot when prompted

# Test camera is working
libcamera-hello
# You should see a preview window for 5 seconds
```

### 2. Install Dependencies

```bash
# Install system packages
sudo apt install -y python3-picamera2 python3-venv python3-pip

# Navigate to pisource directory
cd /path/to/Seymour-pi/pisource

# Create virtual environment
python3 -m venv .venv --system-site-packages

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

**Note:** The `--system-site-packages` flag is important! It allows the venv to access `picamera2` installed via apt.

### 3. Run the Server

```bash
# Make sure you're in pisource directory with venv activated
cd /path/to/Seymour-pi/pisource
source .venv/bin/activate

# Run the server
python3 -m pisource
```

### 4. Access the Stream

Find your Pi's IP address:

```bash
hostname -I
# Example output: 192.168.1.100
```

Open in browser:
- **Web interface:** `http://192.168.1.100:8000`
- **Direct stream:** `http://192.168.1.100:8000/stream.mjpg`
- **Health check:** `http://192.168.1.100:8000/health`

## Configuration

All settings can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PISOURCE_HOST` | `0.0.0.0` | Server bind address |
| `PISOURCE_PORT` | `8000` | Server port |
| `PISOURCE_FPS` | `15` | Target frames per second |
| `PISOURCE_WIDTH` | `640` | Frame width in pixels |
| `PISOURCE_HEIGHT` | `480` | Frame height in pixels |
| `PISOURCE_QUALITY` | `80` | JPEG quality (1-100) |

Example with custom settings:

```bash
PISOURCE_PORT=8080 PISOURCE_FPS=30 PISOURCE_WIDTH=1280 PISOURCE_HEIGHT=720 python3 -m pisource
```

## Command Line Options

```bash
python3 -m pisource --help     # Show help
python3 -m pisource --dummy    # Use dummy camera for testing (no hardware)
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | HTML page with embedded video |
| `/stream.mjpg` | GET | MJPEG video stream |
| `/health` | GET | JSON health status |

### Health Response Example

```json
{
  "status": "ok",
  "camera": "ok",
  "config": {
    "width": 640,
    "height": 480,
    "fps": 15,
    "quality": 80
  },
  "stats": {
    "running": true,
    "frames": 1234,
    "resolution": "640x480",
    "target_fps": 15
  }
}
```

## Troubleshooting

### Camera not detected

```bash
# Check if camera is connected
libcamera-hello

# If you see "No cameras available", try:
# 1. Check cable connection (silver contacts face HDMI port on Pi 4)
# 2. Make sure camera interface is enabled
# 3. Reboot
sudo reboot
```

### Permission denied

```bash
# Add user to video group
sudo usermod -aG video $USER
# Log out and back in for changes to take effect
```

### Low FPS or high latency

1. Reduce resolution: `PISOURCE_WIDTH=320 PISOURCE_HEIGHT=240`
2. Lower quality: `PISOURCE_QUALITY=60`
3. Check network - use ethernet instead of WiFi if possible
4. Close other applications using the camera

### Cannot connect from laptop

1. Check Pi's firewall:
   ```bash
   sudo ufw status
   # If active, allow port 8000:
   sudo ufw allow 8000/tcp
   ```

2. Check Pi and laptop are on same network

3. Try pinging the Pi from laptop:
   ```bash
   ping 192.168.1.100
   ```

### Stream works in browser but not in code

Make sure you're connecting to `/stream.mjpg` (not just `/`) when using OpenCV or other code that expects raw MJPEG.

## Running on Boot (Optional)

Create a systemd service for auto-start:

```bash
sudo nano /etc/systemd/system/pisource.service
```

Add:

```ini
[Unit]
Description=Pi Camera Stream
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Seymour-pi/pisource
Environment=PISOURCE_PORT=8000
ExecStart=/home/pi/Seymour-pi/pisource/.venv/bin/python3 -m pisource
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable pisource
sudo systemctl start pisource
sudo systemctl status pisource
```

## Development

### Testing without a Pi

On any machine (Windows/Mac/Linux), you can test the server with a dummy camera:

```bash
cd pisource
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python3 -m pisource --dummy
```

This generates test pattern frames instead of real camera input.

## License

MIT

