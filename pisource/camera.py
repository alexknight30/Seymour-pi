"""
Picamera2 wrapper for low-latency JPEG frame capture.

This module provides a Camera class that:
  - Initializes Picamera2 with configurable resolution
  - Captures frames continuously in a background thread
  - Encodes frames as JPEG bytes
  - Provides the latest JPEG quickly without blocking
"""

import io
import threading
import time
from typing import Optional

from . import config

# Picamera2 is only available on Raspberry Pi OS with libcamera
try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class CameraError(Exception):
    """Raised when camera operations fail."""
    pass


class StreamingOutput(io.BufferedIOBase):
    """
    A file-like output that stores the latest JPEG frame.
    Used by Picamera2's encoder to write frames.
    """
    
    def __init__(self):
        self.frame: Optional[bytes] = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.frame_count = 0
        self.last_frame_time = 0.0
    
    def write(self, buf: bytes) -> int:
        with self.lock:
            self.frame = buf
            self.frame_count += 1
            self.last_frame_time = time.time()
            self.condition.notify_all()
        return len(buf)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Get the latest frame, waiting up to timeout seconds if none available.
        Returns None if timeout expires without a frame.
        """
        with self.lock:
            if self.frame is None:
                self.condition.wait(timeout=timeout)
            return self.frame


class Camera:
    """
    Wrapper around Picamera2 for MJPEG streaming.
    
    Usage:
        camera = Camera()
        camera.start()
        
        # In your streaming loop:
        jpeg_bytes = camera.get_frame()
        
        # When done:
        camera.stop()
    """
    
    def __init__(
        self,
        width: int = None,
        height: int = None,
        fps: int = None,
        quality: int = None
    ):
        if not PICAMERA2_AVAILABLE:
            raise CameraError(
                "Picamera2 is not installed or not available.\n\n"
                "This module requires a Raspberry Pi with:\n"
                "  1. Raspberry Pi OS (Bullseye or later)\n"
                "  2. A connected camera module\n"
                "  3. Picamera2 library installed\n\n"
                "To install on Raspberry Pi OS:\n"
                "  sudo apt update\n"
                "  sudo apt install -y python3-picamera2\n\n"
                "Or in a virtual environment:\n"
                "  pip install picamera2\n\n"
                "Make sure your camera is enabled:\n"
                "  sudo raspi-config -> Interface Options -> Camera -> Enable\n"
                "  (reboot after enabling)\n\n"
                "Test with: libcamera-hello"
            )
        
        self.width = width or config.WIDTH
        self.height = height or config.HEIGHT
        self.fps = fps or config.FPS
        self.quality = quality or config.JPEG_QUALITY
        
        self._picam2: Optional[Picamera2] = None
        self._output: Optional[StreamingOutput] = None
        self._running = False
    
    def start(self):
        """Initialize and start the camera."""
        if self._running:
            print("[camera] Already running")
            return
        
        print(f"[camera] Initializing Picamera2 ({self.width}x{self.height} @ {self.fps}fps)")
        
        try:
            self._picam2 = Picamera2()
            
            # Configure for video streaming
            video_config = self._picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.fps}
            )
            self._picam2.configure(video_config)
            
            # Set up streaming output
            self._output = StreamingOutput()
            
            # Create MJPEG encoder
            encoder = MJPEGEncoder()
            encoder.output = FileOutput(self._output)
            
            # Start camera and encoder
            self._picam2.start()
            self._picam2.start_encoder(encoder)
            
            self._running = True
            print("[camera] Started successfully")
            
        except Exception as e:
            self._cleanup()
            raise CameraError(f"Failed to start camera: {e}")
    
    def stop(self):
        """Stop the camera and release resources."""
        if not self._running:
            return
        
        print("[camera] Stopping...")
        self._cleanup()
        print("[camera] Stopped")
    
    def _cleanup(self):
        """Clean up camera resources."""
        self._running = False
        if self._picam2:
            try:
                self._picam2.stop_encoder()
            except Exception:
                pass
            try:
                self._picam2.stop()
            except Exception:
                pass
            try:
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None
        self._output = None
    
    def get_frame(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Get the latest JPEG frame.
        
        Args:
            timeout: Max seconds to wait for a frame
            
        Returns:
            JPEG bytes or None if no frame available
        """
        if not self._running or self._output is None:
            return None
        return self._output.get_frame(timeout=timeout)
    
    def is_running(self) -> bool:
        """Check if camera is currently running."""
        return self._running
    
    def get_stats(self) -> dict:
        """Get camera statistics."""
        if self._output is None:
            return {"running": False, "frames": 0}
        return {
            "running": self._running,
            "frames": self._output.frame_count,
            "last_frame_time": self._output.last_frame_time,
            "resolution": f"{self.width}x{self.height}",
            "target_fps": self.fps
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# Fallback for testing without a camera
class DummyCamera:
    """
    A dummy camera for testing without hardware.
    Generates simple test pattern frames.
    """
    
    def __init__(self, width: int = None, height: int = None, fps: int = None, **kwargs):
        self.width = width or config.WIDTH
        self.height = height or config.HEIGHT
        self.fps = fps or config.FPS
        self._running = False
        self._frame_count = 0
        self._last_frame_time = 0.0
    
    def start(self):
        print("[dummy-camera] Starting dummy camera (no real hardware)")
        self._running = True
    
    def stop(self):
        print("[dummy-camera] Stopping")
        self._running = False
    
    def get_frame(self, timeout: float = 1.0) -> Optional[bytes]:
        if not self._running:
            return None
        
        # Generate a simple test pattern
        try:
            import numpy as np
            from PIL import Image
            
            # Create a simple gradient with frame counter
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Gradient background
            for y in range(self.height):
                img[y, :, 0] = int(255 * y / self.height)  # Red gradient
                img[y, :, 2] = int(255 * (1 - y / self.height))  # Blue gradient
            
            # Add some variation based on frame count
            t = self._frame_count % 100
            img[:, :, 1] = int(128 + 127 * (t / 100))  # Green varies
            
            # Convert to JPEG
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=80)
            
            self._frame_count += 1
            self._last_frame_time = time.time()
            
            # Simulate frame rate
            time.sleep(1.0 / self.fps)
            
            return buffer.getvalue()
            
        except ImportError:
            # If numpy/PIL not available, return minimal JPEG
            print("[dummy-camera] numpy/PIL not available for test frames")
            return None
    
    def is_running(self) -> bool:
        return self._running
    
    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "frames": self._frame_count,
            "last_frame_time": self._last_frame_time,
            "resolution": f"{self.width}x{self.height}",
            "target_fps": self.fps,
            "dummy": True
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def create_camera(use_dummy: bool = False, **kwargs):
    """
    Factory function to create appropriate camera instance.
    
    Args:
        use_dummy: Force use of dummy camera for testing
        **kwargs: Passed to camera constructor
        
    Returns:
        Camera or DummyCamera instance
    """
    if use_dummy:
        return DummyCamera(**kwargs)
    
    if not PICAMERA2_AVAILABLE:
        print("[camera] Picamera2 not available, using dummy camera")
        return DummyCamera(**kwargs)
    
    return Camera(**kwargs)

