"""
MJPEG stream receiver.

Connects to an MJPEG HTTP stream and yields decoded frames as numpy arrays.
Handles reconnection on connection loss.
"""

import time
from typing import Generator, Optional
import urllib.request
import urllib.error

import numpy as np
import cv2

from . import config


class StreamError(Exception):
    """Raised when stream operations fail."""
    pass


class MJPEGReceiver:
    """
    Receives and decodes MJPEG stream from HTTP endpoint.
    
    Usage:
        receiver = MJPEGReceiver("http://192.168.1.100:8000/stream.mjpg")
        for frame in receiver.frames():
            # frame is a numpy array (H, W, 3) in BGR format
            process(frame)
    """
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self._stream = None
        self._boundary = None
        self._running = False
        self._frame_count = 0
        self._last_frame_time = 0.0
        self._connected = False
    
    def connect(self) -> bool:
        """
        Establish connection to the MJPEG stream.
        
        Returns:
            True if connected successfully, False otherwise.
        """
        print(f"[receiver] Connecting to {self.stream_url}")
        
        try:
            # Set up request with timeout
            request = urllib.request.Request(self.stream_url)
            self._stream = urllib.request.urlopen(
                request, 
                timeout=config.FRAME_TIMEOUT
            )
            
            # Parse content-type to find boundary
            content_type = self._stream.headers.get('Content-Type', '')
            
            if 'multipart' not in content_type:
                print(f"[receiver] Warning: Expected multipart content, got: {content_type}")
            
            # Extract boundary (usually "frame" for our pisource server)
            if 'boundary=' in content_type:
                self._boundary = content_type.split('boundary=')[-1].strip()
            else:
                self._boundary = 'frame'  # Default for our server
            
            self._connected = True
            print(f"[receiver] Connected (boundary={self._boundary})")
            return True
            
        except urllib.error.URLError as e:
            print(f"[receiver] Connection failed: {e.reason}")
            self._connected = False
            return False
        except Exception as e:
            print(f"[receiver] Connection error: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Close the stream connection."""
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._connected = False
    
    def _read_frame(self) -> Optional[bytes]:
        """
        Read a single JPEG frame from the multipart stream.
        
        Returns:
            JPEG bytes or None if read failed.
        """
        if not self._stream:
            return None
        
        try:
            # Read until we find frame boundary
            # MJPEG format: --boundary\r\nheaders\r\n\r\njpeg_data\r\n
            
            # Skip to boundary
            line = b''
            while True:
                byte = self._stream.read(1)
                if not byte:
                    return None
                line += byte
                if line.endswith(b'\r\n'):
                    if b'--' in line:
                        break
                    line = b''
                if len(line) > 1000:  # Safety limit
                    line = line[-100:]
            
            # Read headers until empty line
            content_length = 0
            while True:
                line = b''
                while True:
                    byte = self._stream.read(1)
                    if not byte:
                        return None
                    line += byte
                    if line.endswith(b'\r\n'):
                        break
                
                line = line.strip()
                if not line:  # Empty line = end of headers
                    break
                
                # Parse Content-Length if present
                if line.lower().startswith(b'content-length:'):
                    try:
                        content_length = int(line.split(b':')[1].strip())
                    except (ValueError, IndexError):
                        pass
            
            # Read JPEG data
            if content_length > 0:
                jpeg_data = self._stream.read(content_length)
            else:
                # No content-length, read until next boundary
                # This is less efficient but more robust
                jpeg_data = b''
                while True:
                    byte = self._stream.read(1)
                    if not byte:
                        break
                    jpeg_data += byte
                    # Check for JPEG end marker + boundary start
                    if jpeg_data.endswith(b'\xff\xd9'):
                        # Read potential boundary
                        peek = self._stream.read(4)
                        if peek.startswith(b'\r\n--'):
                            # Put back for next frame
                            break
                        jpeg_data += peek
                    if len(jpeg_data) > 10_000_000:  # 10MB safety limit
                        print("[receiver] Frame too large, skipping")
                        return None
            
            return jpeg_data if jpeg_data else None
            
        except Exception as e:
            print(f"[receiver] Read error: {e}")
            return None
    
    def _decode_frame(self, jpeg_data: bytes) -> Optional[np.ndarray]:
        """
        Decode JPEG bytes to numpy array.
        
        Args:
            jpeg_data: JPEG image bytes
            
        Returns:
            BGR numpy array (H, W, 3) or None if decode failed.
        """
        try:
            # Decode JPEG
            arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("[receiver] Failed to decode JPEG")
                return None
            
            return frame
            
        except Exception as e:
            print(f"[receiver] Decode error: {e}")
            return None
    
    def frames(self, max_reconnects: int = -1) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields decoded frames.
        
        Args:
            max_reconnects: Max reconnection attempts (-1 for infinite)
            
        Yields:
            BGR numpy arrays (H, W, 3)
        """
        self._running = True
        reconnect_count = 0
        
        while self._running:
            # Connect if needed
            if not self._connected:
                if max_reconnects >= 0 and reconnect_count >= max_reconnects:
                    print(f"[receiver] Max reconnects ({max_reconnects}) reached")
                    break
                
                if not self.connect():
                    reconnect_count += 1
                    print(f"[receiver] Reconnecting in {config.RECONNECT_DELAY}s... (attempt {reconnect_count})")
                    time.sleep(config.RECONNECT_DELAY)
                    continue
                
                reconnect_count = 0
            
            # Read and decode frame
            jpeg_data = self._read_frame()
            
            if jpeg_data is None:
                print("[receiver] Lost connection")
                self.disconnect()
                continue
            
            frame = self._decode_frame(jpeg_data)
            
            if frame is not None:
                self._frame_count += 1
                self._last_frame_time = time.time()
                yield frame
    
    def stop(self):
        """Stop the frame generator."""
        self._running = False
        self.disconnect()
    
    def get_stats(self) -> dict:
        """Get receiver statistics."""
        return {
            "connected": self._connected,
            "frame_count": self._frame_count,
            "last_frame_time": self._last_frame_time,
            "stream_url": self.stream_url
        }


def test_connection(stream_url: str) -> bool:
    """
    Test if stream URL is reachable.
    
    Args:
        stream_url: MJPEG stream URL
        
    Returns:
        True if connection successful, False otherwise.
    """
    receiver = MJPEGReceiver(stream_url)
    success = receiver.connect()
    receiver.disconnect()
    return success

