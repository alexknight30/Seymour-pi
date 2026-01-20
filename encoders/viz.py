"""
Visualization for live video and drift metrics.

Provides OpenCV window with video feed and drift overlay,
plus optional matplotlib drift plot.
"""

import time
from typing import Optional
import numpy as np
import cv2

from . import config


class LiveDisplay:
    """
    OpenCV-based live display with drift overlay.
    
    Shows:
    - Live video feed from stream
    - Current cosine similarity / drift value
    - FPS counter
    - Simple drift bar indicator
    
    Usage:
        display = LiveDisplay()
        display.start()
        
        for frame, similarity in data:
            if not display.update(frame, similarity):
                break  # User pressed 'q'
        
        display.stop()
    """
    
    WINDOW_NAME = "Seymour Vision Stream"
    
    def __init__(self):
        self._running = False
        self._last_time = 0.0
        self._fps = 0.0
        self._frame_count = 0
        
        # FPS smoothing
        self._fps_history = []
        self._fps_window = 30
    
    def start(self):
        """Initialize the display window."""
        print("[viz] Starting display window")
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 800, 600)
        self._running = True
        self._last_time = time.time()
    
    def stop(self):
        """Close the display window."""
        print("[viz] Closing display window")
        self._running = False
        cv2.destroyAllWindows()
    
    def update(
        self,
        frame: np.ndarray,
        similarity: float = 1.0,
        extra_text: str = ""
    ) -> bool:
        """
        Update display with new frame and metrics.
        
        Args:
            frame: BGR numpy array
            similarity: Cosine similarity (0-1)
            extra_text: Additional text to display
            
        Returns:
            False if user pressed 'q' to quit, True otherwise.
        """
        if not self._running:
            return False
        
        # Calculate FPS
        now = time.time()
        dt = now - self._last_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps_history.append(instant_fps)
            if len(self._fps_history) > self._fps_window:
                self._fps_history.pop(0)
            self._fps = sum(self._fps_history) / len(self._fps_history)
        self._last_time = now
        self._frame_count += 1
        
        # Create display frame with overlay
        display_frame = self._draw_overlay(frame.copy(), similarity, extra_text)
        
        # Show frame
        cv2.imshow(self.WINDOW_NAME, display_frame)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("[viz] Quit requested")
            return False
        
        return True
    
    def _draw_overlay(
        self,
        frame: np.ndarray,
        similarity: float,
        extra_text: str
    ) -> np.ndarray:
        """Draw metrics overlay on frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay background
        overlay = frame.copy()
        
        # Top bar background
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Colors (BGR)
        white = (255, 255, 255)
        gray = (128, 128, 128)
        
        # Drift color: green (stable) -> red (changing)
        drift = 1.0 - similarity
        drift_color = (
            int(255 * (1 - drift)),  # B
            int(255 * (1 - drift)),  # G
            int(255 * drift * 2) if drift < 0.5 else 255  # R
        )
        
        # FPS text
        fps_text = f"FPS: {self._fps:.1f}"
        cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
        
        # Similarity text
        sim_text = f"Similarity: {similarity:.4f}"
        cv2.putText(frame, sim_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
        
        # Drift text with color
        drift_text = f"Drift: {drift:.4f}"
        cv2.putText(frame, drift_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, drift_color, 1)
        
        # Drift bar (right side)
        bar_x = w - 60
        bar_w = 40
        bar_h = 200
        bar_y = 90
        
        # Bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), gray, 1)
        
        # Filled portion (drift amount)
        fill_h = int(bar_h * min(drift * 5, 1.0))  # Scale drift for visibility
        if fill_h > 0:
            cv2.rectangle(
                frame,
                (bar_x + 2, bar_y + bar_h - fill_h),
                (bar_x + bar_w - 2, bar_y + bar_h - 2),
                drift_color,
                -1
            )
        
        # Bar label
        cv2.putText(frame, "DRIFT", (bar_x - 5, bar_y + bar_h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 1)
        
        # Extra text (bottom)
        if extra_text:
            cv2.putText(frame, extra_text, (10, h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray, 1)
        
        # Frame counter (bottom right)
        count_text = f"Frame: {self._frame_count}"
        cv2.putText(frame, count_text, (w - 150, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray, 1)
        
        return frame
    
    @property
    def is_running(self) -> bool:
        return self._running


class DriftPlotter:
    """
    Optional matplotlib-based drift plot.
    
    Shows rolling history of drift values over time.
    Note: May cause lag on some systems. Use with caution.
    """
    
    def __init__(self, window_size: int = None):
        self.window_size = window_size or config.ROLLING_WINDOW_SIZE
        self._fig = None
        self._ax = None
        self._line = None
        self._available = False
    
    def start(self):
        """Initialize the plot window."""
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use Tk backend for live updates
            import matplotlib.pyplot as plt
            
            plt.ion()  # Interactive mode
            self._fig, self._ax = plt.subplots(figsize=(8, 3))
            self._ax.set_xlim(0, self.window_size)
            self._ax.set_ylim(0, 0.5)
            self._ax.set_xlabel('Frame')
            self._ax.set_ylabel('Drift (1 - similarity)')
            self._ax.set_title('Embedding Drift Over Time')
            self._ax.grid(True, alpha=0.3)
            
            self._line, = self._ax.plot([], [], 'b-', linewidth=1)
            self._fig.tight_layout()
            
            self._available = True
            print("[viz] Drift plot initialized")
            
        except Exception as e:
            print(f"[viz] Could not initialize plot: {e}")
            print("[viz] Continuing without drift plot")
            self._available = False
    
    def update(self, drift_history: np.ndarray):
        """Update the plot with new drift values."""
        if not self._available or self._line is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            x = np.arange(len(drift_history))
            y = 1.0 - drift_history  # Convert similarity to drift
            
            self._line.set_data(x, y)
            
            # Adjust y-axis if needed
            if len(y) > 0:
                max_drift = max(0.1, np.max(y) * 1.2)
                self._ax.set_ylim(0, max_drift)
            
            self._ax.set_xlim(0, max(self.window_size, len(x)))
            
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            
        except Exception as e:
            print(f"[viz] Plot update error: {e}")
    
    def stop(self):
        """Close the plot window."""
        if self._available:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception:
                pass
        self._available = False


def simple_drift_meter(similarity: float, width: int = 40) -> str:
    """
    Create a simple text-based drift meter.
    
    Args:
        similarity: Cosine similarity (0-1)
        width: Character width of the meter
        
    Returns:
        String like "[=========>          ] 0.1234"
    """
    drift = 1.0 - similarity
    filled = int(width * min(drift * 5, 1.0))  # Scale for visibility
    empty = width - filled
    
    bar = '=' * filled + '>' + ' ' * empty if filled < width else '=' * width
    return f"[{bar}] {drift:.4f}"

