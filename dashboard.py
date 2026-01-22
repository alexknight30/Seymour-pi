"""
Seymour Dashboard - Gradio-based control interface.

Run with: python dashboard.py
Opens at: http://localhost:7860

Features:
- Live video stream from Pi (auto-refreshes)
- Label input for manual labeling
- Generalization score tracking (did labels transfer to new examples?)
- CSV export of all data
- Auto-saves frames to data/images/ (1 per second)
"""

import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from language.generalization import GeneralizationTracker

# Ensure data directories exist
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)


class SeymourState:
    """Manages the state of the Seymour system."""
    
    def __init__(self):
        self.running = False
        self.frame_count = 0
        self.current_embedding = None
        self.last_message = ""  # For quick status messages
        
        # Frame saving (1 per second)
        self.last_frame_save_time = 0
        self.saved_frame_count = 0
        self.frame_save_interval = 1.0  # seconds
        
        # Generalization tracking (the key metric!)
        self.tracker = GeneralizationTracker(similarity_threshold=0.7)
        
        # For mock visualization
        self._mock_object_type = "unknown"
        self._mock_objects = ["tree", "rock", "person", "car", "unknown"]
    
    def generate_mock_frame(self) -> np.ndarray:
        """Generate a mock frame for testing (uses PIL, not OpenCV)."""
        w, h = 640, 480
        
        # Create base image with gradient
        img = Image.new('RGB', (w, h), color=(30, 30, 40))
        draw = ImageDraw.Draw(img)
        
        # Animated background gradient
        for y in range(0, h, 4):
            intensity = int(40 + 20 * np.sin(self.frame_count * 0.03 + y * 0.01))
            draw.line([(0, y), (w, y)], fill=(intensity, intensity, intensity + 10))
        
        # Draw a moving "object" (circle)
        cx = int(320 + 150 * np.sin(self.frame_count * 0.02))
        cy = int(240 + 80 * np.cos(self.frame_count * 0.025))
        radius = 60
        
        # Object color based on mock type
        colors = {
            "tree": (60, 140, 60),
            "rock": (120, 110, 100),
            "person": (180, 140, 120),
            "car": (100, 100, 180),
            "unknown": (150, 150, 150)
        }
        color = colors.get(self._mock_object_type, (150, 150, 150))
        
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=color,
            outline=(255, 255, 255),
            width=2
        )
        
        # Draw object label
        draw.text((cx - 30, cy - 10), self._mock_object_type.upper(), fill=(255, 255, 255))
        
        # Status overlay
        status_color = (0, 255, 0) if self.running else (255, 100, 100)
        status_text = "LIVE" if self.running else "STOPPED"
        draw.rectangle([w - 100, 10, w - 10, 40], fill=(0, 0, 0))
        draw.text((w - 90, 15), status_text, fill=status_color)
        
        # Frame counter
        draw.text((10, 10), f"Frame: {self.frame_count}", fill=(200, 200, 200))
        draw.text((10, 30), f"Saved: {self.saved_frame_count}", fill=(150, 150, 150))
        
        # Last message (quick feedback)
        if self.last_message:
            draw.rectangle([10, h - 45, w - 10, h - 10], fill=(0, 0, 0))
            draw.text((15, h - 40), self.last_message[:70], fill=(200, 200, 100))
        
        return img, np.array(img)
    
    def save_frame(self, img: Image.Image):
        """Save a frame to data/images/ if enough time has passed."""
        now = time.time()
        
        if now - self.last_frame_save_time >= self.frame_save_interval:
            self.saved_frame_count += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = IMAGES_DIR / f"frame_{self.saved_frame_count:05d}_{timestamp}.png"
            img.save(filename)
            self.last_frame_save_time = now
            return True
        return False
    
    def generate_mock_embedding(self) -> np.ndarray:
        """Generate a mock embedding that clusters by object type."""
        # Each object type has a base embedding
        np.random.seed(hash(self._mock_object_type) % 2**31)
        base = np.random.randn(512).astype(np.float32)
        
        # Add frame-specific noise
        np.random.seed(self.frame_count)
        noise = np.random.randn(512).astype(np.float32) * 0.15
        
        return base + noise
    
    def cycle_mock_object(self):
        """Change the mock object (for testing generalization)."""
        idx = self._mock_objects.index(self._mock_object_type)
        self._mock_object_type = self._mock_objects[(idx + 1) % len(self._mock_objects)]
        self.last_message = f"Object changed to: {self._mock_object_type}"


state = SeymourState()


def tick():
    """Called periodically to update the display."""
    if state.running:
        state.frame_count += 1
        state.current_embedding = state.generate_mock_embedding()
        
        # Auto-check generalization (if we have labels)
        if state.tracker.clusters:
            result = state.tracker.check_generalization(
                state.current_embedding, 
                frame_id=state.frame_count
            )
            # Update message with recognition result
            if result['best_label']:
                match_str = "YES" if result['generalized'] else "no"
                state.last_message = f"Sees '{result['best_label']}' ({result['best_score']:.0%}) - Match: {match_str}"
    
    return get_display_data()


def get_display_data():
    """Get all display data: frame, plot, stats."""
    pil_img, frame = state.generate_mock_frame()
    
    # Save frame if running (1 per second)
    if state.running:
        state.save_frame(pil_img)
    
    # Generate plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    scores = state.tracker.get_recent_scores(100)
    if scores:
        x = np.arange(len(scores))
        y = np.array(scores)
        ax.plot(x, y, color='#4CAF50', linewidth=1.5, label='Similarity Score')
        ax.fill_between(x, 0, y, alpha=0.3, color='#4CAF50')
        
        # Draw threshold line
        ax.axhline(y=state.tracker.similarity_threshold, color='#FF5722', 
                   linestyle='--', linewidth=1, label=f'Threshold ({state.tracker.similarity_threshold})')
        
        # Color points above/below threshold
        above = y >= state.tracker.similarity_threshold
        if any(above):
            ax.scatter(x[above], y[above], color='#4CAF50', s=20, zorder=5)
        if any(~above):
            ax.scatter(x[~above], y[~above], color='#FF5722', s=20, zorder=5)
    
    ax.set_xlim(0, max(100, len(scores)))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Check #', color='white', fontsize=10)
    ax.set_ylabel('Similarity', color='white', fontsize=10)
    ax.set_title('Generalization Score', color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='white')
    if scores:
        ax.legend(loc='lower right', facecolor='#333', edgecolor='white', labelcolor='white', fontsize=8)
    plt.tight_layout()
    
    # Status summary
    status_lines = [
        f"Running: {'Yes' if state.running else 'No'}",
        f"Frames processed: {state.frame_count}",
        f"Frames saved: {state.saved_frame_count}",
        f"",
        state.tracker.summary(),
        f"",
        f"Generalization rate: {state.tracker.get_generalization_rate():.0%}",
    ]
    status = "\n".join(status_lines)
    
    plt.close(fig)
    return frame, fig, status


def start_system():
    """Start the Seymour system."""
    state.running = True
    state.frame_count = 0
    state.last_frame_save_time = 0  # Reset to save first frame immediately
    state.last_message = "System started - encoding frames..."
    return gr.update(interactive=False), gr.update(interactive=True)


def stop_system():
    """Stop the Seymour system."""
    state.running = False
    state.last_message = "System stopped"
    return gr.update(interactive=True), gr.update(interactive=False)


def submit_label(label_text):
    """Submit a label for the current frame."""
    if not label_text.strip():
        state.last_message = "Please enter a label"
        return ""
    
    if state.current_embedding is None:
        state.last_message = "No embedding - start the system first"
        return ""
    
    label = label_text.strip().lower()
    
    # Add to generalization tracker
    state.tracker.add_label(
        embedding=state.current_embedding,
        label=label,
        frame_id=state.frame_count
    )
    
    count = state.tracker.clusters[label].count
    state.last_message = f"Labeled frame #{state.frame_count} as '{label}' (total: {count})"
    
    return ""  # Clear the input


def export_data():
    """Export all data to CSV files."""
    if not state.tracker.clusters:
        state.last_message = "No data to export - add labels first"
        return
    
    # Export CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    labels_file, history_file = state.tracker.export_csv(str(DATA_DIR / f"seymour_{timestamp}.csv"))
    state.tracker.export_embeddings_npz(str(DATA_DIR / f"seymour_{timestamp}_embeddings.npz"))
    
    state.last_message = f"Exported to data/seymour_{timestamp}_*.csv"


def change_mock_object():
    """Cycle through mock objects (for testing)."""
    state.cycle_mock_object()
    return f"Current: {state._mock_object_type}"


# Build the Gradio interface
with gr.Blocks(title="Seymour Dashboard") as demo:
    
    gr.Markdown("# SEYMOUR DASHBOARD")
    gr.Markdown("*Vision-Language Feedback Loop - Generalization Tracking*")
    
    with gr.Row():
        # Left: Video stream
        with gr.Column(scale=1):
            gr.Markdown("### Live Stream")
            video_display = gr.Image(
                label="Camera Feed",
                show_label=False,
                height=400,
            )
            # Mock controls (for testing without Pi)
            with gr.Row():
                mock_btn = gr.Button("Change Mock Object", size="sm")
                mock_status = gr.Textbox(
                    value=f"Current: {state._mock_object_type}",
                    show_label=False,
                    interactive=False,
                    scale=2
                )
        
        # Right: Generalization graph
        with gr.Column(scale=1):
            gr.Markdown("### Generalization Score")
            gen_plot = gr.Plot(
                label="",
                show_label=False,
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        # Label input (simplified)
        with gr.Column(scale=2):
            gr.Markdown("### Label Current Frame")
            with gr.Row():
                label_input = gr.Textbox(
                    placeholder="Type label (e.g., 'tree') and press Enter...",
                    show_label=False,
                    scale=3,
                )
                submit_btn = gr.Button("Add Label", variant="primary", scale=1)
        
        # Stats panel
        with gr.Column(scale=1):
            gr.Markdown("### Stats")
            status_box = gr.Textbox(
                interactive=False,
                lines=10,
                show_label=False,
            )
    
    gr.Markdown("---")
    
    # Control buttons
    with gr.Row():
        start_btn = gr.Button("START", variant="primary", size="lg", scale=1)
        stop_btn = gr.Button("STOP", variant="stop", size="lg", scale=1, interactive=False)
        export_btn = gr.Button("EXPORT CSV", variant="secondary", size="lg", scale=1)
    
    # Timer for auto-refresh (Gradio 6.x)
    timer = gr.Timer(value=0.2, active=True)  # 5 FPS refresh
    
    # Wire up timer for auto-refresh
    timer.tick(
        fn=tick,
        outputs=[video_display, gen_plot, status_box],
    )
    
    # Wire up events
    submit_btn.click(
        fn=submit_label,
        inputs=label_input,
        outputs=label_input,
    )
    
    label_input.submit(
        fn=submit_label,
        inputs=label_input,
        outputs=label_input,
    )
    
    start_btn.click(
        fn=start_system,
        outputs=[start_btn, stop_btn],
    )
    
    stop_btn.click(
        fn=stop_system,
        outputs=[start_btn, stop_btn],
    )
    
    export_btn.click(
        fn=export_data,
    )
    
    mock_btn.click(
        fn=change_mock_object,
        outputs=mock_status,
    )
    
    # Initial load
    demo.load(
        fn=get_display_data,
        outputs=[video_display, gen_plot, status_box],
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  SEYMOUR DASHBOARD")
    print("=" * 50)
    print()
    print("Open http://localhost:7860 in your browser")
    print()
    print("Data saved to:")
    print(f"  - {DATA_DIR}/           (CSVs, embeddings)")
    print(f"  - {IMAGES_DIR}/    (frame samples, 1/sec)")
    print()
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
