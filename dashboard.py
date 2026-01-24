"""
Seymour Dashboard - Gradio-based control interface.

Run with: python dashboard.py
Opens at: http://localhost:7860
"""

import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

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
    def __init__(self):
        self.running = False
        self.frame_count = 0
        self.current_embedding = None
        self.last_message = ""
        self.last_frame_save_time = 0
        self.saved_frame_count = 0
        self.frame_save_interval = 1.0
        self.tracker = GeneralizationTracker(similarity_threshold=0.7)
        self._mock_object_type = "unknown"
        self._mock_objects = ["tree", "rock", "person", "car", "unknown"]
    
    def generate_mock_frame(self) -> np.ndarray:
        w, h = 640, 480
        img = Image.new('RGB', (w, h), color=(30, 30, 40))
        draw = ImageDraw.Draw(img)
        
        for y in range(0, h, 4):
            intensity = int(40 + 20 * np.sin(self.frame_count * 0.03 + y * 0.01))
            draw.line([(0, y), (w, y)], fill=(intensity, intensity, intensity + 10))
        
        cx = int(320 + 150 * np.sin(self.frame_count * 0.02))
        cy = int(240 + 80 * np.cos(self.frame_count * 0.025))
        radius = 60
        
        colors = {"tree": (60, 140, 60), "rock": (120, 110, 100), 
                  "person": (180, 140, 120), "car": (100, 100, 180), "unknown": (150, 150, 150)}
        color = colors.get(self._mock_object_type, (150, 150, 150))
        
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=color, outline=(255, 255, 255), width=2)
        draw.text((cx - 30, cy - 10), self._mock_object_type.upper(), fill=(255, 255, 255))
        
        status_color = (0, 255, 0) if self.running else (255, 100, 100)
        status_text = "LIVE" if self.running else "STOPPED"
        draw.rectangle([w - 100, 10, w - 10, 40], fill=(0, 0, 0))
        draw.text((w - 90, 15), status_text, fill=status_color)
        draw.text((10, 10), f"Frame: {self.frame_count} | Saved: {self.saved_frame_count}", fill=(200, 200, 200))
        
        if self.last_message:
            draw.rectangle([10, h - 35, w - 10, h - 10], fill=(0, 0, 0))
            draw.text((15, h - 30), self.last_message[:80], fill=(200, 200, 100))
        
        return img, np.array(img)
    
    def save_frame(self, img: Image.Image):
        now = time.time()
        if now - self.last_frame_save_time >= self.frame_save_interval:
            self.saved_frame_count += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = IMAGES_DIR / f"frame_{self.saved_frame_count:05d}_{timestamp}.png"
            img.save(filename)
            self.last_frame_save_time = now
    
    def generate_mock_embedding(self) -> np.ndarray:
        np.random.seed(hash(self._mock_object_type) % 2**31)
        base = np.random.randn(512).astype(np.float32)
        np.random.seed(self.frame_count)
        noise = np.random.randn(512).astype(np.float32) * 0.15
        return base + noise
    
    def cycle_mock_object(self):
        idx = self._mock_objects.index(self._mock_object_type)
        self._mock_object_type = self._mock_objects[(idx + 1) % len(self._mock_objects)]
        self.last_message = f"Object changed to: {self._mock_object_type}"


state = SeymourState()


def tick():
    if state.running:
        state.frame_count += 1
        state.current_embedding = state.generate_mock_embedding()
        if state.tracker.clusters:
            result = state.tracker.check_generalization(state.current_embedding, frame_id=state.frame_count)
            if result['best_label']:
                match_str = "YES" if result['generalized'] else "no"
                state.last_message = f"Sees '{result['best_label']}' ({result['best_score']:.0%}) - Match: {match_str}"
    return get_display_data()


def get_display_data():
    pil_img, frame = state.generate_mock_frame()
    if state.running:
        state.save_frame(pil_img)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    scores = state.tracker.get_recent_scores(100)
    if scores:
        x = np.arange(len(scores))
        y = np.array(scores)
        ax.plot(x, y, color='#4CAF50', linewidth=1.5)
        ax.fill_between(x, 0, y, alpha=0.3, color='#4CAF50')
        ax.axhline(y=state.tracker.similarity_threshold, color='#FF5722', linestyle='--', linewidth=1)
        above = y >= state.tracker.similarity_threshold
        if any(above):
            ax.scatter(x[above], y[above], color='#4CAF50', s=15, zorder=5)
        if any(~above):
            ax.scatter(x[~above], y[~above], color='#FF5722', s=15, zorder=5)
    
    ax.set_xlim(0, max(100, len(scores) if scores else 1))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Check #', color='white', fontsize=9)
    ax.set_ylabel('Similarity', color='white', fontsize=9)
    ax.set_title('Generalization Score', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='white')
    plt.tight_layout()
    
    # Stats - balanced format
    labels_str = ", ".join(f"'{k}': {v.count}" for k, v in state.tracker.clusters.items()) if state.tracker.clusters else "None yet"
    status = f"Frames: {state.frame_count}  |  Saved: {state.saved_frame_count}  |  Rate: {state.tracker.get_generalization_rate():.0%}\nLabels: {labels_str}"
    
    plt.close(fig)
    return frame, fig, status


def start_system():
    state.running = True
    state.frame_count = 0
    state.last_frame_save_time = 0
    state.last_message = "System started"
    return gr.update(interactive=False), gr.update(interactive=True)


def stop_system():
    state.running = False
    state.last_message = "System stopped"
    return gr.update(interactive=True), gr.update(interactive=False)


def submit_label(label_text):
    if not label_text.strip():
        state.last_message = "Enter a label"
        return ""
    if state.current_embedding is None:
        state.last_message = "Start system first"
        return ""
    label = label_text.strip().lower()
    state.tracker.add_label(embedding=state.current_embedding, label=label, frame_id=state.frame_count)
    state.last_message = f"Labeled #{state.frame_count} as '{label}'"
    return ""


def export_data():
    if not state.tracker.clusters:
        state.last_message = "No data to export"
        return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    state.tracker.export_csv(str(DATA_DIR / f"seymour_{timestamp}.csv"))
    state.tracker.export_embeddings_npz(str(DATA_DIR / f"seymour_{timestamp}_embeddings.npz"))
    state.last_message = f"Exported to data/seymour_{timestamp}_*"


def change_mock_object():
    state.cycle_mock_object()
    return f"Current: {state._mock_object_type}"


# Build interface
with gr.Blocks(title="Seymour Dashboard") as demo:
    gr.Markdown("# SEYMOUR DASHBOARD")
    gr.Markdown("*Vision-Language Feedback Loop - Generalization Tracking*")
    
    with gr.Row():
        # Left column: video
        with gr.Column(scale=1):
            video_display = gr.Image(show_label=False, height=350)
            with gr.Row():
                mock_btn = gr.Button("Change Object", size="sm")
                mock_status = gr.Textbox(value=f"Current: {state._mock_object_type}", show_label=False, interactive=False, scale=2)
        
        # Right column: graph + stats
        with gr.Column(scale=1):
            gen_plot = gr.Plot(show_label=False)
            status_box = gr.Textbox(interactive=False, lines=2, show_label=False)
    
    with gr.Row():
        label_input = gr.Textbox(placeholder="Type label (e.g., 'tree') and press Enter...", show_label=False, scale=4)
        submit_btn = gr.Button("Add Label", variant="primary", scale=1)
    
    with gr.Row():
        start_btn = gr.Button("START", variant="primary", size="lg", scale=1)
        stop_btn = gr.Button("STOP", variant="stop", size="lg", scale=1, interactive=False)
        export_btn = gr.Button("EXPORT CSV", variant="secondary", size="lg", scale=1)
    
    timer = gr.Timer(value=0.2, active=True)
    timer.tick(fn=tick, outputs=[video_display, gen_plot, status_box])
    
    submit_btn.click(fn=submit_label, inputs=label_input, outputs=label_input)
    label_input.submit(fn=submit_label, inputs=label_input, outputs=label_input)
    start_btn.click(fn=start_system, outputs=[start_btn, stop_btn])
    stop_btn.click(fn=stop_system, outputs=[start_btn, stop_btn])
    export_btn.click(fn=export_data)
    mock_btn.click(fn=change_mock_object, outputs=mock_status)
    demo.load(fn=get_display_data, outputs=[video_display, gen_plot, status_box])


if __name__ == "__main__":
    print("SEYMOUR DASHBOARD - http://localhost:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
