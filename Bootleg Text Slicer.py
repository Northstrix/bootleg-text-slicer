import os
import threading
import time
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import customtkinter as ctk
import tkinter as tk
from faster_whisper import WhisperModel
from tkinter import Canvas, filedialog

# --- CONFIG & COLORS ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
C_BG = "#080808"
C_CARD = "#121212"
C_PURPLE = "#A020F0" # Playhead
C_ACTIVE = "#00F2FF" # Cyan
C_INACTIVE = "#0078D4" # Blue
C_WAVE = "#444444"    # Bright Waveform

INCREMENTS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

class ReviewDashboard(tk.Toplevel):
    def __init__(self, parent, word_list, audio_data, sr):
        super().__init__(parent)
        self.title("REVIEW DASHBOARD - Bootleg Text Slicer")
        self.geometry("1800x950")
        self.configure(bg=C_BG)
        self.attributes("-topmost", True)
        
        self.parent = parent
        self.audio = audio_data
        self.sr = sr
        self.words = word_list 
        self.idx = 0

        self.setup_ui()
        self.update_display()
        
        # KEYBOARD BINDINGS
        self.bind("<Right>", lambda e: self.approve())
        self.bind("<Left>", lambda e: self.prev_word())
        self.bind("<Down>", lambda e: self.play_segment())
        self.bind("<Escape>", lambda e: self.reject())
        
        print(f"[DEBUG] Dashboard: Initialized for {len(self.words)} words.")
        self.after(500, self.play_segment)

    def setup_ui(self):
        # Top Display
        self.lbl_word = tk.Label(self, text="", font=("Segoe UI", 100, "bold"), bg=C_BG, fg=C_ACTIVE)
        self.lbl_word.pack(pady=5)

        # Margin Stats Label
        self.lbl_margins = tk.Label(self, text="", font=("Consolas", 18, "bold"), bg=C_BG, fg=C_PURPLE)
        self.lbl_margins.pack(pady=5)

        self.lbl_stats = tk.Label(self, text="", font=("Consolas", 14), bg=C_BG, fg="#888")
        self.lbl_stats.pack(pady=5)

        ctrl_frame = tk.Frame(self, bg=C_BG)
        ctrl_frame.pack(fill="both", expand=True, padx=20)

        def create_10btn_row(parent, title, key, is_global=False):
            frame = tk.LabelFrame(parent, text=title, bg=C_CARD, fg="#AAA", font=("Arial", 10), padx=5, pady=5)
            frame.pack(fill="x", pady=5)
            
            p_row = tk.Frame(frame, bg=C_CARD)
            p_row.pack(fill="x")
            for val in INCREMENTS:
                tk.Button(p_row, text=f"+{val}", width=8, bg="#1a3a1a", fg="white", relief="flat",
                          command=lambda v=val: self.adj(key, v, is_global)).pack(side="left", padx=2, pady=2)
            
            m_row = tk.Frame(frame, bg=C_CARD)
            m_row.pack(fill="x")
            for val in INCREMENTS:
                tk.Button(m_row, text=f"-{val}", width=8, bg="#3a1a1a", fg="white", relief="flat",
                          command=lambda v=val: self.adj(key, -v, is_global)).pack(side="left", padx=2, pady=2)

        create_10btn_row(ctrl_frame, "INDIVIDUAL START MARGIN (S:Prev/W:Next)", "m_start", False)
        create_10btn_row(ctrl_frame, "INDIVIDUAL END MARGIN", "m_end", False)
        
        tk.Label(ctrl_frame, text="GLOBAL OVERRIDE (ALL WORDS)", bg=C_BG, fg=C_PURPLE, font=("Arial", 10, "bold")).pack(pady=5)
        create_10btn_row(ctrl_frame, "GLOBAL START", "m_start", True)
        create_10btn_row(ctrl_frame, "GLOBAL END", "m_end", True)

        footer = tk.Frame(self, bg=C_BG)
        footer.pack(side="bottom", pady=20)
        tk.Button(footer, text="[LEFT] PREV", width=15, bg="#333", fg="white", command=self.prev_word).pack(side="left", padx=10)
        tk.Button(footer, text="[DOWN] PLAY", width=15, bg=C_INACTIVE, fg="white", command=self.play_segment).pack(side="left", padx=10)
        tk.Button(footer, text="[RIGHT] APPROVE", width=25, bg="#28a745", fg="white", command=self.approve).pack(side="left", padx=10)
        tk.Button(footer, text="[ESC] SKIP", width=15, bg="#dc3545", fg="white", command=self.reject).pack(side="left", padx=10)

    def adj(self, key, val, is_global):
        if is_global:
            for w in self.words: w[key] = round(w.get(key, 0.0) + val, 4)
        else:
            self.words[self.idx][key] = round(self.words[self.idx].get(key, 0.0) + val, 4)
        self.update_display()
        self.play_segment()
        self.parent.update_canvas()

    def update_display(self):
        w = self.words[self.idx]
        self.lbl_word.config(text=f"'{w['word']}'")
        s, e = self.parent.get_effective_times(w)
        self.lbl_margins.config(text=f"LOCAL OFFSETS -> Start: {w['m_start']:+.3f}s | End: {w['m_end']:+.3f}s")
        self.lbl_stats.config(text=f"WORD {self.idx+1}/{len(self.words)} | FINAL DURATION: {e-s:.3f}s")

    def play_segment(self):
        w = self.words[self.idx]
        s, e = self.parent.get_effective_times(w)
        sd.stop()
        sd.play(self.audio[int(max(0, s*self.sr)):int(min(len(self.audio), e*self.sr))], self.sr)

    def approve(self):
        w = self.words[self.idx]
        if not os.path.exists("ApprovedWords"): os.makedirs("ApprovedWords")
        s, e = self.parent.get_effective_times(w)
        chunk = self.audio[int(s*self.sr):int(e*self.sr)]
        clean = "".join(x for x in w['word'] if x.isalnum())
        path = f"ApprovedWords/{clean}_{int(time.time())}.wav"
        sf.write(path, chunk, self.sr)
        print(f"[EXPORT] Saved {path} | Duration: {e-s:.3f}s")
        self.next_word()

    def reject(self): self.next_word()
    def next_word(self):
        self.idx += 1
        if self.idx >= len(self.words): self.destroy()
        else: self.update_display(); self.play_segment()
    def prev_word(self):
        self.idx = max(0, self.idx - 1)
        self.update_display(); self.play_segment()

class BootlegTextSlicer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Bootleg Text Slicer")
        self.geometry("2480x932")
        self.configure(fg_color=C_BG)

        self.audio_data = None
        self.sr = 16000
        self.words = []
        self.current_time = 0.0
        self.view_offset = 0.0
        self.view_duration = 6.0
        self.is_playing = False
        
        self.dragging = None
        self.drag_target_idx = -1

        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        self.header = ctk.CTkFrame(self, height=60, fg_color=C_CARD, corner_radius=0)
        self.header.pack(fill="x")
        self.status = ctk.CTkLabel(self.header, text="LOAD AUDIO FILE", font=("Segoe UI", 16, "bold"), text_color=C_INACTIVE)
        self.status.pack(side="left", padx=30)

        self.canvas = Canvas(self, bg=C_BG, highlightthickness=0, borderwidth=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.ctrl = ctk.CTkFrame(self, height=250, fg_color=C_CARD)
        self.ctrl.pack(fill="x", side="bottom", padx=10, pady=10)
        
        f_left = ctk.CTkFrame(self.ctrl, fg_color="transparent")
        f_left.pack(side="left", padx=20, fill="y", pady=10)
        ctk.CTkButton(f_left, text="LOAD FILE", command=self.load_audio).pack(pady=5)
        ctk.CTkButton(f_left, text="DASHBOARD", command=self.open_dashboard, fg_color="#6200EE").pack(pady=5)

        f_mid = ctk.CTkFrame(self.ctrl, fg_color="#111", border_width=1, border_color="#333")
        f_mid.pack(side="left", padx=20, expand=True, fill="both", pady=10)
        ctk.CTkLabel(f_mid, text="GLOBAL MARGINS (ALL WORDS)", font=("Segoe UI", 12, "bold"), text_color=C_ACTIVE).pack(pady=2)

        def create_global_row(parent, label, key):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(pady=2, fill="x")
            ctk.CTkLabel(row, text=label, width=100).pack(side="left", padx=5)
            for v in INCREMENTS:
                ctk.CTkButton(row, text=f"+{v}", width=60, command=lambda k=key, val=v: self.adj_global(k, val)).pack(side="left", padx=1)
            for v in INCREMENTS:
                ctk.CTkButton(row, text=f"-{v}", width=60, fg_color="#422", command=lambda k=key, val=-v: self.adj_global(k, val)).pack(side="left", padx=1)

        create_global_row(f_mid, "START", "m_start")
        create_global_row(f_mid, "END", "m_end")

        f_right = ctk.CTkFrame(self.ctrl, fg_color="transparent")
        f_right.pack(side="right", padx=20, fill="y", pady=10)
        self.btn_play = ctk.CTkButton(f_right, text="PLAY TRACK", command=self.toggle_play)
        self.btn_play.pack(pady=5)
        ctk.CTkButton(f_right, text="PLAY WORD", fg_color="#28a745", command=self.play_word_static).pack(pady=5)

        self.canvas.bind("<MouseWheel>", self.handle_scroll)
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, 'dragging', None))

    def get_effective_times(self, w):
        return w['start'] + w['m_start'], w['end'] + w['m_end']

    def adj_global(self, key, val):
        for w in self.words: w[key] = round(w.get(key, 0.0) + val, 4)
        self.update_canvas()

    def load_audio(self):
        path = filedialog.askopenfilename()
        if path:
            self.status.configure(text="PROCESSING...")
            threading.Thread(target=self.transcribe, args=(path,), daemon=True).start()

    def transcribe(self, path):
        try:
            start_proc = time.time()
            y, sr = librosa.load(path, sr=self.sr)
            self.audio_data = y
            duration = len(y) / sr
            
            model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(path, word_timestamps=True)
            self.words = [{'word': w.word.strip(), 'start': w.start, 'end': w.end, 'm_start': 0.0, 'm_end': 0.0} 
                          for s in segments for w in s.words]
            
            end_proc = time.time()
            elapsed = end_proc - start_proc
            
            print("\n" + "="*50)
            print("BOOTLEG TEXT SLICER - ANALYSIS COMPLETE")
            print(f"Track: {os.path.basename(path)}")
            print(f"Track Duration: {duration:.2f} seconds")
            print(f"Words Detected: {len(self.words)}")
            print(f"Processing Time: {elapsed:.2f} seconds")
            print(f"Efficiency: {duration/elapsed:.2f}x Realtime")
            print("="*50 + "\n")
            
            self.status.configure(text=f"ACTIVE: {len(self.words)} WORDS LOADED")
        except Exception as e: print(f"[ERROR] {e}")

    def toggle_play(self):
        if self.is_playing:
            sd.stop(); self.is_playing = False
            self.btn_play.configure(text="PLAY TRACK")
        else:
            if self.audio_data is not None:
                self.is_playing = True
                self.start_time_real = time.time()
                self.start_timestamp = self.current_time
                sd.play(self.audio_data[int(self.current_time*self.sr):], self.sr)
                self.btn_play.configure(text="STOP TRACK")

    def play_word_static(self):
        for w in self.words:
            s, e = self.get_effective_times(w)
            if s <= self.current_time <= e:
                sd.stop(); sd.play(self.audio_data[int(s*self.sr):int(e*self.sr)], self.sr)
                return

    def open_dashboard(self):
        if self.words: ReviewDashboard(self, self.words, self.audio_data, self.sr)

    def handle_scroll(self, event):
        self.view_offset = max(0, self.view_offset - (event.delta / 120) * 0.5)
        self.update_canvas()

    def on_press(self, e):
        pxs = self.canvas.winfo_width() / self.view_duration
        t = self.view_offset + (e.x / pxs)
        
        for i, w in enumerate(self.words):
            if abs(t - w['start']) < 0.08: self.dragging = "s"; self.drag_target_idx = i; return
            if abs(t - w['end']) < 0.08: self.dragging = "e"; self.drag_target_idx = i; return

        self.current_time = t
        if self.is_playing:
            self.toggle_play(); self.toggle_play() 
        self.update_canvas()

    def on_drag(self, e):
        pxs = self.canvas.winfo_width() / self.view_duration
        t = max(0, self.view_offset + (e.x / pxs))
        if self.dragging == "s": self.words[self.drag_target_idx]['start'] = t
        elif self.dragging == "e": self.words[self.drag_target_idx]['end'] = t
        else: self.current_time = t
        self.update_canvas()

    def update_loop(self):
        if self.is_playing:
            self.current_time = self.start_timestamp + (time.time() - self.start_time_real)
            if self.current_time > self.view_offset + self.view_duration:
                self.view_offset = self.current_time - (self.view_duration * 0.1)
        self.update_canvas()
        self.after(30, self.update_loop)

    def update_canvas(self):
        self.canvas.delete("all")
        win_w, win_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if win_w < 10: return
        pxs, midy = win_w / self.view_duration, win_h / 2

        # Brighter Waveform using Peak Sampling
        if self.audio_data is not None:
            vi = int(self.view_offset * self.sr)
            spp = int((self.view_duration * self.sr) / win_w)
            for x in range(0, win_w, 4):
                idx = vi + int((x / win_w) * (self.view_duration * self.sr))
                if 0 <= idx < len(self.audio_data):
                    chunk = self.audio_data[idx : idx + spp]
                    if chunk.size > 0:
                        amp = np.max(np.abs(chunk)) * (win_h * 0.45) * 1.6
                        self.canvas.create_line(x, midy-amp, x, midy+amp, fill=C_WAVE, width=2)

        # Word Rendering
        for word in self.words:
            s_eff, e_eff = self.get_effective_times(word)
            xs_sol, xe_sol = (word['start']-self.view_offset)*pxs, (word['end']-self.view_offset)*pxs
            xs_dash, xe_dash = (s_eff-self.view_offset)*pxs, (e_eff-self.view_offset)*pxs
            
            is_active = s_eff <= self.current_time <= e_eff
            color = C_ACTIVE if is_active else C_INACTIVE

            if xe_sol > 0 and xs_sol < win_w:
                # Solid lines
                self.canvas.create_line(xs_sol, 65, xs_sol, win_h-65, fill=color, width=2)
                self.canvas.create_line(xe_sol, 65, xe_sol, win_h-65, fill=color, width=2)
                
                # Dashed lines + Labels
                if word['m_start'] != 0:
                    self.canvas.create_line(xs_dash, 45, xs_dash, win_h-45, fill=color, dash=(5,5), width=2)
                    self.canvas.create_text(xs_dash, 35, text=f"{word['m_start']:+.3f}", fill=color, font=("Consolas", 9))
                if word['m_end'] != 0:
                    self.canvas.create_line(xe_dash, 45, xe_dash, win_h-45, fill=color, dash=(5,5), width=2)
                    self.canvas.create_text(xe_dash, 35, text=f"{word['m_end']:+.3f}", fill=color, font=("Consolas", 9))

                self.canvas.create_text(xs_sol+(xe_sol-xs_sol)/2, win_h*0.2, text=word['word'], fill="white", font=("Segoe UI", 12, "bold"))

        # Playhead
        phx = (self.current_time - self.view_offset) * pxs
        if 0 <= phx <= win_w:
            self.canvas.create_line(phx, 0, phx, win_h, fill=C_PURPLE, width=3)

if __name__ == "__main__":
    app = BootlegTextSlicer()
    app.mainloop()