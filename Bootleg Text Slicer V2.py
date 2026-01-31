import os
import io
import time
import threading

import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from pydub import AudioSegment
import pygame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import librosa
import soundfile as sf
from faster_whisper import WhisperModel

# --- CONFIGURABLE VISUALS (blue waveform scheme) ---
WAVEFORM_STROKE = 1.2
TIMELINE_AUDIO = "#1177ED"   # waveform (blue)
TIMELINE_CLICK = "#ED1177"   # pink/red for playhead
BG_BLACK = "#000000"

# Main Colors
ACCENT_BLUE = "#1177ED"      # blue
ACCENT_PURPLE = "#8711ED"    # purple
COLOR_SUCCESS = "#11ED19"    # green (start line / slider)
COLOR_ERROR = "#ED1911"
COLOR_MARGINAL = "#11E5ED"
TEXT_WHITE = "#FFFFFF"

# 20% Darker Hover Colors
HOVER_BLUE = "#0E5FBD"
HOVER_PURPLE = "#6C0EBD"
HOVER_SUCCESS = "#0EBD14"
HOVER_ERROR = "#BD140E"
HOVER_NEUTRAL = "#1B1B1B"

# Margin button colors (only used on margin adjust buttons)
MARGIN_POS_FG = "#182A41"
MARGIN_POS_HOVER = "#0E71BD"
MARGIN_NEG_FG = "#44150A"
MARGIN_NEG_HOVER = "#BD140E"

CUT_TEMPLATE_FILE = "cutTemplate.txt"

# Margin increments: -0.10 ... -0.01 +0.01 ... +0.10
NEG_INCREMENTS = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
POS_INCREMENTS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

pygame.mixer.init()
ctk.set_appearance_mode("dark")


class BootlegTextSlicer(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Bootleg Text Slicer")
        self.geometry("2376x980")
        self.configure(fg_color=BG_BLACK)

        # Audio (pydub) + path
        self.audio_seg: AudioSegment | None = None
        self.audio_path = None
        self.global_avg_dbfs = -20.0

        # Numpy audio for Whisper
        self.audio_np = None
        self.sr = 16000

        # Words
        # Each word dict:
        # {word, start_ms, end_ms, m_start, m_end, export_path, g_start, g_end}
        self.words = []
        self.current_index = 0

        # Timeline / zoom / playhead
        self.zoom_factor = 1.0
        self.view_start = 0.0
        self.playhead_pos = 0.0
        self.dragging_playhead = False
        self.is_playing = False
        self._playhead_updater_id = None

        # Selected range for transcription (ms)
        self.sel_start_ms = 0.0
        self.sel_end_ms = 0.0

        # Transcription background
        self._transcribe_thread = None

        self._play_lock = threading.Lock()

        # Delayed playback handle (for margin adjustments)
        self._margin_play_after_id = None

        self._init_cut_template()
        self.setup_ui()
        self.setup_bindings()

    # -------------------- CUT TEMPLATE INIT --------------------
    def _init_cut_template(self):
        header = "# word,start_ms,end_ms,global_m_start,global_m_end,local_m_start,local_m_end,export_path\n"

        if not os.path.exists(CUT_TEMPLATE_FILE):
            with open(CUT_TEMPLATE_FILE, "w", encoding="utf-8") as f:
                f.write(header)
            return

        has_content = False
        try:
            with open(CUT_TEMPLATE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        has_content = True
                        break
        except Exception:
            has_content = False

        if has_content:
            ans = messagebox.askyesno(
                "cutTemplate.txt",
                "cutTemplate.txt already contains entries.\n\n"
                "Do you want to clean it (start fresh)?"
            )
            if ans:
                with open(CUT_TEMPLATE_FILE, "w", encoding="utf-8") as f:
                    f.write(header)
        else:
            with open(CUT_TEMPLATE_FILE, "w", encoding="utf-8") as f:
                f.write(header)

    # -------------------- UI SETUP --------------------
    def setup_ui(self):
        # Header / status (filename + word info only after words exist)
        self.status_label = ctk.CTkLabel(
            self,
            text="No audio loaded",
            font=("Segoe UI", 28, "bold"),
            text_color=ACCENT_BLUE
        )
        self.status_label.pack(pady=(10, 5))

        # Transcription status label
        self.transcribe_label = ctk.CTkLabel(
            self,
            text="Idle",
            font=("Segoe UI", 14),
            text_color=TEXT_WHITE
        )
        self.transcribe_label.pack(pady=(0, 5))

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=10)

        # LEFT side: waveform + sliders + buttons
        left = ctk.CTkFrame(main, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Waveform
        self.canvas_frame = ctk.CTkFrame(
            left,
            fg_color="#050505",
            border_width=1,
            border_color="#222"
        )
        self.canvas_frame.pack(fill="both", expand=True, padx=5, pady=(5, 10))

        self.fig, self.ax = plt.subplots(figsize=(9, 3))
        self.fig.patch.set_facecolor(BG_BLACK)
        self.ax.set_facecolor(BG_BLACK)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.25)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # Sliders for transcription range
        trim_frame = ctk.CTkFrame(left, fg_color="transparent")
        trim_frame.pack(fill="x", padx=5, pady=(0, 10))

        # Start slider: green theme
        self.start_slider = ctk.CTkSlider(
            trim_frame,
            from_=0, to=100,
            button_color=COLOR_SUCCESS,
            button_hover_color=HOVER_SUCCESS,
            progress_color=COLOR_SUCCESS,
            command=self.on_slider_change
        )
        self.start_slider.pack(fill="x", pady=2)

        # End slider: purple
        self.end_slider = ctk.CTkSlider(
            trim_frame,
            from_=0, to=100,
            button_color=ACCENT_PURPLE,
            button_hover_color=HOVER_PURPLE,
            progress_color=ACCENT_PURPLE,
            command=self.on_slider_change
        )
        self.end_slider.pack(fill="x", pady=2)

        # Slider values label
        self.lbl_range = ctk.CTkLabel(
            left,
            text="Start: 0.000s   End: 0.000s   (0.000s)",
            font=("Consolas", 14),
            text_color=TEXT_WHITE
        )
        self.lbl_range.pack(pady=(0, 10))

        # Buttons row (same height, one row)
        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.pack(fill="x", padx=5, pady=(0, 10))

        BTN_H = 44

        self.btn_open = ctk.CTkButton(
            btn_row,
            text="Open Audio",
            fg_color=ACCENT_BLUE,
            hover_color=HOVER_BLUE,
            width=150,
            height=BTN_H,
            command=self.open_file
        )
        self.btn_open.pack(side="left", padx=4, pady=4)

        self.btn_load_template = ctk.CTkButton(
            btn_row,
            text="Load cutTemplate + Audio",
            fg_color="#222",
            hover_color=HOVER_NEUTRAL,
            width=210,
            height=BTN_H,
            command=self.load_from_template
        )
        self.btn_load_template.pack(side="left", padx=4, pady=4)

        self.btn_play = ctk.CTkButton(
            btn_row,
            text="Play from Playhead",
            fg_color=ACCENT_PURPLE,
            hover_color=HOVER_PURPLE,
            width=180,
            height=BTN_H,
            command=self.play_from_playhead
        )
        self.btn_play.pack(side="left", padx=4, pady=4)

        self.btn_play_sel = ctk.CTkButton(
            btn_row,
            text="Play Selected Range",
            fg_color=ACCENT_BLUE,
            hover_color=HOVER_BLUE,
            width=180,
            height=BTN_H,
            command=self.play_selection
        )
        self.btn_play_sel.pack(side="left", padx=4, pady=4)

        self.btn_stop = ctk.CTkButton(
            btn_row,
            text="Stop",
            fg_color=COLOR_ERROR,
            hover_color=HOVER_ERROR,
            width=110,
            height=BTN_H,
            command=self.stop_playback
        )
        self.btn_stop.pack(side="left", padx=4, pady=4)

        self.btn_transcribe = ctk.CTkButton(
            btn_row,
            text="TRANSCRIBE SELECTED RANGE",
            fg_color=COLOR_SUCCESS,
            hover_color=HOVER_SUCCESS,
            text_color="black",
            width=260,
            height=BTN_H,
            command=self.transcribe_selected_async
        )
        self.btn_transcribe.pack(side="left", padx=4, pady=4)

        # RIGHT side: words panel
        right = ctk.CTkFrame(main, fg_color="#0a0a0a", border_width=1, border_color="#1a1a1a")
        right.pack(side="left", fill="y", padx=(0, 0), pady=5)

        self.lbl_word = ctk.CTkLabel(
            right,
            text="No word",
            font=("Segoe UI", 32, "bold"),
            text_color=ACCENT_BLUE
        )
        self.lbl_word.pack(pady=(10, 0))

        # Current word index label: "Word 10 / 345"
        self.lbl_word_index = ctk.CTkLabel(
            right,
            text="",
            font=("Consolas", 14),
            text_color="#CCCCCC"
        )
        self.lbl_word_index.pack(pady=(0, 4))

        self.lbl_margins = ctk.CTkLabel(
            right,
            text="",
            font=("Consolas", 14, "bold"),
            text_color=ACCENT_PURPLE
        )
        self.lbl_margins.pack(pady=2)

        self.lbl_stats = ctk.CTkLabel(
            right,
            text="",
            font=("Consolas", 12),
            text_color="#AAAAAA"
        )
        self.lbl_stats.pack(pady=2)

        # Per-word waveform
        word_wave_frame = ctk.CTkFrame(
            right,
            fg_color="#050505",
            border_width=1,
            border_color="#222"
        )
        word_wave_frame.pack(fill="both", expand=False, padx=10, pady=10)

        self.fig_word, self.ax_word = plt.subplots(figsize=(4, 2))
        self.fig_word.patch.set_facecolor(BG_BLACK)
        self.ax_word.set_facecolor(BG_BLACK)
        self.fig_word.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.2)
        self.canvas_word = FigureCanvasTkAgg(self.fig_word, master=word_wave_frame)
        self.canvas_word.get_tk_widget().pack(fill="both", expand=True)

        # Export name
        rename_frame = ctk.CTkFrame(right, fg_color="transparent")
        rename_frame.pack(fill="x", padx=10, pady=(0, 5))
        ctk.CTkLabel(
            rename_frame,
            text="Export Name:",
            font=("Arial", 14, "bold"),
            text_color=TEXT_WHITE,
            width=100
        ).pack(side="left", padx=5)
        self.entry_export_name = ctk.CTkEntry(rename_frame, width=180, height=32, font=("Arial", 14))
        self.entry_export_name.pack(side="left", padx=5)

        # Global margins
        global_box = ctk.CTkFrame(right, fg_color="transparent")
        global_box.pack(fill="x", padx=10, pady=(5, 5))

        ctk.CTkLabel(
            global_box,
            text="Global Margins (all words)",
            font=("Arial", 12, "bold"),
            text_color=TEXT_WHITE
        ).pack(pady=2)

        self.lbl_global_offsets = ctk.CTkLabel(
            global_box,
            text="Global: start +0.000s, end +0.000s",
            font=("Consolas", 11),
            text_color=COLOR_MARGINAL
        )
        self.lbl_global_offsets.pack(pady=(0, 4))

        def create_global_row(parent, label, key):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(pady=2, fill="x")
            ctk.CTkLabel(row, text=label, width=80).pack(side="left", padx=5)

            # Negative side: -0.10 ... -0.01
            for v in NEG_INCREMENTS:
                ctk.CTkButton(
                    row,
                    text=f"-{v:.2f}",
                    width=48,
                    height=26,
                    fg_color=MARGIN_NEG_FG,
                    hover_color=MARGIN_NEG_HOVER,
                    command=lambda k=key, val=-v: self.adj_global(k, val)
                ).pack(side="left", padx=1)

            # Positive side: +0.01 ... +0.10
            for v in POS_INCREMENTS:
                ctk.CTkButton(
                    row,
                    text=f"+{v:.2f}",
                    width=48,
                    height=26,
                    fg_color=MARGIN_POS_FG,
                    hover_color=MARGIN_POS_HOVER,
                    command=lambda k=key, val=v: self.adj_global(k, val)
                ).pack(side="left", padx=1)

        create_global_row(global_box, "Start", "m_start")
        create_global_row(global_box, "End", "m_end")

        # Individual margins
        indiv_box = ctk.CTkFrame(right, fg_color="transparent")
        indiv_box.pack(fill="x", padx=10, pady=(5, 5))
        ctk.CTkLabel(
            indiv_box,
            text="Current Word Margins",
            font=("Arial", 12, "bold"),
            text_color=TEXT_WHITE
        ).pack(pady=2)

        self.lbl_indiv_offsets = ctk.CTkLabel(
            indiv_box,
            text="Local: start +0.000s, end +0.000s",
            font=("Consolas", 11),
            text_color=COLOR_MARGINAL
        )
        self.lbl_indiv_offsets.pack(pady=(0, 4))

        def create_indiv_row(parent, label, key):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(pady=2, fill="x")
            ctk.CTkLabel(row, text=label, width=80).pack(side="left", padx=5)

            # Negative side: -0.10 ... -0.01
            for v in NEG_INCREMENTS:
                ctk.CTkButton(
                    row,
                    text=f"-{v:.2f}",
                    width=48,
                    height=26,
                    fg_color=MARGIN_NEG_FG,
                    hover_color=MARGIN_NEG_HOVER,
                    command=lambda k=key, val=-v: self.adj_individual(k, val)
                ).pack(side="left", padx=1)

            # Positive side: +0.01 ... +0.10
            for v in POS_INCREMENTS:
                ctk.CTkButton(
                    row,
                    text=f"+{v:.2f}",
                    width=48,
                    height=26,
                    fg_color=MARGIN_POS_FG,
                    hover_color=MARGIN_POS_HOVER,
                    command=lambda k=key, val=v: self.adj_individual(k, val)
                ).pack(side="left", padx=1)

        create_indiv_row(indiv_box, "Start", "m_start")
        create_indiv_row(indiv_box, "End", "m_end")

        # Navigation / actions
        nav_box = ctk.CTkFrame(right, fg_color="transparent")
        nav_box.pack(fill="x", padx=10, pady=(10, 10))

        self.btn_prev_word = ctk.CTkButton(
            nav_box,
            text="Prev [←]",
            fg_color="#222",
            hover_color=HOVER_NEUTRAL,
            width=80,
            command=self.prev_word
        )
        self.btn_prev_word.pack(side="left", padx=2, pady=4)

        self.btn_play_word = ctk.CTkButton(
            nav_box,
            text="Play [↓]",
            fg_color=ACCENT_BLUE,
            hover_color=HOVER_BLUE,
            width=80,
            command=self.play_current_word
        )
        self.btn_play_word.pack(side="left", padx=2, pady=4)

        self.btn_approve = ctk.CTkButton(
            nav_box,
            text="Approve [→]",
            fg_color=COLOR_SUCCESS,
            hover_color=HOVER_SUCCESS,
            text_color="black",
            width=120,
            command=self.approve_current_word
        )
        self.btn_approve.pack(side="left", padx=2, pady=4)

        self.btn_skip = ctk.CTkButton(
            nav_box,
            text="Skip [ESC]",
            fg_color=COLOR_ERROR,
            hover_color=HOVER_ERROR,
            width=80,
            command=self.skip_current_word
        )
        self.btn_skip.pack(side="left", padx=2, pady=4)

    def setup_bindings(self):
        self.canvas_widget.bind("<Button-1>", self.on_canvas_click)
        self.canvas_widget.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas_widget.bind("<ButtonRelease-1>", self.on_canvas_release)

        self.canvas_widget.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas_widget.bind("<Button-4>", self.on_mouse_wheel_mac_up)
        self.canvas_widget.bind("<Button-5>", self.on_mouse_wheel_mac_down)

        # Word shortcuts only
        self.bind("<Right>", lambda e: self.approve_current_word())
        self.bind("<Left>", lambda e: self.prev_word())
        self.bind("<Down>", lambda e: self.play_current_word())
        self.bind("<Escape>", lambda e: self.skip_current_word())

    # -------------------- Helpers --------------------
    def audio_length_ms(self):
        return len(self.audio_seg) if self.audio_seg else 0

    # -------------------- File I/O --------------------
    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.aac *.m4a"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return
        try:
            self.audio_seg = AudioSegment.from_file(file_path)
            self.audio_path = file_path
            self.global_avg_dbfs = (
                self.audio_seg.dBFS if self.audio_seg.dBFS != float("-inf") else -20.0
            )

            total_ms = self.audio_length_ms()
            self.start_slider.configure(from_=0, to=total_ms)
            self.end_slider.configure(from_=0, to=total_ms)
            self.start_slider.set(0)
            self.end_slider.set(total_ms)
            self.sel_start_ms = 0.0
            self.sel_end_ms = total_ms

            self.zoom_factor = 1.0
            self.view_start = 0.0
            self.playhead_pos = 0.0

            self._update_status_filename(extra="")
            y, _ = librosa.load(file_path, sr=self.sr)
            self.audio_np = y

            self.update_range_label()
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file:\n{e}")

    def load_from_template(self):
        if not os.path.exists(CUT_TEMPLATE_FILE):
            messagebox.showerror("Missing", f"{CUT_TEMPLATE_FILE} not found.")
            return

        entries = []
        with open(CUT_TEMPLATE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",", 7)
                # word,start_ms,end_ms,global_m_start,global_m_end,local_m_start,local_m_end,export_path
                if len(parts) < 7:
                    # legacy format (word,start_ms,end_ms,export_path)
                    if len(parts) == 4:
                        name, s_ms, e_ms, pth = parts
                        try:
                            s_ms = float(s_ms)
                            e_ms = float(e_ms)
                        except:
                            continue
                        entries.append((name, s_ms, e_ms, 0.0, 0.0, 0.0, 0.0, pth))
                    continue
                name = parts[0]
                try:
                    s_ms = float(parts[1])
                    e_ms = float(parts[2])
                except:
                    continue
                try:
                    g_start = float(parts[3])
                    g_end = float(parts[4])
                    l_start = float(parts[5])
                    l_end = float(parts[6])
                except:
                    g_start = 0.0
                    g_end = 0.0
                    l_start = 0.0
                    l_end = 0.0
                pth = parts[7] if len(parts) >= 8 else ""
                entries.append((name, s_ms, e_ms, g_start, g_end, l_start, l_end, pth))

        if not entries:
            messagebox.showinfo("Empty", "cutTemplate.txt has no entries.")
            return

        audio_path = filedialog.askopenfilename(
            title="Select original audio file for this cutTemplate.txt",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac"), ("All files", "*.*")]
        )
        if not audio_path:
            return

        try:
            self.audio_seg = AudioSegment.from_file(audio_path)
            self.audio_path = audio_path
            self.global_avg_dbfs = (
                self.audio_seg.dBFS if self.audio_seg.dBFS != float("-inf") else -20.0
            )

            total_ms = self.audio_length_ms()
            self.start_slider.configure(from_=0, to=total_ms)
            self.end_slider.configure(from_=0, to=total_ms)
            self.start_slider.set(0)
            self.end_slider.set(total_ms)
            self.sel_start_ms = 0.0
            self.sel_end_ms = total_ms

            self.zoom_factor = 1.0
            self.view_start = 0.0
            self.playhead_pos = 0.0

            y, _ = librosa.load(audio_path, sr=self.sr)
            self.audio_np = y

            self.words = []
            for (name, s_ms, e_ms, g_start, g_end, l_start, l_end, pth) in entries:
                self.words.append(
                    {
                        "word": name,
                        "start_ms": s_ms,
                        "end_ms": e_ms,
                        "m_start": l_start,
                        "m_end": l_end,
                        "g_start": g_start,
                        "g_end": g_end,
                        "export_path": pth,
                    }
                )

            self.current_index = 0
            self._update_status_filename(extra=f"({len(self.words)} words)")
            self.update_range_label()
            self.update_word_display()
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load from template:\n{e}")

    def _update_status_filename(self, extra=""):
        if self.audio_path:
            base = os.path.basename(self.audio_path)
            suffix = f" {extra}" if extra else ""
            self.status_label.configure(text=f"{base}{suffix}", text_color=ACCENT_BLUE)
        else:
            self.status_label.configure(text="No audio loaded", text_color=ACCENT_BLUE)

    # -------------------- Zoom / scroll / playhead --------------------
    def on_mouse_wheel(self, event):
        if not self.audio_seg:
            return
        if (event.state & 0x4) != 0:  # Ctrl = zoom
            direction = 1 if event.delta > 0 else -1
            self.adjust_zoom(direction, event.x)
        else:
            delta = -event.delta / 120.0
            self.adjust_scroll(delta * 0.05)
        self.update_plot()

    def on_mouse_wheel_mac_up(self, event):
        if not self.audio_seg:
            return
        self.adjust_zoom(1, event.x)
        self.update_plot()

    def on_mouse_wheel_mac_down(self, event):
        if not self.audio_seg:
            return
        self.adjust_zoom(-1, event.x)
        self.update_plot()

    def adjust_zoom(self, direction, mouse_x):
        old_zoom = self.zoom_factor
        self.zoom_factor *= (1.0 + 0.2 * direction)
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))

        if not self.audio_seg:
            return

        width = self.canvas_widget.winfo_width() or 1
        mouse_rel = mouse_x / width

        old_view_width = 1.0 / old_zoom
        old_pos_global = self.view_start + mouse_rel * old_view_width

        new_view_width = 1.0 / self.zoom_factor
        self.view_start = old_pos_global - mouse_rel * new_view_width
        self.view_start = max(0.0, min(1.0 - new_view_width, self.view_start))

    def adjust_scroll(self, amount):
        if not self.audio_seg:
            return
        view_width = 1.0 / self.zoom_factor
        self.view_start += amount * view_width
        self.view_start = max(0.0, min(1.0 - view_width, self.view_start))

    def _event_x_to_playhead_norm(self, event_x):
        if not self.audio_seg:
            return 0.0
        inv = self.ax.transAxes.inverted()
        x_axes, _ = inv.transform((event_x, 0))
        x_axes = max(0.0, min(1.0, x_axes))
        view_width = 1.0 / self.zoom_factor
        return self.view_start + x_axes * view_width

    def on_canvas_click(self, event):
        if not self.audio_seg:
            return
        if self.is_playing:
            return
        self.playhead_pos = self._event_x_to_playhead_norm(event.x)
        self.playhead_pos = max(0.0, min(1.0, self.playhead_pos))
        self.dragging_playhead = True
        self.update_plot()

    def on_canvas_drag(self, event):
        if not self.audio_seg or not self.dragging_playhead:
            return
        if self.is_playing:
            return
        self.playhead_pos = self._event_x_to_playhead_norm(event.x)
        self.playhead_pos = max(0.0, min(1.0, self.playhead_pos))
        self.update_plot()

    def on_canvas_release(self, _event):
        self.dragging_playhead = False

    # -------------------- Plot --------------------
    def update_plot(self):
        if not self.audio_seg:
            self.ax.clear()
            self.ax.axis("off")
            self.canvas.draw()
            return

        self.ax.clear()

        samples = np.array(self.audio_seg.get_array_of_samples(), dtype=np.float32)
        total_len = len(samples)
        if total_len == 0:
            self.canvas.draw()
            return

        view_width = 1.0 / self.zoom_factor
        start_norm = self.view_start
        end_norm = self.view_start + view_width
        start_norm = max(0.0, start_norm)
        end_norm = min(1.0, end_norm)

        start_idx = int(start_norm * total_len)
        end_idx = int(end_norm * total_len)
        end_idx = max(start_idx + 1, end_idx)

        view_samples = samples[start_idx:end_idx]
        step = max(1, len(view_samples) // 6000)
        view_samples_ds = view_samples[::step]

        x_data = np.arange(len(view_samples_ds))
        self.ax.plot(
            x_data,
            view_samples_ds,
            color=TIMELINE_AUDIO,
            linewidth=WAVEFORM_STROKE,
            alpha=0.9
        )

        total_ms = self.audio_length_ms()

        def ms_to_x(ms_val):
            if total_ms <= 0:
                return 0.0
            sample_global = (ms_val / total_ms) * total_len
            norm_global = sample_global / total_len
            if end_norm == start_norm:
                rel = 0.0
            else:
                rel = (norm_global - start_norm) / (end_norm - start_norm)
            rel = max(0.0, min(1.0, rel))
            return rel * (len(view_samples_ds) - 1)

        # Selected range markers
        s_val = self.sel_start_ms
        e_val = self.sel_end_ms
        s_val = max(0.0, min(total_ms, s_val))
        e_val = max(s_val, min(total_ms, e_val))

        s_x = ms_to_x(s_val)
        e_x = ms_to_x(e_val)

        self.ax.axvline(s_x, color=COLOR_SUCCESS, lw=1.5)   # green start
        self.ax.axvline(e_x, color=ACCENT_PURPLE, lw=1.5)   # purple end

        # Highlight selection
        self.ax.axvspan(s_x, e_x, color="#333333", alpha=0.25)

        # Word markers
        for i, w in enumerate(self.words):
            s_eff, e_eff = self.get_effective_ms(w)
            sx = ms_to_x(s_eff)
            ex = ms_to_x(e_eff)
            if ex < 0 or sx > len(view_samples_ds) - 1:
                continue
            color = COLOR_MARGINAL if i == self.current_index else "#444444"
            self.ax.axvline(sx, color=color, lw=0.8, alpha=0.7)
            self.ax.axvline(ex, color=color, lw=0.8, alpha=0.7)

        # Playhead
        ph_norm = self.playhead_pos
        if end_norm > start_norm:
            rel_view = (ph_norm - start_norm) / (end_norm - start_norm)
        else:
            rel_view = 0.0
        rel_view = max(0.0, min(1.0, rel_view))
        ph_x = rel_view * (len(view_samples_ds) - 1)
        self.ax.axvline(ph_x, color=TIMELINE_CLICK, lw=1.5)

        self.draw_time_ruler(start_norm, end_norm)

        self.ax.axis("off")
        self.canvas.draw()

    def draw_time_ruler(self, start_norm, end_norm):
        total_ms = self.audio_length_ms()
        if total_ms <= 0:
            return

        total_sec = total_ms / 1000.0
        start_sec = start_norm * total_sec
        end_sec = end_norm * total_sec
        span_sec = max(end_sec - start_sec, 1e-6)

        rough_step = span_sec / 8.0
        if rough_step <= 0:
            return
        mag = 10 ** int(np.floor(np.log10(rough_step)))
        norm = rough_step / mag
        if norm < 1.5:
            step = 1 * mag
        elif norm < 3:
            step = 2 * mag
        elif norm < 7:
            step = 5 * mag
        else:
            step = 10 * mag

        sec = np.floor(start_sec / step) * step
        while sec <= end_sec + 1e-6:
            t_norm = sec / total_sec
            if end_norm > start_norm:
                rel_view = (t_norm - start_norm) / (end_norm - start_norm)
            else:
                rel_view = 0.0
            rel_view = max(0.0, min(1.0, rel_view))

            x_frac = rel_view
            pct = t_norm * 100.0
            label = f"{sec:0.1f}s | {pct:0.0f}%"

            self.ax.text(
                x_frac,
                -0.15,
                label,
                transform=self.ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color="#AAAAAA"
            )
            sec += step

    # -------------------- Slider / range --------------------
    def on_slider_change(self, _=None):
        if not self.audio_seg:
            return
        total_ms = self.audio_length_ms()
        s_val = float(self.start_slider.get() or 0.0)
        e_val = float(self.end_slider.get() or total_ms)
        s_val = max(0.0, min(total_ms, s_val))
        e_val = max(s_val, min(total_ms, e_val))
        self.sel_start_ms = s_val
        self.sel_end_ms = e_val
        self.update_range_label()
        self.update_plot()

    def update_range_label(self):
        dur = max(0.0, (self.sel_end_ms - self.sel_start_ms) / 1000.0)
        self.lbl_range.configure(
            text=f"Start: {self.sel_start_ms/1000.0:0.3f}s   "
                 f"End: {self.sel_end_ms/1000.0:0.3f}s   "
                 f"({dur:0.3f}s)"
        )

    # -------------------- Playback helpers --------------------
    def stop_playback(self):
        self.is_playing = False
        pygame.mixer.music.stop()
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        if self._playhead_updater_id is not None:
            self.after_cancel(self._playhead_updater_id)
            self._playhead_updater_id = None

    def _segment_ms(self, start_ms, end_ms):
        if not self.audio_seg:
            return None
        total_ms = self.audio_length_ms()
        start_ms = max(0.0, min(total_ms, start_ms))
        end_ms = max(start_ms, min(total_ms, end_ms))
        return self.audio_seg[int(start_ms):int(end_ms)]

    def play_from_playhead(self):
        if not self.audio_seg:
            return
        self.stop_playback()
        total_ms = self.audio_length_ms()
        ph_ms = self.playhead_pos * total_ms
        seg = self._segment_ms(ph_ms, total_ms)
        if not seg or len(seg) == 0:
            return
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        buf.seek(0)
        pygame.mixer.music.load(buf)
        pygame.mixer.music.play()
        self.is_playing = True

    def play_selection(self):
        if not self.audio_seg:
            return
        self.stop_playback()
        seg = self._segment_ms(self.sel_start_ms, self.sel_end_ms)
        if not seg or len(seg) == 0:
            return
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        buf.seek(0)
        pygame.mixer.music.load(buf)
        pygame.mixer.music.play()
        self.is_playing = True
        if self.audio_length_ms() > 0:
            self.playhead_pos = self.sel_start_ms / self.audio_length_ms()
            self.update_plot()

    def play_current_word(self):
        """Immediate manual playback (Down arrow, Play button)."""
        if not self.audio_seg or not self.words:
            return
        w = self.words[self.current_index]
        s_eff, e_eff = self.get_effective_ms(w)
        seg = self._segment_ms(s_eff, e_eff)
        if not seg or len(seg) == 0:
            return
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        buf.seek(0)
        with self._play_lock:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(buf)
            pygame.mixer.music.play()

    def play_current_word_auto(self):
        """Auto-play used for word activation or margin changes."""
        if not self.audio_seg or not self.words:
            return
        w = self.words[self.current_index]
        s_eff, e_eff = self.get_effective_ms(w)
        seg = self._segment_ms(s_eff, e_eff)
        if not seg or len(seg) == 0:
            return
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        buf.seek(0)
        with self._play_lock:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(buf)
            pygame.mixer.music.play()

    def schedule_margin_play(self, delay_ms=150):
        """Schedule a single playback after margins change."""
        if self._margin_play_after_id is not None:
            self.after_cancel(self._margin_play_after_id)
            self._margin_play_after_id = None
        self._margin_play_after_id = self.after(delay_ms, self._margin_play_callback)

    def _margin_play_callback(self):
        self._margin_play_after_id = None

    # -------------------- Transcription --------------------
    def transcribe_selected_async(self):
        if self.audio_np is None or self.audio_seg is None:
            messagebox.showerror("Error", "Load an audio file first.")
            return
        if self.sel_end_ms <= self.sel_start_ms:
            messagebox.showerror("Error", "Selected range is empty.")
            return
        if self._transcribe_thread and self._transcribe_thread.is_alive():
            return

        self.btn_transcribe.configure(state="disabled")
        self.transcribe_label.configure(text="Transcribing...", text_color=COLOR_MARGINAL)
        self._transcribe_thread = threading.Thread(
            target=self._transcribe_selected,
            daemon=True
        )
        self._transcribe_thread.start()

    def _transcribe_selected(self):
        start_time = time.time()
        try:
            total_ms = self.audio_length_ms()
            total_s = total_ms / 1000.0
            if total_s <= 0:
                raise RuntimeError("Audio length is zero.")

            full_len = len(self.audio_np)
            s_sec = self.sel_start_ms / 1000.0
            e_sec = self.sel_end_ms / 1000.0
            s_idx = int((s_sec / total_s) * full_len)
            e_idx = int((e_sec / total_s) * full_len)
            s_idx = max(0, min(full_len - 1, s_idx))
            e_idx = max(s_idx + 1, min(full_len, e_idx))

            segment_np = self.audio_np[s_idx:e_idx]
            sel_duration_s = (self.sel_end_ms - self.sel_start_ms) / 1000.0

            tmp_path = "_tmp_bt_range.wav"
            sf.write(tmp_path, segment_np, self.sr)

            model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(tmp_path, word_timestamps=True)

            os.remove(tmp_path)

            new_words = []
            for seg in segments:
                for w in seg.words:
                    if not w.word.strip():
                        continue
                    start_ms = self.sel_start_ms + float(w.start) * 1000.0
                    end_ms = self.sel_start_ms + float(w.end) * 1000.0
                    new_words.append(
                        {
                            "word": w.word.strip(),
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "m_start": 0.0,
                            "m_end": 0.0,
                            "g_start": 0.0,
                            "g_end": 0.0,
                            "export_path": "",
                        }
                    )

            self.words = new_words
            self.current_index = 0

            elapsed = time.time() - start_time
            num_words = len(self.words)
            rtf = sel_duration_s / elapsed if elapsed > 0 else 0.0

            print("\n" + "=" * 60)
            print("BOOTLEG TEXT SLICER - TRANSCRIPTION COMPLETE")
            if self.audio_path:
                print(f"File: {os.path.basename(self.audio_path)}")
            print(f"Selected Range: {self.sel_start_ms/1000.0:0.3f}s -> {self.sel_end_ms/1000.0:0.3f}s "
                  f"({sel_duration_s:0.3f}s)")
            print(f"Words Detected: {num_words}")
            print(f"Processing Time: {elapsed:0.2f}s")
            print(f"Efficiency: {rtf:0.2f}x Realtime")
            print("=" * 60 + "\n")

            self.after(0, lambda: self._after_transcribe_success(num_words))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Transcription failed:\n{e}"))
            self.after(0, self._after_transcribe_done)

    def _after_transcribe_success(self, num_words):
        self._after_transcribe_done()
        self._update_status_filename(extra=f"({num_words} words)")
        self.update_word_display()
        self.update_plot()
        messagebox.showinfo("Done", f"Transcription complete.\nWords detected: {num_words}")

    def _after_transcribe_done(self):
        self.btn_transcribe.configure(state="normal")
        self.transcribe_label.configure(text="Idle", text_color=TEXT_WHITE)

    # -------------------- Word / margins --------------------
    def get_effective_ms(self, w):
        # effective = base times + global + local
        g_start = w.get("g_start", 0.0)
        g_end = w.get("g_end", 0.0)
        s = w["start_ms"] + (g_start + w.get("m_start", 0.0)) * 1000.0
        e = w["end_ms"] + (g_end + w.get("m_end", 0.0)) * 1000.0
        return max(0.0, s), max(0.0, e)

    def _compute_current_global_margins(self):
        if not self.words:
            return 0.0, 0.0
        avg_start = sum(w.get("g_start", 0.0) for w in self.words) / len(self.words)
        avg_end = sum(w.get("g_end", 0.0) for w in self.words) / len(self.words)
        return avg_start, avg_end

    def _recompute_global_offsets_label(self):
        if not self.words:
            self.lbl_global_offsets.configure(
                text="Global: start +0.000s, end +0.000s"
            )
            return
        g_start, g_end = self._compute_current_global_margins()
        self.lbl_global_offsets.configure(
            text=f"Global: start {g_start:+0.3f}s, end {g_end:+0.3f}s"
        )

    def _update_indiv_offsets_label(self):
        if not self.words:
            self.lbl_indiv_offsets.configure(
                text="Local: start +0.000s, end +0.000s"
            )
            return
        w = self.words[self.current_index]
        self.lbl_indiv_offsets.configure(
            text=f"Local: start {w.get('m_start', 0.0):+0.3f}s, "
                 f"end {w.get('m_end', 0.0):+0.3f}s"
        )

    def adj_global(self, key, val):
        # key is "m_start" or "m_end" but we treat it as global in g_start/g_end
        if key == "m_start":
            for w in self.words:
                w["g_start"] = round(w.get("g_start", 0.0) + val, 4)
        else:
            for w in self.words:
                w["g_end"] = round(w.get("g_end", 0.0) + val, 4)
        self._recompute_global_offsets_label()
        self.update_word_display()
        self.update_plot()
        # Only schedule delayed play (no direct play) to avoid double-play
        self.schedule_margin_play()

    def adj_individual(self, key, val):
        if not self.words:
            return
        w = self.words[self.current_index]
        w[key] = round(w.get(key, 0.0) + val, 4)
        self._update_indiv_offsets_label()
        self.update_word_display()
        self.update_plot()
        # Only schedule delayed play (no direct play) to avoid double-play
        self.schedule_margin_play()

    def prev_word(self):
        if not self.words:
            return
        self.current_index = max(0, self.current_index - 1)
        self.update_word_display()
        self.update_plot()

    def next_word(self):
        if not self.words:
            return
        if self.current_index + 1 >= len(self.words):
            messagebox.showinfo("Done", "All words have been processed.")
            return
        self.current_index += 1
        self.update_word_display()
        self.update_plot()

    def skip_current_word(self):
        if not self.words:
            return
        if self.current_index == len(self.words) - 1:
            messagebox.showinfo("Done", "All words have been processed (last word skipped).")
            return
        self.current_index += 1
        self.update_word_display()
        self.update_plot()

    def approve_current_word(self):
        if not self.audio_seg or not self.words:
            return
        w = self.words[self.current_index]

        # Make sure g_start/g_end reflect the current global margins
        g_start, g_end = self._compute_current_global_margins()
        w["g_start"] = g_start
        w["g_end"] = g_end

        s_eff, e_eff = self.get_effective_ms(w)
        seg = self._segment_ms(s_eff, e_eff)
        if not seg or len(seg) == 0:
            return

        if not os.path.exists("ApprovedWords"):
            os.makedirs("ApprovedWords")

        export_name = self.entry_export_name.get().strip()
        if not export_name:
            clean = "".join(x for x in w["word"] if x.isalnum())
            export_name = clean if clean else f"word_{self.current_index+1}"

        filename = f"{export_name}_{int(time.time())}.wav"
        path = os.path.join("ApprovedWords", filename)
        seg.export(path, format="wav")

        w["export_path"] = path

        base_s_ms = w["start_ms"]
        base_e_ms = w["end_ms"]
        # Persist global + local margins and export path
        line = (
            f"{export_name},"
            f"{base_s_ms:.4f},"
            f"{base_e_ms:.4f},"
            f"{w.get('g_start', 0.0):.4f},"
            f"{w.get('g_end', 0.0):.4f},"
            f"{w.get('m_start', 0.0):.4f},"
            f"{w.get('m_end', 0.0):.4f},"
            f"{path}\n"
        )
        with open(CUT_TEMPLATE_FILE, "a", encoding="utf-8") as f:
            f.write(line)

        print(f"[EXPORT] Saved {path} | Duration: {(e_eff - s_eff)/1000.0:.3f}s")

        self.entry_export_name.delete(0, "end")

        if self.current_index == len(self.words) - 1:
            messagebox.showinfo("Done", "All words have been exported.")
        else:
            self.current_index += 1
            self.update_word_display()
            self.update_plot()

    def update_word_display(self):
        if not self.words:
            self.lbl_word.configure(text="No word")
            self.lbl_word_index.configure(text="")
            self.lbl_margins.configure(text="")
            self.lbl_stats.configure(text="")
            self.ax_word.clear()
            self.ax_word.axis("off")
            self.canvas_word.draw()
            self._update_indiv_offsets_label()
            self._recompute_global_offsets_label()
            self._update_status_filename(extra="")
            return

        w = self.words[self.current_index]
        s_eff, e_eff = self.get_effective_ms(w)
        dur = max(0.0, (e_eff - s_eff) / 1000.0)

        self.lbl_word.configure(text=f"'{w['word']}'")
        self.lbl_word_index.configure(
            text=f"Word {self.current_index+1} / {len(self.words)}"
        )
        self.lbl_margins.configure(
            text=f"Offsets -> Global: {w.get('g_start', 0.0):+0.3f}s/{w.get('g_end', 0.0):+0.3f}s | "
                 f"Local: {w.get('m_start', 0.0):+0.3f}s/{w.get('m_end', 0.0):+0.3f}s"
        )
        self.lbl_stats.configure(
            text=f"Base: {w['start_ms']/1000.0:0.3f}s-{w['end_ms']/1000.0:0.3f}s | "
                 f"Final duration: {dur:0.3f}s"
        )

        clean = "".join(x for x in w["word"] if x.isalnum())
        self.entry_export_name.delete(0, "end")
        if clean:
            self.entry_export_name.insert(0, clean)

        self.ax_word.clear()
        if self.audio_seg:
            seg = self._segment_ms(s_eff, e_eff)
            if seg and len(seg) > 0:
                data = np.array(seg.get_array_of_samples(), dtype=np.float32)
                if len(data) > 20000:
                    factor = len(data) // 20000
                    data = data[::factor]
                self.ax_word.plot(
                    data,
                    color=ACCENT_BLUE,
                    linewidth=1.0,
                    alpha=0.9
                )
        self.ax_word.axis("off")
        self.canvas_word.draw()

        self._update_indiv_offsets_label()
        self._recompute_global_offsets_label()
        self._update_status_filename(extra=f"({len(self.words)} words)")

        # Auto-play tiny snippet once whenever a word becomes active
        self.play_current_word_auto()


if __name__ == "__main__":
    app = BootlegTextSlicer()
    app.mainloop()

