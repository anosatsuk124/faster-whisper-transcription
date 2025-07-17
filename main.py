import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from faster_whisper import WhisperModel

import os

if os.name == 'nt':
    os.add_dll_directory(r"C:\Program Files\NVIDIA\CUDNN\v9.11\bin\12.9")

class TranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcriber")
        self.root.geometry("600x400")

        # selected file path
        self.audio_file = None

        self.model_name = "kotoba-tech/kotoba-whisper-v2.0-faster"  # default model
        self.is_gpu = False  # default to CPU

        # UI elements
        self.create_widgets()

    def create_widgets(self):
        # File selection
        file_frame = ttk.Frame(self.root, padding=10)
        file_frame.pack(fill="x")

        self.file_label = ttk.Label(file_frame, text="音声ファイル未選択")
        self.file_label.pack(side="left", expand=True)

        select_btn = ttk.Button(file_frame, text="ファイル選択", command=self.select_file)
        select_btn.pack(side="right")

        # Transcribe button
        transcribe_frame = ttk.Frame(self.root, padding=10)
        transcribe_frame.pack(fill="x")

        self.transcribe_btn = ttk.Button(transcribe_frame, text="文字起こし開始", command=self.start_transcription, state="disabled")
        self.transcribe_btn.pack()

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=(0,10))

        # Text area for output
        text_frame = ttk.Frame(self.root, padding=10)
        text_frame.pack(fill="both", expand=True)

        self.text_widget = tk.Text(text_frame, wrap="word")
        self.text_widget.pack(fill="both", expand=True)

        # Checkbox for GPU usage
        gpu_frame = ttk.Frame(self.root, padding=10)
        gpu_frame.pack(fill="x")
        self.gpu_var = tk.BooleanVar(value=False)
        gpu_checkbox = ttk.Checkbutton(gpu_frame, text="GPUを使用する", variable=self.gpu_var, command=self.toggle_gpu)
        gpu_checkbox.pack(side="left")
        self.is_gpu = self.gpu_var.get()
        # Model selection
        model_frame = ttk.Frame(self.root, padding=10)
        model_frame.pack(fill="x")
        self.model_var = tk.StringVar(value=self.model_name)
        model_label = ttk.Label(model_frame, text="モデル:")
        model_label.pack(side="left")
        model_entry = ttk.Entry(model_frame, textvariable=self.model_var)
        model_entry.pack(side="left", fill="x", expand=True)
        model_entry.bind("<Return>", self.update_model)
        model_entry.bind("<FocusOut>", self.update_model)

    def toggle_gpu(self):
        self.is_gpu = self.gpu_var.get()
        if self.is_gpu:
            messagebox.showinfo("GPU使用", "GPUを使用する設定になりました。")
        else:
            messagebox.showinfo("CPU使用", "CPUを使用する設定になりました。")

    def update_model(self, event=None):
        new_model = self.model_var.get().strip()
        if new_model:
            self.model_name = new_model
            messagebox.showinfo("モデル更新", f"使用するモデルを '{self.model_name}' に更新しました。")
        else:
            messagebox.showwarning("モデル更新", "モデル名が空です。デフォルトのモデルを使用します。")
            self.model_name = "kotoba-tech/kotoba-whisper-v2.0-faster"
            self.model_var.set(self.model_name)

    def select_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.flac"), ("All Files", "*.*")]
        )
        if path:
            self.audio_file = path
            self.file_label.config(text=path)
            self.transcribe_btn.config(state="normal")

    def start_transcription(self):
        if not self.audio_file:
            return
        # disable UI
        self.transcribe_btn.config(state="disabled")
        self.progress.start()
        self.text_widget.delete(1.0, tk.END)
        # run in thread to keep UI responsive
        thread = threading.Thread(target=self.run_transcription, daemon=True)
        thread.start()

    def run_transcription(self):
        if self.is_gpu:
            model = WhisperModel(self.model_name, device="cuda", compute_type="int8_float16")
        else:
            model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

        full_text = ""
        try:
            if not self.audio_file:
                messagebox.showerror("エラー", "音声ファイルが選択されていません。")
                return
            segments, _ = model.transcribe(self.audio_file, chunk_length=15, condition_on_previous_text=False, language="ja")
            for segment in segments:
                line = "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
                full_text += segment.text
                # insert into text widget in UI thread
                self.text_widget.insert(tk.END, line)
                self.text_widget.see(tk.END)

            # final full transcription
            self.text_widget.insert(tk.END, "\n=== 完全テキスト ===\n" + full_text + "\n")
            # save to file
            out_path = f"{self.audio_file}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                for segment in segments:
                    f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
                f.write("\n" + full_text + "\n")
            messagebox.showinfo("完了", f"文字起こしが完了し、{out_path} に保存しました。")
        except Exception as e:
            messagebox.showerror("エラー", f"文字起こし中にエラーが発生しました:\n{e}")
        finally:
            # re-enable UI
            self.progress.stop()
            self.transcribe_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriberApp(root)
    root.mainloop()
