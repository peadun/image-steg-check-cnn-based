import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
import subprocess
import json

# WSL Python environment and inference scripts
_WSL_PYTHON        = "/home/farhad/srnet_env/bin/python"
_WSL_SCRIPT_SRNET  = "/mnt/c/Users/Ahmad Farhad/Documents/UniKL/FYP/srnet_inference.py"
_WSL_SCRIPT_EFFNET = "/mnt/c/Users/Ahmad Farhad/Documents/UniKL/FYP/efficientnet_inference.py"


def _windows_to_wsl_path(win_path: str) -> str:
    """Convert a Windows absolute path to its /mnt/<drive>/... WSL equivalent."""
    path = win_path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path = f"/mnt/{drive}" + path[2:]
    return path


def _run_wsl_inference(image_path: str, model: str = "srnet"):
    """
    Call the chosen inference script inside WSL and return (is_stego, confidence).
    model: "srnet" or "efficientnet"
    Raises RuntimeError on any failure.
    """
    wsl_image = _windows_to_wsl_path(image_path)
    script = _WSL_SCRIPT_EFFNET if model == "efficientnet" else _WSL_SCRIPT_SRNET
    # Use --exec to invoke Python directly (no shell), so spaces in paths
    # are handled at the OS level without any quoting needed.
    result = subprocess.run(
        ["wsl", "--exec", _WSL_PYTHON, script, wsl_image],
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    raw = lines[-1] if lines else ""
    if not raw:
        raise RuntimeError(
            f"WSL inference returned no output.\nstderr: {result.stderr.strip()}"
        )
    data = json.loads(raw)
    if "error" in data:
        raise RuntimeError(data["error"])
    return bool(data["is_stego"]), float(data["confidence"])

class SteganographyDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Steganography Detection System")
        self.root.geometry("900x800")
        self.root.configure(bg="#2b1e3b")
        
        # Variables
        self.selected_image_path = None
        self.image_display = None
        self.is_analyzing = False
        self.model_var = tk.StringVar(value="efficientnet")
        
        # Create UI
        self.create_header()
        self.create_main_content()
        self.create_footer()
        
    def create_header(self):
        """Create header section"""
        header_frame = tk.Frame(self.root, bg="#2e1e3b")
        header_frame.pack(pady=20, padx=20, fill="x")
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="🛡️ Steganography Detection using CNN",
            font=("Arial", 24, "bold"),
            bg="#2a1e3b",
            fg="#c084fc"
        )
        title_label.pack()
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="CNN Models for Image Analysis",
            font=("Arial", 12),
            bg="#2e1e3b",
            fg="#d8b4fe"
        )
        subtitle_label.pack(pady=(5, 0))

        # Model selector
        model_frame = tk.Frame(header_frame, bg="#2e1e3b")
        model_frame.pack(pady=(10, 0))

        tk.Label(
            model_frame, text="Models:",
            font=("Arial", 10, "bold"),
            bg="#2e1e3b", fg="#d8b4fe"
        ).pack(side="left", padx=(0, 8))

        for label, value in [("EfficientNet-B0 (JPEG)", "efficientnet"),
                             ("SRNet (PNG/PGM)", "srnet")]:
            tk.Radiobutton(
                model_frame, text=label, variable=self.model_var, value=value,
                font=("Arial", 10), bg="#2e1e3b", fg="#d8b4fe",
                selectcolor="#4c1d95", activebackground="#2e1e3b",
                activeforeground="#c084fc", cursor="hand2",
                command=self._on_model_change
            ).pack(side="left", padx=6)

    def create_main_content(self):
        """Create main content area"""
        # Outer container (holds canvas + scrollbar)
        outer_frame = tk.Frame(self.root, bg="#ffffff", relief="raised", bd=2)
        outer_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # Scrollbar
        scrollbar = tk.Scrollbar(outer_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        # Canvas — everything scrollable goes inside here
        self._canvas = tk.Canvas(outer_frame, bg="#ffffff",
                                 yscrollcommand=scrollbar.set,
                                 highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self._canvas.yview)

        # Inner frame placed on the canvas
        main_frame = tk.Frame(self._canvas, bg="#ffffff")
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=main_frame, anchor="nw"
        )

        # Keep canvas window width in sync with canvas width
        def _on_canvas_resize(event):
            self._canvas.itemconfig(self._canvas_window, width=event.width)
        self._canvas.bind("<Configure>", _on_canvas_resize)

        # Update scroll region whenever inner frame changes size
        def _on_frame_resize(event):
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        main_frame.bind("<Configure>", _on_frame_resize)

        # Mousewheel scroll (Windows)
        def _on_mousewheel(event):
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Section title
        section_header = tk.Frame(main_frame, bg="#7c3aed")
        section_header.pack(fill="x")

        section_title = tk.Label(
            section_header,
            text="📁 Image Upload",
            font=("Arial", 16, "bold"),
            bg="#7c3aed",
            fg="white",
            anchor="w"
        )
        section_title.pack(pady=15, padx=20, fill="x")

        self.section_subtitle = tk.Label(
            section_header,
            text="Upload a JPEG image to detect hidden steganographic content",
            font=("Arial", 9),
            bg="#7c3aed",
            fg="#ede9fe",
            anchor="w"
        )
        self.section_subtitle.pack(pady=(0, 15), padx=20, fill="x")

        # Content area
        self.content_frame = tk.Frame(main_frame, bg="#ffffff")
        content_frame = self.content_frame
        content_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Upload area frame
        self.upload_frame = tk.Frame(
            content_frame,
            bg="#f3e8ff",
            relief="solid",
            bd=2,
            highlightbackground="#d8b4fe",
            highlightthickness=2
        )
        self.upload_frame.pack(fill="both", expand=True)
        
        # Upload button and text
        upload_icon = tk.Label(
            self.upload_frame,
            text="📤",
            font=("Arial", 48),
            bg="#f3e8ff",
            fg="#a855f7"
        )
        upload_icon.pack(pady=(40, 10))
        
        upload_text = tk.Label(
            self.upload_frame,
            text="Click to Upload Image",
            font=("Arial", 14, "bold"),
            bg="#f3e8ff",
            fg="#6d28d9"
        )
        upload_text.pack()
        
        self.upload_subtext = tk.Label(
            self.upload_frame,
            text="Supported formats: JPEG, JPG (Max 10MB)",
            font=("Arial", 9),
            bg="#f3e8ff",
            fg="#64748b"
        )
        self.upload_subtext.pack(pady=(5, 0))
        
        # Upload button
        self.upload_btn = tk.Button(
            self.upload_frame,
            text="Browse Files",
            font=("Arial", 11, "bold"),
            bg="#a855f7",
            fg="white",
            relief="flat",
            cursor="hand2",
            padx=30,
            pady=10,
            command=self.upload_image
        )
        self.upload_btn.pack(pady=20)
        
        # Image preview frame (initially hidden)
        self.preview_frame = tk.Frame(content_frame, bg="#f1f5f9")
        
        # Buttons frame
        self.buttons_frame = tk.Frame(main_frame, bg="#ffffff")
        self.buttons_frame.pack(pady=10, padx=20, fill="x")
        
        # Analyze button
        self.analyze_btn = tk.Button(
            self.buttons_frame,
            text="Analyse for Steganography",
            font=("Arial", 12, "bold"),
            bg="#7c3aed",
            fg="white",
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=12,
            command=self.analyze_image,
            state="disabled"
        )
        self.analyze_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        # Upload different button
        self.reupload_btn = tk.Button(
            self.buttons_frame,
            text="Upload Different Image",
            font=("Arial", 12, "bold"),
            bg="white",
            fg="#7c3aed",
            relief="solid",
            bd=2,
            cursor="hand2",
            padx=20,
            pady=12,
            command=self.upload_image,
            state="disabled"
        )
        self.reupload_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))
        
        # Results frame (initially hidden)
        self.results_frame = tk.Frame(main_frame, bg="#ffffff")
    
    def create_footer(self):
        """Create footer section"""
        footer_frame = tk.Frame(self.root, bg="#2c1e3b")
        footer_frame.pack(pady=10, padx=20, fill="x")
        
        footer_text2 = tk.Label(
            footer_frame,
            text="",
            font=("Arial", 9),
            bg="#2e1e3b",
            fg="#d8b4fe"
        )
        footer_text2.pack()
    
    def _on_model_change(self):
        """Update UI text when model selection changes."""
        if self.model_var.get() == "efficientnet":
            self.section_subtitle.config(text="Upload a JPEG image to detect hidden steganographic content")
            if self.upload_subtext.winfo_exists():
                self.upload_subtext.config(text="Supported formats: JPEG, JPG (Max 10MB)")
        else:
            self.section_subtitle.config(text="Upload a PNG/PGM image to detect hidden steganographic content")
            if self.upload_subtext.winfo_exists():
                self.upload_subtext.config(text="Supported formats: PNG, PGM (Max 10MB)")
        # Clear current selection when switching models
        self.selected_image_path = None
        self.analyze_btn.config(state="disabled")
        self.reupload_btn.config(state="disabled")
        self.results_frame.pack_forget()
        self.content_frame.pack_configure(expand=True)

    def upload_image(self):
        """Handle image upload"""
        if self.model_var.get() == "efficientnet":
            filetypes = [
                ("JPEG files", "*.jpg *.jpeg"),
                ("JPEG", "*.jpg"),
                ("JPEG", "*.jpeg"),
            ]
        else:
            filetypes = [
                ("Image files", "*.png *.pgm"),
                ("PNG files", "*.png"),
                ("PGM files", "*.pgm"),
            ]
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=filetypes,
        )
        
        if file_path:
            # Validate file size (max 10MB)
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:
                messagebox.showerror("Error", "File size too large. Maximum size is 10MB.")
                return
            
            self.selected_image_path = file_path
            self.display_image_preview()
            self.analyze_btn.config(state="normal")
            self.reupload_btn.config(state="normal")

            # Hide results and restore upload area expansion
            self.results_frame.pack_forget()
            self.content_frame.pack_configure(expand=True)
    
    def display_image_preview(self):
        """Display selected image preview"""
        # Clear upload frame
        for widget in self.upload_frame.winfo_children():
            widget.destroy()
        
        # Show preview frame
        self.preview_frame.pack_forget()
        self.preview_frame = tk.Frame(self.upload_frame, bg="#f1f5f9")
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Load and display image
        try:
            img = Image.open(self.selected_image_path)
            
            # Resize image to fit display (max 400x200)
            img.thumbnail((400, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Image label
            img_label = tk.Label(self.preview_frame, image=photo, bg="#f1f5f9")
            img_label.image = photo  # Keep a reference
            img_label.pack(pady=10)
            
            # File info
            file_name = os.path.basename(self.selected_image_path)
            file_size_kb = os.path.getsize(self.selected_image_path) / 1024
            
            info_frame = tk.Frame(self.preview_frame, bg="#f1f5f9")
            info_frame.pack(pady=5)
            
            name_label = tk.Label(
                info_frame,
                text=f"📄 {file_name}",
                font=("Arial", 10, "bold"),
                bg="#f1f5f9",
                fg="#6d28d9"
            )
            name_label.pack()
            
            size_label = tk.Label(
                info_frame,
                text=f"Size: {file_size_kb:.2f} KB",
                font=("Arial", 9),
                bg="#f1f5f9",
                fg="#64748b"
            )
            size_label.pack()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def analyze_image(self):
        """Analyze image for steganography"""
        if not self.selected_image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        # Disable buttons during analysis
        self.analyze_btn.config(state="disabled", text="Analysing...")
        self.reupload_btn.config(state="disabled")

        # Clear any previous results immediately and show loading message
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.content_frame.pack_configure(expand=False)
        self._loading_label = tk.Label(
            self.results_frame,
            text="Running SRNet inference, please wait...",
            font=("Arial", 10, "italic"),
            bg="#ffffff",
            fg="#6d28d9",
        )
        self.results_frame.pack(pady=10, padx=20, fill="x")
        self._loading_label.pack(pady=(8, 2))

        self._progress_bar = ttk.Progressbar(
            self.results_frame, mode="indeterminate", length=400,
            style="Purple.Horizontal.TProgressbar"
        )
        self._progress_bar.pack(pady=(0, 8))
        self._progress_bar.start(12)
        self.root.update()

        # Run inference in a background thread to keep the UI responsive
        path  = self.selected_image_path
        model = self.model_var.get()
        loading_text = ("Running EfficientNet inference, please wait..."
                        if model == "efficientnet"
                        else "Running SRNet inference, please wait...")
        self._loading_label.config(text=loading_text)
        self.root.update()

        def _run():
            try:
                is_stego, confidence = _run_wsl_inference(path, model)
                self.root.after(0, lambda: self.show_results(is_stego, confidence))
            except Exception as exc:
                def _on_error(e=exc):
                        if hasattr(self, '_progress_bar') and self._progress_bar.winfo_exists():
                            self._progress_bar.stop()
                        for widget in self.results_frame.winfo_children():
                            widget.destroy()
                        messagebox.showerror("Inference Error", str(e))
                        self.analyze_btn.config(state="normal", text="Analyse for Steganography")
                        self.reupload_btn.config(state="normal")
                self.root.after(0, _on_error)

        threading.Thread(target=_run, daemon=True).start()
    
    def show_results(self, is_stego: bool, confidence: float):
        """Display analysis results from SRNet inference."""
        # Stop and clear loading bar
        if hasattr(self, '_progress_bar') and self._progress_bar.winfo_exists():
            self._progress_bar.stop()
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Shrink upload area so results frame gets vertical space
        self.content_frame.pack_configure(expand=False)
        self.results_frame.pack(pady=10, padx=20, fill="x")
        
        # Results container
        if is_stego:
            bg_color = "#fef2f2"
            border_color = "#fca5a5"
            text_color = "#991b1b"
            icon = "⚠️"
            title = "Steganography Detected"
            message = "Hidden data patterns detected in the image. This image may contain concealed information."
        else:
            bg_color = "#f0fdf4"
            border_color = "#86efac"
            text_color = "#166534"
            icon = "✅"
            title = "Clean Image"
            message = "No steganographic artifacts detected. This appears to be a clean image."
        
        result_container = tk.Frame(
            self.results_frame,
            bg=bg_color,
            relief="solid",
            bd=2,
            highlightbackground=border_color,
            highlightthickness=2
        )
        result_container.pack(fill="x", pady=5)
        
        # Icon and title
        header_frame = tk.Frame(result_container, bg=bg_color)
        header_frame.pack(fill="x", padx=15, pady=(15, 5))
        
        icon_label = tk.Label(
            header_frame,
            text=icon,
            font=("Arial", 32),
            bg=bg_color
        )
        icon_label.pack(side="left", padx=(0, 10))
        
        text_frame = tk.Frame(header_frame, bg=bg_color)
        text_frame.pack(side="left", fill="x", expand=True)
        
        title_label = tk.Label(
            text_frame,
            text=title,
            font=("Arial", 16, "bold"),
            bg=bg_color,
            fg=text_color,
            anchor="w"
        )
        title_label.pack(fill="x")
        
        message_label = tk.Label(
            text_frame,
            text=message,
            font=("Arial", 10),
            bg=bg_color,
            fg=text_color,
            anchor="w",
            wraplength=650,
            justify="left"
        )
        message_label.pack(fill="x", pady=(5, 0))
        
        # Confidence score
        confidence_frame = tk.Frame(result_container, bg="white", relief="solid", bd=1)
        confidence_frame.pack(fill="x", padx=15, pady=15)
        
        conf_header = tk.Frame(confidence_frame, bg="white")
        conf_header.pack(fill="x", padx=10, pady=(10, 5))
        
        tk.Label(
            conf_header,
            text="Confidence Score:",
            font=("Arial", 10, "bold"),
            bg="white",
            fg="#374151"
        ).pack(side="left")
        
        tk.Label(
            conf_header,
            text=f"{confidence}%",
            font=("Arial", 14, "bold"),
            bg="white",
            fg=text_color
        ).pack(side="right")
        
        # Progress bar — draw after layout so winfo_width() is valid
        progress_bg = tk.Canvas(confidence_frame, height=12, bg="#e5e7eb", highlightthickness=0)
        progress_bg.pack(fill="x", padx=10, pady=(0, 10))

        bar_color = "#ef4444" if is_stego else "#22c55e"
        def _draw_bar(event, c=progress_bg, pct=confidence, col=bar_color):
            c.delete("bar")
            c.create_rectangle(0, 0, pct / 100 * event.width, 12,
                               fill=col, outline="", tags="bar")
        progress_bg.bind("<Configure>", _draw_bar)
        
        # Security advisory for detected steganography
        if is_stego:
            advisory_frame = tk.Frame(result_container, bg="#fef3c7", relief="solid", bd=1)
            advisory_frame.pack(fill="x", padx=15, pady=(0, 15))
            
            advisory_label = tk.Label(
                advisory_frame,
                text="⚠️ Security Advisory: This image may contain hidden information. Further investigation is recommended.",
                font=("Arial", 9),
                bg="#fef3c7",
                fg="#92400e",
                wraplength=700,
                justify="left"
            )
            advisory_label.pack(padx=10, pady=10)
        
        # Re-enable buttons
        self.analyze_btn.config(state="normal", text="Analyze for Steganography")
        self.reupload_btn.config(state="normal")

def main():
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("default")
    style.configure(
        "Purple.Horizontal.TProgressbar",
        troughcolor="#e9d5ff",
        background="#7c3aed",
        bordercolor="#e9d5ff",
        lightcolor="#7c3aed",
        darkcolor="#7c3aed",
    )
    app = SteganographyDetectionUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()