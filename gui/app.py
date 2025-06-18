import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading

from model import train_model, predict
from config import DEFAULT_PREDICT_LIMIT

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Premier League Predictor")
        self._build_ui()

    def _build_ui(self):
        tk.Label(self, text="Season:").grid(row=0, column=0, padx=5, pady=5)
        self.season_cb = ttk.Combobox(self, values=[str(y) for y in [2023, 2024]])
        self.season_cb.current(0)
        self.season_cb.grid(row=0, column=1, padx=5)

        self.train_btn = tk.Button(self, text="Train Model", command=self._start_train)
        self.train_btn.grid(row=1, column=0, padx=5, pady=5)
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.grid(row=1, column=1, padx=5)

        self.predict_btn = tk.Button(self, text="Predict", command=self._do_predict)
        self.predict_btn.grid(row=2, column=0, padx=5, pady=5)
        self.limit_entry = tk.Entry(self, width=5)
        self.limit_entry.insert(0, str(DEFAULT_PREDICT_LIMIT))
        self.limit_entry.grid(row=2, column=1, padx=5)

        self.output = scrolledtext.ScrolledText(self, width=60, height=20)
        self.output.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def _start_train(self):
        def task():
            try:
                self.progress.start()
                path = train_model()
                messagebox.showinfo("Success", f"Model saved to {path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                self.progress.stop()
        threading.Thread(target=task).start()

    def _do_predict(self):
        try:
            limit = int(self.limit_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid limit")
            return
        self.output.delete('1.0', tk.END)
        results = predict(limit=limit)
        for r in results:
            line = f"{r['date']} {r['home']} vs {r['away']}: {r['prob']['home_win']*100:.1f}% / {r['prob']['draw']*100:.1f}% / {r['prob']['away_win']*100:.1f}%\n"
            self.output.insert(tk.END, line)

if __name__ == "__main__":
    App().mainloop()
