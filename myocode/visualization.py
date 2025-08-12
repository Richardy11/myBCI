"""
Lightweight visualization utilities for realtime UI during prediction.

Currently provides:
- RealtimeLabelDisplay: a centered, large-font text window that shows
  current classifier output. Designed to be minimal but extensible
  (e.g., future confidence bars, shortcuts, etc.).
"""

from __future__ import annotations

import time
from typing import Optional, Callable, List


class RealtimeLabelDisplay:
    """Matplotlib-based window showing current label in large font.

    - Designed to be resilient in headless/CI environments: if matplotlib
      cannot be imported or a backend cannot be initialized, it degrades to
      a no-op implementation without raising.
    - Call update(text) as frequently as you like; internal rate-limiter
      ensures the window refreshes at most ~30 FPS.
    """

    def __init__(self, title: str = "Prediction",
                 actions: Optional[List[str]] = None,
                 on_start_adapt: Optional[Callable[[str, float], None]] = None) -> None:
        self._available = False
        self._last_draw = 0.0
        self._min_interval = 1 / 30.0  # 30 FPS
        self._plt = None
        self._fig = None
        self._ax = None
        self._text = None

        try:
            import matplotlib.pyplot as plt  # type: ignore
            import matplotlib.gridspec as gridspec  # type: ignore
            from matplotlib.widgets import RadioButtons, Button, TextBox  # type: ignore

            # Try to initialize a simple figure
            self._plt = plt
            # Layout: 1 row, 2 cols (left: label; right: controls)
            self._fig = plt.figure(figsize=(7, 4))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.2)

            # Left label area
            self._ax = self._fig.add_subplot(gs[0, 0])
            self._ax.set_axis_off()
            self._text = self._ax.text(0.5, 0.5, "...", fontsize=36,
                                       ha="center", va="center", transform=self._ax.transAxes)
            # Status line (bottom)
            self._status = self._fig.text(0.5, 0.05, "", ha='center', va='center', fontsize=10)
            self._status_until = 0.0
            try:
                self._fig.canvas.manager.set_window_title(title)  # type: ignore[attr-defined]
            except Exception:
                pass

            # Optional right-side controls
            self._on_start_adapt = on_start_adapt
            self._radio = None
            self._txt = None
            self._btn = None
            self._selected_action = None
            if actions:
                ax_ctrl = self._fig.add_subplot(gs[0, 1])
                ax_ctrl.set_axis_off()
                # Radio buttons for actions (vertical)
                ax_radio = self._fig.add_axes([0.66, 0.45, 0.30, 0.45])  # [left,bottom,width,height] in fig coords
                self._radio = RadioButtons(ax_radio, actions)
                self._selected_action = actions[0]
                def _on_radio(label):
                    self._selected_action = label
                self._radio.on_clicked(_on_radio)

                # TextBox for duration seconds
                ax_txt = self._fig.add_axes([0.66, 0.33, 0.30, 0.07])
                self._txt = TextBox(ax_txt, "Seconds", initial="3.0")

                # Start button
                ax_btn = self._fig.add_axes([0.66, 0.20, 0.30, 0.08])
                self._btn = Button(ax_btn, "Start Online Train")
                def _on_click(_event):
                    if self._on_start_adapt and self._selected_action:
                        try:
                            sec = float(self._txt.text) if self._txt else 3.0
                        except Exception:
                            sec = 3.0
                        self._on_start_adapt(self._selected_action, max(0.2, float(sec)))
                self._btn.on_clicked(_on_click)

            # Mark available only if we reached this point
            self._available = True
        except Exception:
            # No-op fallback; keep _available=False
            self._available = False

    def update(self, label_text: str) -> None:
        if not self._available:
            return
        now = time.time()
        if self._text is not None:
            self._text.set_text(label_text)
        # auto-clear status if expired
        if self._status is not None and now >= self._status_until and self._status.get_text():
            try:
                self._status.set_text("")
            except Exception:
                pass
        if (now - self._last_draw) >= self._min_interval:
            try:
                self._fig.canvas.draw_idle()  # type: ignore[union-attr]
                self._plt.pause(0.001)  # type: ignore[union-attr]
            except Exception:
                # If drawing fails once, degrade to no-op
                self._available = False
            self._last_draw = now

    def close(self) -> None:
        if not self._available:
            return
        try:
            self._plt.close(self._fig)  # type: ignore[union-attr]
        except Exception:
            pass

    # ---- status helpers ----
    def show_status(self, msg: str, duration: float = 2.0):
        if not self._available:
            return
        try:
            self._status.set_text(msg)
            self._status_until = time.time() + max(0.2, float(duration))
            # draw on main thread only; avoid starting GUI in background threads
            try:
                import threading
                if threading.current_thread() is threading.main_thread():
                    self._fig.canvas.draw_idle()
                    self._plt.pause(0.001)
            except Exception:
                pass
        except Exception:
            pass


