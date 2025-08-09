# realtime_emg_plot.py
import asyncio
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class EMGPlotter:
    """
    实时绘 8 通道 EMG。把 (ts_ns, (ch1..ch8)) 流喂进来即可。
    - srate: 估计采样率（Myo ~200Hz）
    - span_s: 横轴显示秒数
    """
    def __init__(self, srate: int = 200, span_s: float = 5.0):
        self.srate = srate
        self.span_s = span_s
        self.N = max(10, int(srate * span_s))

        # 每通道一个环形缓冲
        self.buf = [deque([0]*self.N, maxlen=self.N) for _ in range(8)]
        self.x = np.linspace(-self.span_s, 0.0, self.N)

        # 画布
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.set_xlim(-self.span_s, 0.0)

        # int8 范围很小，做垂直错位
        self.offset = 80
        self.y0 = np.arange(8) * self.offset
        self.lines = []
        for i in range(8):
            (ln,) = self.ax.plot(self.x, np.array(self.buf[i]) + self.y0[i], lw=1)
            self.lines.append(ln)

        self.ax.set_yticks(self.y0)
        self.ax.set_yticklabels([f"ch{i+1}" for i in range(8)])
        self.ax.set_ylim(-self.offset, self.y0[-1] + self.offset)
        self.ax.set_xlabel("time (s)")
        self.ax.set_title("Realtime EMG (8ch)")

        self._upd_interval = 1/30  # ~30 FPS
        self._last_upd = 0.0

    def _push(self, chs):
        # chs: 长度8的可迭代
        for i, v in enumerate(chs):
            self.buf[i].append(int(v))

    def _draw_once(self):
        for i in range(8):
            y = np.array(self.buf[i]) + self.y0[i]
            self.lines[i].set_data(self.x, y)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    async def run(self, source_iter):
        """
        source_iter: 异步可迭代，yield (ts_ns, (ch1..ch8))
        直到外部取消（Ctrl+C）/source_iter 结束
        """
        loop = asyncio.get_running_loop()
        self._last_upd = loop.time()
        try:
            async for _ts, chs in source_iter:
                self._push(chs)

                now = loop.time()
                if now - self._last_upd >= self._upd_interval:
                    self._draw_once()
                    self._last_upd = now
        finally:
            plt.close(self.fig)
