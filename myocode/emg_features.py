import asyncio
from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import AsyncIterator, Iterable, Tuple, Optional, List, Sequence

import numpy as np

# ========== 滑窗 ==========
@dataclass
class SlidingWindow:
    srate: int = 200           # Myo 原始采样率（近似 200 Hz）
    win_ms: int = 200          # 窗长(ms)
    step_ms: int = 10          # 步长(ms)

    def __post_init__(self):
        self.win_n = max(1, int(round(self.srate * self.win_ms / 1000.0)))
        self.step_n = max(1, int(round(self.srate * self.step_ms / 1000.0)))
        self.buf = deque()

    def push(self, sample: np.ndarray):
        """逐样本推进；当累积到一窗时产出 (N,8) 的窗口，随后左弹 step_n 个样本。"""
        self.buf.append(sample)
        if len(self.buf) >= self.win_n:
            start = len(self.buf) - self.win_n
            win = np.stack(list(islice(self.buf, start, None)), axis=0)
            yield win
            # 防御性 popleft，避免极端参数时抛错
            for _ in range(self.step_n):
                if self.buf:
                    self.buf.popleft()
                else:
                    break

# ========== TD 特征（含 ZC / SSC）==========
@dataclass
class TDFeatures:
    zc_thresh: float = 0.0     # 零交叉阈值（噪声稍大时可 0.01~0.05 或按标定）
    ssc_thresh: float = 0.0    # 斜率符号变化阈值

    def __call__(self, win: np.ndarray) -> np.ndarray:
        """
        输入:
            win: (N, 8) 原始窗口
        输出:
            feat: (8*5,) 按 [MAV, RMS, WL, ZC, SSC] 拼接
        """
        x = win.astype(np.float32)             # (N, 8)

        # MAV
        mav = np.mean(np.abs(x), axis=0)

        # RMS
        rms = np.sqrt(np.mean(x * x, axis=0) + 1e-12)

        # Waveform Length
        dx = np.diff(x, axis=0)                # (N-1, 8)
        wl = np.sum(np.abs(dx), axis=0)

        # Zero Crossings（阈值 + 相邻符号相反）
        x1 = x[:-1, :]
        x2 = x[1:, :]
        prod = x1 * x2
        if self.zc_thresh > 0.0:
            zc_mask = (prod < 0) & (np.abs(x2 - x1) > self.zc_thresh)
        else:
            zc_mask = (prod < 0)
        zc = np.sum(zc_mask, axis=0).astype(np.float32)

        # Slope Sign Changes（对一阶差分做 ZC）
        dx1 = dx[:-1, :]
        dx2 = dx[1:, :]
        prod_s = dx1 * dx2
        if self.ssc_thresh > 0.0:
            ssc_mask = (prod_s < 0) & (np.abs(dx2 - dx1) > self.ssc_thresh)
        else:
            ssc_mask = (prod_s < 0)
        ssc = np.sum(ssc_mask, axis=0).astype(np.float32)

        feat = np.concatenate([mav, rms, wl, zc, ssc], axis=0)   # (40,)
        return feat

# ========== 指数平滑（EMA）==========
@dataclass
class ExpSmoother:
    alpha: float = 0.25
    _y: Optional[np.ndarray] = None

    def __post_init__(self):
        self.alpha = float(self.alpha)
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("ExpSmoother alpha must be in (0,1).")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if self._y is None:
            # 首帧直通，避免“低一拍”
            self._y = x.copy()
        else:
            self._y = self.alpha * x + (1.0 - self.alpha) * self._y
        return self._y

# ========== 起始阈值（静息标定）==========
@dataclass
class OnsetThreshold:
    scalar: float = 1.3  # 常见范围 1.1~1.5

    def calibrate(self, rest_feats: Iterable[np.ndarray]) -> float:
        """
        用静息段的 MAV 均值来估计阈值：mean + scalar*std
        rest_feats: 迭代器/列表，元素为 (40,) 特征，或至少含前8维 MAV
        """
        mv = []
        for f in rest_feats:
            f = np.asarray(f)
            mv.append(np.mean(f[:8]))  # 只用 MAV
        mv = np.asarray(mv, dtype=np.float32)
        mu, sd = float(np.mean(mv)), float(np.std(mv) + 1e-8)
        return mu + self.scalar * sd

# ========== 频段能量（FFT Bands）==========
@dataclass
class FFTBands:
    srate: int = 200
    fftlen: int = 64
    bands_hz: Sequence[tuple] = ((5,15), (15,30), (30,50), (50,80), (80,100))
    log_power: bool = True
    eps: float = 1e-10

    def __post_init__(self):
        # 计算 rFFT bin 频率
        freqs = np.fft.rfftfreq(self.fftlen, d=1.0 / self.srate)  # (fftlen/2+1,)
        self.band_bins: List[np.ndarray] = []
        for lo, hi in self.bands_hz:
            idx = np.where((freqs >= lo) & (freqs < hi))[0]
            # 防止空段：至少包含一个 bin
            if idx.size == 0:
                idx = np.array([np.argmin(np.abs(freqs - lo))], dtype=int)
            self.band_bins.append(idx)

    def __call__(self, win: np.ndarray) -> np.ndarray:
        """
        win: (N,8) -> return (8 * n_bands,)
        """
        x = win.astype(np.float32)
        N = x.shape[0]

        # 去均值 + Hann
        x = x - np.mean(x, axis=0, keepdims=True)
        w = np.hanning(N).astype(np.float32)[:, None]
        xw = x * w

        # 零填充或截断到 fftlen
        if N < self.fftlen:
            pad = np.zeros((self.fftlen - N, x.shape[1]), dtype=np.float32)
            xw = np.vstack([xw, pad])
        elif N > self.fftlen:
            xw = xw[-self.fftlen:, :]

        X = np.fft.rfft(xw, n=self.fftlen, axis=0)              # (fftlen/2+1, 8)
        P = (np.abs(X) ** 2) / (np.sum(w[:, 0] ** 2) + self.eps)

        feats = []
        for bins in self.band_bins:
            feats.append(np.sum(P[bins, :], axis=0))             # (8,)
        F = np.concatenate(feats, axis=0)                        # (8 * n_bands,)
        if self.log_power:
            F = np.log10(F + self.eps)
        return F

# ========== 异步特征流 ==========
async def feature_stream(
    emg_stream: AsyncIterator[Tuple[int, Tuple[int, ...]]],
    srate: int = 200,
    win_ms: int = 200,
    step_ms: int = 10,
    zc_thresh: float = 0.0,
    ssc_thresh: float = 0.0,
    smooth_alpha: Optional[float] = 0.25,   # None 或 <=0/>=1 → 不平滑
    use_fft: bool = True,
    fftlen: int = 64,
    bands_hz: Sequence[tuple] = ((5,15), (15,30), (30,50), (50,80), (80,100)),
) -> AsyncIterator[Tuple[int, np.ndarray]]:
    """
    逐窗输出: (ts_ns, feat)
    feat = TD(40) [+ FFT(8*n_bands)]
    """
    win = SlidingWindow(srate=srate, win_ms=win_ms, step_ms=step_ms)
    td = TDFeatures(zc_thresh=zc_thresh, ssc_thresh=ssc_thresh)
    fft = FFTBands(srate=srate, fftlen=fftlen, bands_hz=bands_hz) if use_fft else None

    # 仅当 0<alpha<1 时启用 EMA
    if smooth_alpha is not None and (0.0 < float(smooth_alpha) < 1.0):
        smoother = ExpSmoother(alpha=float(smooth_alpha))
    else:
        smoother = None

    async for ts, ch in emg_stream:
        x = np.asarray(ch, dtype=np.float32)  # (8,)
        for w in win.push(x):
            feat = td(w)                                  # (40,)
            if fft is not None:
                f = fft(w)                                 # (8 * n_bands,)
                feat = np.concatenate([feat, f], axis=0)
            if smoother is not None:
                feat = smoother(feat)
            yield (ts, feat)