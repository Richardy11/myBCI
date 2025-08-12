# myocode/main_myo_run.py
import argparse, asyncio
from typing import Optional
import numpy as np

from myo_emg_stream import MyoEMGStreamer, MYO_ADDR_DEFAULT
from realtime_emg_plot import EMGPlotter
from emg_features import feature_stream   # 用于 feat 视图


# --------- 工具：解析 8 通道 scaler ---------
def parse_scales(s: Optional[str]) -> np.ndarray:
    """
    支持：
    - 空/None -> 全 1
    - 单个数字 -> 8 通道统一这个数字
    - 逗号分隔 8 个数字 -> 分通道缩放
    """
    if not s:
        return np.ones(8, dtype=np.float32)
    try:
        parts = [float(x) for x in s.replace(" ", "").split(",") if x != ""]
        if len(parts) == 1:
            return np.full(8, parts[0], dtype=np.float32)
        if len(parts) >= 8:
            return np.array(parts[:8], dtype=np.float32)
        # 不足 8 个：用最后一个值补齐
        return np.array(parts + [parts[-1]] * (8 - len(parts)), dtype=np.float32)
    except Exception:
        return np.ones(8, dtype=np.float32)


# --------- Level 视图：整流 + EMA 平滑 + 通道缩放（画图） ----------
class EMA:
    def __init__(self, alpha: float = 0.3):
        self.alpha = float(alpha)
        self.y: Optional[np.ndarray] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.y is None:
            self.y = x.astype(np.float32)
            return self.y
        self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y


async def _level_stream(raw_stream, alpha: float, scales_str: Optional[str]):
    """把原始流变换成：abs -> scale -> EMA 的 8 通道流，供 EMGPlotter 画图。"""
    ema = EMA(alpha=alpha)
    scales = parse_scales(scales_str)  # (8,)
    async for ts, ch in raw_stream:
        x = np.abs(np.asarray(ch, dtype=np.float32))   # 整流
        x *= scales                                    # 通道缩放
        y = ema(x)                                     # EMA 平滑
        # EMGPlotter 期望 (ts, 8-tuple)
        yield ts, tuple(float(v) for v in y)


async def run_level(addr: str, mode: str, alpha: float, scales_str: Optional[str], _print_hz_ignored: float):
    print(f"[Level] Connecting to Myo: addr={addr}, mode={mode}, alpha={alpha}")
    async with MyoEMGStreamer(addr=addr, mode=mode) as s:
        print("Connected ✅  streaming LEVEL (rectified+scaled+EMA) …")
        plotter = EMGPlotter(srate=200, span_s=5.0)
        # 把转换后的流交给同一个画图器
        await plotter.run(_level_stream(s.stream(), alpha=alpha, scales_str=scales_str))


# --------- 传统视图：raw ----------
async def run_raw(addr: str, mode: str):
    print(f"[Raw] Connecting to Myo: addr={addr}, mode={mode}")
    async with MyoEMGStreamer(addr=addr, mode=mode) as s:
        print("Connected ✅  streaming raw EMG…")
        plotter = EMGPlotter(srate=200, span_s=5.0)
        await plotter.run(s.stream())


# --------- 传统视图：feat ----------
async def run_feat(addr: str, mode: str, srate=200, win_ms=200, step_ms=10, use_fft=True, fftlen=64):
    print(f"[Feat] Connecting to Myo: addr={addr}, mode={mode}")
    async with MyoEMGStreamer(addr=addr, mode=mode) as s:
        print("Connected ✅  computing features…（Ctrl+C 结束）")
        count = 0
        t0 = asyncio.get_event_loop().time()
        async for ts, feat in feature_stream(
            s.stream(),
            srate=srate, win_ms=win_ms, step_ms=step_ms,
            use_fft=use_fft, fftlen=fftlen
        ):
            count += 1
            now = asyncio.get_event_loop().time()
            if now - t0 >= 1.0:
                print(f"[{count} feats/s] shape={feat.shape}, first8={feat[:8]}")
                t0 = now


async def amain(args):
    try:
        if args.view == "raw":
            await run_raw(addr=args.addr, mode=args.mode)
        elif args.view == "feat":
            await run_feat(addr=args.addr, mode=args.mode,
                           srate=200, win_ms=args.win, step_ms=args.step,
                           use_fft=(not args.no_fft), fftlen=args.fftlen)
        elif args.view == "level":
            await run_level(addr=args.addr, mode=args.mode,
                            alpha=args.smooth_alpha, scales_str=args.scales, _print_hz_ignored=args.print_hz)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--addr", type=str, default=MYO_ADDR_DEFAULT, help="macOS 的 Myo UUID")
    p.add_argument("--mode", choices=["filtered", "raw"], default="filtered", help="数据模式")
    p.add_argument("--view", choices=["raw", "feat", "level"], default="raw", help="raw/feat/level 三种视图")
    # feat 参数
    p.add_argument("--win", type=int, default=200, help="特征窗长 ms")
    p.add_argument("--step", type=int, default=10, help="特征步长 ms")
    p.add_argument("--fftlen", type=int, default=64, help="rFFT 长度")
    p.add_argument("--no-fft", action="store_true", help="禁用频段能量特征")
    # level 参数（不再打印，只画图；为兼容保留 print-hz）
    p.add_argument("--smooth-alpha", dest="smooth_alpha", type=float, default=0.30, help="Level 的 EMA 平滑 α（0~1）")
    p.add_argument("--scales", type=str, default="1,1,1,1,1,1,1,1", help="8 通道缩放（单值或8个逗号分隔值）")
    p.add_argument("--print-hz", dest="print_hz", type=float, default=20.0, help="兼容参数（已忽略）")
    args = p.parse_args()
    asyncio.run(amain(args))