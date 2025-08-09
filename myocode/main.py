# main_myo_run.py
import argparse, asyncio
from myo_emg_stream import MyoEMGStreamer, MYO_ADDR_DEFAULT   # ← 修正文件名
from realtime_emg_plot import EMGPlotter
from emg_features import feature_stream                          # ← 用于特征流

async def run_raw(addr: str, mode: str):
    print(f"Connecting to Myo: addr={addr}, mode={mode}")
    async with MyoEMGStreamer(addr=addr, mode=mode) as s:
        print("Connected ✅  streaming raw EMG…")
        plotter = EMGPlotter(srate=200, span_s=5.0)
        await plotter.run(s.stream())   # 实时显示 8 通道

async def run_feat(addr: str, mode: str,
                   srate=200, win_ms=200, step_ms=10,
                   use_fft=True, fftlen=64):
    print(f"Connecting to Myo: addr={addr}, mode={mode}")
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
            if now - t0 >= 1.0:  # 每秒打印一次
                print(f"[{count} feats] shape={feat.shape}, first8={feat[:8]}")
                t0 = now

async def amain(args):
    try:
        if args.view == "raw":
            await run_raw(addr=args.addr, mode=args.mode)
        else:
            await run_feat(addr=args.addr, mode=args.mode,
                           srate=200, win_ms=args.win, step_ms=args.step,
                           use_fft=(not args.no_fft), fftlen=args.fftlen)
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--addr", type=str, default=MYO_ADDR_DEFAULT, help="macOS 的 Myo UUID")
    p.add_argument("--mode", choices=["filtered", "raw"], default="filtered", help="初始数据模式（无数据会自动切换）")
    p.add_argument("--view", choices=["raw", "feat"], default="raw", help="raw=画8通道；feat=计算特征")
    # 特征参数（feat 模式有效）
    p.add_argument("--win", type=int, default=200, help="特征窗长 ms（默认200）")
    p.add_argument("--step", type=int, default=10, help="滑动步长 ms（默认10）")
    p.add_argument("--fftlen", type=int, default=64, help="rFFT 长度（默认64）")
    p.add_argument("--no-fft", action="store_true", help="禁用频段能量特征")
    args = p.parse_args()
    asyncio.run(amain(args))