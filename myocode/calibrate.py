# myocode/calibrate.py
import argparse, asyncio, time
from pathlib import Path
import numpy as np
from scipy.io import savemat

from myo_emg_stream import MyoEMGStreamer
from emg_features import SlidingWindow, TDFeatures, FFTBands, ExpSmoother  # ← 新增：可选特征

EMG_SRATE = 200  # Myo 近似 200 Hz

async def countdown_print(sec: float, msg: str):
    """轻量倒计时文本提示"""
    t_end = time.time() + sec
    n = int(sec)
    print(f"[CAL] {msg} ({sec:.1f}s)")
    while time.time() < t_end:
        left = t_end - time.time()
        print(f"\r   ... {left:4.1f}s", end="", flush=True)
        await asyncio.sleep(0.2)
    print("\r", end="")

async def collect_one_action(stream, action: str, hold_s: float, rest_s: float,
                             srate=EMG_SRATE,
                             feat_cfg=None):
    """
    采集一个动作：hold_s 秒数据 -> rest_s 秒休息
    返回 raw_emg: (N,8) int16
         feat_mat: (M,D) float32 或 None
    feat_cfg: None 或 dict{ 'win_ms','step_ms','use_fft','fftlen','smooth_alpha' }
    """
    # --- 采集阶段 ---
    buf = []
    feat_rows = []  # 可选特征
    if feat_cfg is not None:
        win = SlidingWindow(srate=srate, win_ms=feat_cfg['win_ms'], step_ms=feat_cfg['step_ms'])
        td  = TDFeatures()
        fft = FFTBands(srate=srate, fftlen=feat_cfg['fftlen']) if feat_cfg['use_fft'] else None
        sm  = ExpSmoother(alpha=feat_cfg['smooth_alpha']) if feat_cfg['smooth_alpha'] is not None else None

    print(f"[CAL] >>> {action}: HOLD {hold_s:.1f}s")
    t_end = time.time() + hold_s
    async for ts, ch in stream:
        buf.append(ch)  # ch: tuple[8] of int
        if feat_cfg is not None:
            x = np.asarray(ch, dtype=np.float32)
            for w in win.push(x):
                f = td(w)
                if fft is not None:
                    f = np.concatenate([f, fft(w)], axis=0)
                if sm is not None:
                    f = sm(f)
                feat_rows.append(f.astype(np.float32))
        if time.time() >= t_end:
            break

    raw = np.array(buf, dtype=np.int16)  # (N,8)
    feat_mat = np.vstack(feat_rows).astype(np.float32) if feat_rows else None
    print(f"[CAL] {action}: got {raw.shape[0]} samples" + ("" if feat_mat is None else f", {feat_mat.shape[0]} feats"))

    # --- 休息阶段 ---
    if rest_s > 0:
        await countdown_print(rest_s, f"REST")

    return raw, feat_mat

def compute_rest_threshold(rest_trials: list[np.ndarray], scalar: float = 1.3) -> float:
    """
    用所有 REST trial 的 MAV 均值估阈：mean + scalar*std
    rest_trials: List[(N,8) int]  # 原始 EMG
    """
    if not rest_trials:
        return 0.0
    mavs = []
    for arr in rest_trials:
        x = np.asarray(arr, dtype=np.float32)
        mavs.append(np.mean(np.abs(x)))
    mv = np.asarray(mavs, dtype=np.float32)
    mu, sd = float(np.mean(mv)), float(np.std(mv) + 1e-8)
    return mu + scalar * sd

async def amain(args):
    actions = [a.strip().upper() for a in args.actions.split(",") if a.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"calibrate_subject_{args.subject}.mat"

    print(f"[CAL] subj={args.subject} actions={actions} trials={args.trials} hold={args.hold}s rest={args.rest}s")
    all_raw  = {a: [] for a in actions}
    all_feat = {a: [] for a in actions} if args.save_feat else None

    feat_cfg = None
    if args.save_feat:
        feat_cfg = dict(win_ms=args.win, step_ms=args.step, use_fft=(not args.no_fft),
                        fftlen=args.fftlen, smooth_alpha=0.25)

    async with MyoEMGStreamer(addr=args.addr, mode=args.mode) as s:
        print("[CAL] Connected. Start sequence… (终端可观察倒计时与进度)")
        for trial in range(1, args.trials+1):
            print(f"[CAL] ===== Trial {trial}/{args.trials} =====")
            for a in actions:
                # 开始前给 2s 准备
                await countdown_print(2.0, f"准备 {a}")
                raw, feat = await collect_one_action(s.stream(), a, hold_s=args.hold, rest_s=args.rest,
                                                     srate=EMG_SRATE, feat_cfg=feat_cfg)
                all_raw[a].append(raw)
                if all_feat is not None:
                    all_feat[a].append(feat)

    # 计算 REST 阈值
    rest_thr = compute_rest_threshold(all_raw.get("REST", []), scalar=args.onset_scalar)
    print(f"[CAL] REST onset threshold ≈ {rest_thr:.3f}")

    # 组织 .mat
    mdict = {
        "subject": args.subject,
        "actions": actions,
        "trials": args.trials,
        "srate": EMG_SRATE,
        "mode": args.mode,
        "addr": args.addr,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "onset_threshold": rest_thr,
        "onset_scalar": args.onset_scalar,
    }
    # 每个动作：raw_trials, (可选) feat_trials
    for a in actions:
        mdict[f"{a}_raw"]  = np.array(all_raw[a],  dtype=object)
        if all_feat is not None:
            mdict[f"{a}_feat"] = np.array(all_feat[a], dtype=object)

    savemat(str(outfile), mdict)
    print(f"[CAL] Saved -> {outfile}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--addr", type=str, required=True)
    p.add_argument("--mode", choices=["filtered","raw"], default="filtered")
    p.add_argument("--subject", type=str, default="1")
    p.add_argument("--actions", type=str, default="REST,OPEN,CLOSE")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--hold", type=float, default=3.0, help="每个动作保持秒数")
    p.add_argument("--rest", type=float, default=2.0, help="动作间歇秒数")
    p.add_argument("--outdir", type=str, default="./data")
    p.add_argument("--save-feat", action="store_true", help="同时保存 TD(+FFT) 特征")
    p.add_argument("--win", type=int, default=200, help="特征窗长 ms")
    p.add_argument("--step", type=int, default=10, help="滑动步长 ms")
    p.add_argument("--fftlen", type=int, default=64, help="rFFT 长度")
    p.add_argument("--no-fft", action="store_true", help="禁用频段能量特征")
    p.add_argument("--onset-scalar", type=float, default=1.3)
    args = p.parse_args()
    asyncio.run(amain(args))