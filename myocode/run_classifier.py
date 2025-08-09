import argparse, asyncio, time, os
import numpy as np
import joblib
from dataclasses import dataclass

from myo_emg_stream import MyoEMGStreamer
from emg_features import feature_stream

@dataclass
class SavedModelMeta:
    label_names: list
    onset_threshold: float
    feature_cfg: dict

class MajoritySmoother:
    """简单多数票平滑（长度 k），用于稳定输出"""
    def __init__(self, k=5):
        from collections import deque
        self.buf = deque(maxlen=k)
    def push(self, label: int) -> int:
        self.buf.append(label)
        vals, cnts = np.unique(self.buf, return_counts=True)
        return int(vals[np.argmax(cnts)])

def _safe_label_name(label_names, idx: int) -> str:
    try:
        return str(label_names[idx])
    except Exception:
        return str(idx)

def load_model(path: str):
    """兼容 dataclass / dict 的 meta；统一转成 dict 返回。"""
    bundle = joblib.load(path)
    scaler = bundle["scaler"]
    clf     = bundle["clf"]
    meta    = bundle.get("meta", {})
    if hasattr(meta, "__dict__"):  # dataclass 场景
        meta = dict(meta.__dict__)
    return scaler, clf, meta

async def amain(args):
    scaler, clf, meta = load_model(args.model)
    label_names = list(meta.get("label_names", []))
    onset_thr   = float(meta.get("onset_threshold", 0.0))
    feat_cfg    = dict(meta.get("feature_cfg", {}))

    print(f"[RUN] model loaded. labels={label_names or '[0..K-1]'}, onset_thr={onset_thr:.3f}")
    print(f"[RUN] feature_cfg={feat_cfg}")

    smoother = MajoritySmoother(k=args.smooth_k)

    # 分离计时器：一个管日志打印节流，一个管热重载
    last_log = time.time()
    last_reload = time.time()
    expect_dim = getattr(scaler, "n_features_in_", None)
    warned_dim = False

    async with MyoEMGStreamer(addr=args.addr, mode=args.mode) as s:
        print("[RUN] Connected. Streaming + predicting…  (Ctrl+C 停止)")

        async for ts, feat in feature_stream(
            s.stream(),
            srate=feat_cfg.get("srate", 200),
            win_ms=feat_cfg.get("win_ms", 200),
            step_ms=feat_cfg.get("step_ms", 10),
            use_fft=feat_cfg.get("use_fft", True),
            fftlen=feat_cfg.get("fftlen", 64),
            smooth_alpha=feat_cfg.get("smooth_alpha", 0.25),
        ):
            now = time.time()

            # 可选热重载（与日志节流互不干扰）
            if args.reload_sec > 0 and (now - last_reload) >= args.reload_sec:
                last_reload = now
                try:
                    m = os.path.getmtime(args.model)
                    # 用 mtime 比较是否更新
                    if not hasattr(load_model, "_last_mtime"):
                        load_model._last_mtime = m
                    if m > load_model._last_mtime:
                        scaler, clf, meta = load_model(args.model)
                        label_names = list(meta.get("label_names", []))
                        onset_thr   = float(meta.get("onset_threshold", 0.0))
                        feat_cfg    = dict(meta.get("feature_cfg", {}))
                        expect_dim  = getattr(scaler, "n_features_in_", None)
                        warned_dim  = False
                        load_model._last_mtime = m
                        print("[RUN] model reloaded.")
                except Exception as e:
                    print("[RUN] reload failed:", e)

            # 维度健壮性（只在第一次不匹配时警告一次）
            if expect_dim is not None and feat.shape[0] != expect_dim:
                if not warned_dim:
                    print(f"[RUN][WARN] feature dim mismatch: current {feat.shape[0]} vs model {expect_dim}. Skip predicting.")
                    warned_dim = True
                # 仍然做 onset 打印，但不做分类
                mav_mean = float(np.mean(feat[:8])) if feat.shape[0] >= 8 else 0.0
                if (now - last_log) >= 0.5:
                    print(f"[RUN] (dim-mismatch) MAV={mav_mean:.2f}")
                    last_log = now
                continue

            # onset 门控（feat 前8维来自 TD.MAV）
            mav_mean = float(np.mean(feat[:8])) if feat.shape[0] >= 8 else 0.0
            if mav_mean < onset_thr:
                pred = 0  # REST
            else:
                try:
                    xs = scaler.transform(feat[None, :])
                    pred = int(clf.predict(xs)[0])
                except Exception as e:
                    if not warned_dim:
                        print("[RUN][WARN] predict failed:", e)
                        warned_dim = True
                    continue

            pred_s = smoother.push(pred) if args.smooth_k > 1 else pred

            # 低频打印
            if (now - last_log) >= 0.5:
                print(f"[RUN] { _safe_label_name(label_names, pred_s) }  (raw={_safe_label_name(label_names, pred)}, MAV={mav_mean:.2f})")
                last_log = now

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--addr", type=str, required=True)
    ap.add_argument("--mode", choices=["filtered","raw"], default="filtered")
    ap.add_argument("--smooth-k", type=int, default=5, help="多数票窗口")
    ap.add_argument("--reload-sec", type=float, default=0.0, help=">0 定期检测并热重载模型（秒）")
    args = ap.parse_args()
    asyncio.run(amain(args))

if __name__ == "__main__":
    main()