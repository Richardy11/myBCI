import argparse, asyncio, time, os
import numpy as np
import joblib
from dataclasses import dataclass

from myo_emg_stream import MyoEMGStreamer
from emg_features import feature_stream
from visualization import RealtimeLabelDisplay
import threading
from collections import deque
from typing import Dict, List, Tuple
from online_adaptation import load_old_feats_per_label, train_with_replacement, persist_action_into_mat

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
    # 维度检查标志（每轮从 active_bundle 的 scaler 获取维度）
    expect_dim = getattr(scaler, "n_features_in_", None)
    warned_dim = False

    # Online adaptation buffers and lock-free swap primitives
    active_bundle = {"scaler": scaler, "clf": clf}
    shadow_bundle = {"scaler": None, "clf": None}
    bundle_lock = threading.Lock()  # used only for short swap

    # Ring buffer for raw windows used for quick capture (capacity set after srate known)
    raw_ring = deque(maxlen=1)
    # On-demand recording buffer (click-then-record)
    rec = {"active": False, "buf": [], "sec": 0.0, "start": 0.0}
    rec_lock = threading.Lock()

    # Optional: preload old features from provided mat for replacement policy
    old_feats_per_label: Dict[int, np.ndarray] = load_old_feats_per_label(
        label_names=label_names,
        feat_cfg=feat_cfg,
        old_mat=args.old_mat,
        old_data_dir=args.old_data_dir,
        old_subject=args.old_subject,
        old_use_feat=args.old_use_feat,
    )
    # Also try to load persisted adaptation cache tied to model path
    try:
        feats_cache_path = f"{args.model}.feats.pkl"
        if os.path.isfile(feats_cache_path):
            cache = joblib.load(feats_cache_path)
            if isinstance(cache, dict):
                for k, v in cache.items():
                    try:
                        ki = int(k)
                        arr = np.asarray(v, dtype=np.float32)
                        if arr.ndim == 2 and arr.size > 0:
                            old_feats_per_label[ki] = arr
                    except Exception:
                        pass
            print(f"[ADAPT] loaded feats cache: {feats_cache_path}")
    except Exception as e:
        print("[ADAPT] load feats cache failed:", e)

    # Background training worker
    training_thread = {"th": None, "busy": False}

    # Async persister to avoid blocking the hot-swap path
    def _persist_in_background(model_path: str,
                               scaler_obj,
                               clf_obj,
                               meta_obj,
                               feats_by_label: Dict[int, np.ndarray],
                               action_name: str,
                               mat_path_opt: str | None,
                               raw_used_opt: np.ndarray | None,
                               feats_new_opt: np.ndarray | None):
        def _persist_job():
            try:
                # Save model bundle
                bundle = {"scaler": scaler_obj, "clf": clf_obj, "meta": meta_obj}
                joblib.dump(bundle, model_path)
                # Save feats cache (cap rows per class to 5000)
                try:
                    feats_cache_path = f"{model_path}.feats.pkl"
                    persisted: Dict[int, np.ndarray] = {}
                    for ki, arr in feats_by_label.items():
                        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.size > 0:
                            take = arr[-5000:] if len(arr) > 5000 else arr
                            persisted[int(ki)] = take.astype(np.float32, copy=False)
                    joblib.dump(persisted, feats_cache_path)
                except Exception as e:
                    print("[ADAPT] persist feats cache failed:", e)
                # Write back to calibrate .mat (last trial replacement)
                try:
                    if mat_path_opt and os.path.isfile(mat_path_opt):
                        persist_action_into_mat(mat_path_opt, action_name, raw_used_opt, feats_new_opt)
                except Exception as e:
                    print("[ADAPT] persist mat failed:", e)
                print("[ADAPT] persist done")
            except Exception as e:
                print("[ADAPT] persist failed:", e)
        th = threading.Thread(target=_persist_job, daemon=True)
        th.start()

    def start_online_train(action: str, seconds: float):
        # Kick off a background job that snapshots recent raw data to build a dataset,
        # trains in shadow, then atomically swaps bundles
        if training_thread["busy"]:
            print("[ADAPT] training already running; ignore new request")
            return

        def _job():
            training_thread["busy"] = True
            try:
                # Click-then-record: record seconds of fresh raw into rec.buf
                rec_sec = max(float(seconds), float(feat_cfg.get("win_ms", 200)) / 1000.0)
                rec_sec = min(rec_sec, float(args.adapt_max_sec))
                # start recording
                with rec_lock:
                    rec["active"] = True
                    rec["buf"] = []
                    rec["sec"] = rec_sec
                    rec["start"] = time.time()
                if display is not None:
                    display.show_status(f"Recording… {rec_sec:.1f}s", 1.0)
                # wait until recorded
                while True:
                    with rec_lock:
                        elapsed = time.time() - rec["start"]  # type: ignore[index]
                    left = rec_sec - elapsed
                    if left <= 0:
                        break
                    if display is not None:
                        display.show_status(f"Recording… {max(0.0,left):.1f}s", 0.5)
                    time.sleep(0.1)
                # snapshot and stop recording
                with rec_lock:
                    samples = np.array(rec["buf"], dtype=np.float32)
                    rec["active"] = False
                if samples.ndim == 1:
                    samples = samples.reshape(-1, 8)

                try:
                    scaler_s, clf_s, old_feats_updated, raw_used, feats_new = train_with_replacement(
                        action=action,
                        seconds=rec_sec,
                        ring_samples=samples,
                        old_feats_per_label=old_feats_per_label,
                        label_names=label_names,
                        feat_cfg=feat_cfg,
                        onset_thr=onset_thr,
                        adapt_max_sec=float(args.adapt_max_sec),
                        cap_other_secs=3.0,
                    )
                    # Update reference safely: train_with_replacement mutates in place,
                    # so we do NOT clear; just ensure we keep the same object
                    if old_feats_updated is not old_feats_per_label:
                        old_feats_per_label.update(old_feats_updated)
                    # status: how many new windows used
                    num_new = int(len(samples))
                    if display is not None:
                        display.show_status(f"Adapt OK: {action} replaced ~{num_new} raw samples", duration=2.0)
                except ValueError as e:
                    if str(e) == "no_raw":
                        print("[ADAPT] no raw samples available; skip")
                        if display is not None:
                            display.show_status("Adapt skipped: no raw", 2.0)
                        return
                    if str(e) == "no_window":
                        print("[ADAPT] not enough data for a single window; skip")
                        if display is not None:
                            display.show_status("Adapt skipped: not enough for 1 window", 2.0)
                        return
                    if str(e) == "insufficient_classes":
                        print("[ADAPT] need at least 2 classes (e.g., include REST/negatives). Skip training.")
                        if display is not None:
                            display.show_status("Adapt skipped: need ≥2 classes", 2.0)
                        return
                    raise

                # Swap bundles atomically
                with bundle_lock:
                    shadow_bundle["scaler"] = scaler_s
                    shadow_bundle["clf"] = clf_s
                    # flip active
                    active_bundle["scaler"] = shadow_bundle["scaler"]
                    active_bundle["clf"] = shadow_bundle["clf"]
                print("[ADAPT] model swapped")
                # Persist updated artifacts in background to avoid blocking
                mat_auto = None
                if args.old_mat:
                    mat_auto = args.old_mat
                elif args.old_data_dir and args.old_subject:
                    mat_auto = os.path.join(args.old_data_dir, f"calibrate_subject_{args.old_subject}.mat")
                _persist_in_background(
                    model_path=args.model,
                    scaler_obj=active_bundle["scaler"],
                    clf_obj=active_bundle["clf"],
                    meta_obj=meta,
                    feats_by_label=old_feats_per_label,
                    action_name=action,
                    mat_path_opt=mat_auto,
                    raw_used_opt=raw_used,
                    feats_new_opt=feats_new,
                )
            except Exception as e:
                print("[ADAPT] training failed:", e)
            finally:
                training_thread["busy"] = False

        th = threading.Thread(target=_job, daemon=True)
        training_thread["th"] = th
        th.start()

    actions_for_ui = [n for n in label_names if isinstance(n, str) and n and n != "REST"] or ["OPEN", "CLOSE"]
    display = RealtimeLabelDisplay("Classifier Output", actions=actions_for_ui,
                                   on_start_adapt=start_online_train) if args.show_gui else None

    async with MyoEMGStreamer(addr=args.addr, mode=args.mode) as s:
        print("[RUN] Connected. Streaming + predicting…  (Ctrl+C 停止)")

        async def _tapped_stream(base_stream):
            async for ts, ch in base_stream:
                # append raw 8-ch sample for adaptation ring
                try:
                    raw_ring.append(np.asarray(ch, dtype=np.float32))
                except Exception:
                    pass
                # if recording is active, also append into on-demand buffer
                try:
                    with rec_lock:
                        if rec["active"]:
                            rec["buf"].append(np.asarray(ch, dtype=np.float32))
                except Exception:
                    pass
                yield ts, ch

        # After srate is known, set ring capacity to adapt-max-sec
        try:
            raw_ring.maxlen = int(feat_cfg.get("srate", 200) * args.adapt_max_sec)
        except Exception:
            pass

        async for ts, feat in feature_stream(
            _tapped_stream(s.stream()),
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
                        # 将重载的模型切为 active
                        with bundle_lock:
                            active_bundle["scaler"] = scaler
                            active_bundle["clf"] = clf
                        expect_dim  = getattr(scaler, "n_features_in_", None)
                        warned_dim  = False
                        load_model._last_mtime = m
                        print("[RUN] model reloaded.")
                except Exception as e:
                    print("[RUN] reload failed:", e)

            # 维度健壮性（只在第一次不匹配时警告一次）
            # 每帧从当前 active 模型获取期望维度
            with bundle_lock:
                scaler_now = active_bundle["scaler"]
                clf_now = active_bundle["clf"]
            expect_dim = getattr(scaler_now, "n_features_in_", None)
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
                    xs = scaler_now.transform(feat[None, :])
                    pred = int(clf_now.predict(xs)[0])
                except Exception as e:
                    if not warned_dim:
                        print("[RUN][WARN] predict failed:", e)
                        warned_dim = True
                    continue

            pred_s = smoother.push(pred) if args.smooth_k > 1 else pred

            # 可视化 + 低频打印
            label_str = _safe_label_name(label_names, pred_s)
            if display is not None:
                display.update(label_str)
            if (now - last_log) >= 0.5:
                print(f"[RUN] { label_str }  (raw={_safe_label_name(label_names, pred)}, MAV={mav_mean:.2f})")
                last_log = now

    if display is not None:
        display.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--addr", type=str, required=True)
    ap.add_argument("--mode", choices=["filtered","raw"], default="filtered")
    ap.add_argument("--smooth-k", type=int, default=5, help="多数票窗口")
    ap.add_argument("--reload-sec", type=float, default=0.0, help=">0 定期检测并热重载模型（秒）")
    ap.add_argument("--show-gui", action="store_true", help="弹出实时分类结果窗口")
    # Optional: provide old dataset to achieve true 80/20 mixing
    ap.add_argument("--old-mat", type=str, default="", help="calibrate_subject_*.mat 路径（若提供则用于旧样本）")
    ap.add_argument("--old-data-dir", type=str, default="", help="calibrate 输出目录（与 --old-subject 配合）")
    ap.add_argument("--old-subject", type=str, default="", help="subject id（与 --old-data-dir 配合）")
    ap.add_argument("--old-use-feat", action="store_true", help="优先使用 .mat 内保存的特征")
    ap.add_argument("--adapt-max-sec", type=float, default=5.0, help="环形缓冲最大秒数（录制秒数将被限制不超过该值）")
    args = ap.parse_args()
    asyncio.run(amain(args))

if __name__ == "__main__":
    main()