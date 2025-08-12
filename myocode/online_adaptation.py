from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import os
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from emg_features import SlidingWindow, TDFeatures, FFTBands, ExpSmoother


def _build_feat_pipeline_from_cfg(feat_cfg: dict):
    srate = int(feat_cfg.get("srate", 200))
    win_ms = int(feat_cfg.get("win_ms", 200))
    step_ms = int(feat_cfg.get("step_ms", 10))
    use_fft = bool(feat_cfg.get("use_fft", True))
    fftlen = int(feat_cfg.get("fftlen", 64))
    smooth_alpha = float(feat_cfg.get("smooth_alpha", 0.0))

    win = SlidingWindow(srate=srate, win_ms=win_ms, step_ms=step_ms)
    td = TDFeatures()
    fft = FFTBands(srate=srate, fftlen=fftlen) if use_fft else None
    smoother = ExpSmoother(alpha=smooth_alpha) if (0.0 < smooth_alpha < 1.0) else None
    return srate, win, td, fft, smoother


def load_old_feats_per_label(
    label_names: Sequence[str],
    feat_cfg: dict,
    old_mat: str = "",
    old_data_dir: str = "",
    old_subject: str = "",
    old_use_feat: bool = False,
) -> Dict[int, np.ndarray]:
    """Load per-class old features from a calibrate .mat.

    Priority: old_mat > (old_data_dir + old_subject).
    If *_feat exists and old_use_feat=True, use it; otherwise compute from *_raw using current feature_cfg.
    Returns a dict: {label_index: np.ndarray[num_windows, feat_dim]}.
    """
    mat_path = None
    if old_mat:
        mat_path = old_mat
    elif old_data_dir and old_subject:
        mat_path = os.path.join(old_data_dir, f"calibrate_subject_{old_subject}.mat")
    if not mat_path or not os.path.isfile(mat_path):
        return {}

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    srate, win, td, fft, smoother = _build_feat_pipeline_from_cfg(feat_cfg)
    result: Dict[int, np.ndarray] = {}

    for li, a in enumerate(label_names):
        if not isinstance(a, str):
            continue
        key_feat = f"{a}_feat"
        key_raw = f"{a}_raw"
        feats_list: List[np.ndarray] = []

        if key_feat in mat and old_use_feat:
            trials = mat[key_feat]
            for t in trials:
                arr = np.asarray(t, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                feats_list.append(arr)
        elif key_raw in mat:
            for t in mat[key_raw]:
                arr = np.asarray(t, dtype=np.float32)
                win.buf.clear()
                rows: List[np.ndarray] = []
                for row in arr:
                    for w in win.push(row):
                        f = td(w)
                        if fft is not None:
                            f = np.concatenate([f, fft(w)], axis=0)
                        if smoother is not None:
                            f = smoother(f)
                        rows.append(f.astype(np.float32))
                if rows:
                    feats_list.append(np.asarray(rows, dtype=np.float32))

        if feats_list:
            result[li] = np.vstack(feats_list)

    return result


def train_with_replacement(
    action: str,
    seconds: float,
    ring_samples: np.ndarray,  # shape (N, 8)
    old_feats_per_label: Dict[int, np.ndarray],
    label_names: Sequence[str],
    feat_cfg: dict,
    onset_thr: float | None = None,
    adapt_max_sec: float = 5.0,
    cap_other_secs: float = 3.0,
):
    """Build dataset and train new scaler+clf by replacing selected class with new windows.

    - seconds is clamped to adapt_max_sec
    - Features computed to match feature_cfg (TD + optional FFT + optional smoothing)
    - Other classes are capped to wins_per_sec*cap_other_secs windows to bound runtime
    Returns: (scaler, clf, updated_old_feats_per_label)
    May raise ValueError("insufficient_classes") if <2 classes present.
    """
    rec_sec = min(float(seconds), float(adapt_max_sec))
    # Ensure we request at least one full window worth of data
    try:
        min_window_sec = float(feat_cfg.get("win_ms", 200)) / 1000.0
        if rec_sec < min_window_sec:
            rec_sec = min_window_sec
    except Exception:
        pass
    srate, win, td, fft, smoother = _build_feat_pipeline_from_cfg(feat_cfg)

    step_n = max(1, int(round(srate * win.step_ms / 1000.0))) if hasattr(win, 'step_ms') else max(1, int(round(srate * feat_cfg.get('step_ms', 10) / 1000.0)))
    wins_per_sec = max(1, int(srate // step_n))
    w_new_target = max(1, int(round(wins_per_sec * rec_sec)))

    # Slice ring
    need = max(1, int(rec_sec * srate))
    raw_tail = ring_samples[-need:] if ring_samples.size else np.zeros((0, 8), np.float32)
    # If still not enough to make a single window but ring has enough, take last win_n
    min_needed = getattr(win, 'win_n', max(1, int(round(srate * (feat_cfg.get('win_ms', 200)/1000.0)))))
    if raw_tail.shape[0] < min_needed and ring_samples.shape[0] >= min_needed:
        raw_tail = ring_samples[-min_needed:]
    if raw_tail.shape[0] == 0:
        raise ValueError("no_raw")

    # Compute new windows
    X_new: List[np.ndarray] = []
    for row in raw_tail:
        for w in win.push(row):
            f = td(w)
            if fft is not None:
                f = np.concatenate([f, fft(w)], axis=0)
            if smoother is not None:
                f = smoother(f)
            X_new.append(f.astype(np.float32))
    if not X_new:
        raise ValueError("no_window")
    X_new = np.asarray(X_new, dtype=np.float32)
    if len(X_new) > w_new_target:
        X_new = X_new[-w_new_target:]

    # Optional: split some windows as REST negatives based on onset threshold
    li_rest = label_names.index("REST") if ("REST" in label_names) else None
    if li_rest is not None and onset_thr is not None and onset_thr > 0.0 and X_new.shape[0] > 0 and X_new.shape[1] >= 8:
        mav = np.mean(X_new[:, :8], axis=1)
        rest_mask = mav < float(onset_thr)
        if np.any(rest_mask):
            X_rest = X_new[rest_mask]
            X_new = X_new[~rest_mask]
        else:
            X_rest = np.zeros((0, X_new.shape[1]), dtype=X_new.dtype)
    else:
        X_rest = np.zeros((0, X_new.shape[1] if X_new.ndim == 2 else 0), dtype=X_new.dtype)

    li_sel = label_names.index(action) if action in label_names else None
    X_parts = []
    y_parts = []
    if X_new.size:
        X_parts.append(X_new)
        y_parts.append(np.full((len(X_new),), fill_value=(li_sel or 0), dtype=np.int64))
    if X_rest.size and li_rest is not None:
        X_parts.append(X_rest)
        y_parts.append(np.full((len(X_rest),), fill_value=li_rest, dtype=np.int64))

    # Replace tail of selected class cache with new windows
    if li_sel is not None and li_sel in old_feats_per_label and len(X_new) > 0:
        Xo_sel = old_feats_per_label.get(li_sel, np.zeros((0, X_new.shape[1]), dtype=X_new.dtype))
        drop = min(len(X_new), len(Xo_sel))
        Xo_keep = Xo_sel[:-drop] if drop < len(Xo_sel) else np.zeros((0, Xo_sel.shape[1]), dtype=Xo_sel.dtype)
        # mutate in place but keep the dict object identity
        old_feats_per_label[li_sel] = np.vstack([Xo_keep, X_new]) if len(X_new) > 0 else Xo_keep
        if Xo_keep.size:
            X_parts.append(Xo_keep)
            y_parts.append(np.full((len(Xo_keep),), fill_value=li_sel, dtype=np.int64))

    # Other classes with cap
    cap_other = int(wins_per_sec * cap_other_secs)
    for lj, Xo in old_feats_per_label.items():
        if li_sel is not None and lj == li_sel:
            continue
        if len(Xo) == 0:
            continue
        Xo_take = Xo[-cap_other:] if len(Xo) > cap_other else Xo
        X_parts.append(Xo_take)
        y_parts.append(np.full((len(Xo_take),), fill_value=lj, dtype=np.int64))

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    if len(np.unique(y_train)) < 2:
        raise ValueError("insufficient_classes")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xs, y_train)

    return scaler, clf, old_feats_per_label, raw_tail, X_new


def persist_action_into_mat(mat_path: str, action: str,
                            new_raw: np.ndarray | None,
                            new_feat: np.ndarray | None) -> None:
    """Replace the last trial of a specific action in calibrate .mat with new data.
    If no existing trials, append as the first trial.
    """
    if not mat_path or not os.path.isfile(mat_path):
        return
    try:
        mdict = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        changed = False
        # RAW
        if new_raw is not None and new_raw.size > 0:
            key_raw = f"{action}_raw"
            if key_raw in mdict:
                arr = mdict[key_raw]
                try:
                    trials = list(arr)
                except Exception:
                    trials = [arr]
                if len(trials) > 0:
                    trials[-1] = new_raw.astype(np.int16, copy=False)
                else:
                    trials = [new_raw.astype(np.int16, copy=False)]
                mdict[key_raw] = np.array(trials, dtype=object)
            else:
                mdict[key_raw] = np.array([new_raw.astype(np.int16, copy=False)], dtype=object)
            changed = True
        # FEAT
        if new_feat is not None and new_feat.size > 0:
            key_feat = f"{action}_feat"
            if key_feat in mdict:
                arrf = mdict[key_feat]
                try:
                    trialsf = list(arrf)
                except Exception:
                    trialsf = [arrf]
                if len(trialsf) > 0:
                    trialsf[-1] = new_feat.astype(np.float32, copy=False)
                else:
                    trialsf = [new_feat.astype(np.float32, copy=False)]
                mdict[key_feat] = np.array(trialsf, dtype=object)
            else:
                mdict[key_feat] = np.array([new_feat.astype(np.float32, copy=False)], dtype=object)
            changed = True
        if changed:
            # Remove MATLAB header/meta keys to avoid underscore warnings
            try:
                for k in list(mdict.keys()):
                    if isinstance(k, str) and k.startswith("__"):
                        mdict.pop(k, None)
            except Exception:
                pass
            savemat(mat_path, mdict)
    except Exception:
        pass


