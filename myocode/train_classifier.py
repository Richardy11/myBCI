import argparse, json, os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from dataclasses import dataclass, asdict

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from emg_features import SlidingWindow, TDFeatures, FFTBands, ExpSmoother

# ---------------- Dataclasses ----------------
@dataclass
class FeatureConfig:
    srate: int = 200
    win_ms: int = 200
    step_ms: int = 10
    use_fft: bool = True
    fftlen: int = 64
    smooth_alpha: float = 0.25

@dataclass
class SavedModelMeta:
    label_names: list
    onset_threshold: float
    feature_cfg: dict

# ---------------- Helpers ----------------
def _maybe_compute_feats(raw_trials, cfg: FeatureConfig) -> np.ndarray:
    """raw_trials: List[(N,8)] -> (M,D)"""
    win = SlidingWindow(srate=cfg.srate, win_ms=cfg.win_ms, step_ms=cfg.step_ms)
    td  = TDFeatures()
    fft = FFTBands(srate=cfg.srate, fftlen=cfg.fftlen) if cfg.use_fft else None
    sm  = ExpSmoother(alpha=cfg.smooth_alpha) if cfg.smooth_alpha is not None else None
    feats = []
    for arr in raw_trials:
        win.buf.clear()
        for row in arr.astype(np.float32):
            for w in win.push(row):
                f = td(w)
                if fft is not None:
                    f = np.concatenate([f, fft(w)], axis=0)
                if sm is not None:
                    f = sm(f)
                feats.append(f.astype(np.float32))
    return np.asarray(feats, dtype=np.float32)

def _find_calib_mat(data_dir: Path, subject: str) -> Path:
    cand = data_dir / f"calibrate_subject_{subject}.mat"
    if not cand.exists():
        raise FileNotFoundError(f"找不到标定文件：{cand}")
    return cand

def load_calib_mat(mat_path: Path, actions: list[str], want_feat: bool, cfg: FeatureConfig):
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    X_list, y_list = [], []
    for li, a in enumerate(actions):
        key_feat = f"{a}_feat"
        key_raw  = f"{a}_raw"
        if want_feat and key_feat in mat:
            trials = mat[key_feat]  # object-array
            for t in trials:
                arr = np.asarray(t, dtype=np.float32)
                if arr.ndim == 1:  # 防止单 trial 被 squeeze
                    arr = arr.reshape(1, -1)
                X_list.append(arr)
                y_list.append(np.full((len(arr),), li, dtype=np.int64))
        else:
            if key_raw not in mat:
                print(f"[WARN] {key_raw} not found, skip.")
                continue
            raw_trials = [np.asarray(t, dtype=np.float32) for t in mat[key_raw]]
            feats = _maybe_compute_feats(raw_trials, cfg)
            X_list.append(feats)
            y_list.append(np.full((len(feats),), li, dtype=np.int64))

    if not X_list:
        raise RuntimeError("No training samples constructed.")
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    onset_thr = float(mat.get("onset_threshold", 0.0))
    return X, y, onset_thr

def build_clf(name: str):
    name = name.lower()
    if name == "rlda":
        # Shrinkage LDA：比纯 LDA 更稳（小样本/共线）
        return LDA(solver="lsqr", shrinkage="auto")
    if name == "lda":
        return LDA()
    if name == "svm":
        return LinearSVC()  # 实时友好
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if name == "logreg":
        return LogisticRegression(max_iter=300, n_jobs=1, multi_class="auto")
    raise ValueError(f"未知分类器: {name}")

def train_and_save(X, y, model_out: Path, label_names: list[str], onset_thr: float, feature_cfg: FeatureConfig, clf_name="rlda"):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = build_clf(clf_name)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(Xtr, ytr)
    ypr = clf.predict(Xte)
    acc = accuracy_score(yte, ypr)
    print(f"[TRAIN] hold-out acc = {acc:.3f}")
    try:
        print(classification_report(yte, ypr, target_names=label_names, digits=3))
    except Exception:
        print(classification_report(yte, ypr, digits=3))

    meta = SavedModelMeta(label_names=label_names, onset_threshold=onset_thr, feature_cfg=asdict(feature_cfg))
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "clf": clf, "meta": meta}, model_out)
    print(f"[TRAIN] model saved -> {model_out}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    # 方式A：GUI默认——目录 + subject
    ap.add_argument("--data-dir", type=str, help="calibrate 输出目录")
    ap.add_argument("--subject", type=str, help="subject id，如 1")
    # 方式B：老参数——直接给 .mat
    ap.add_argument("--mat", type=str, help="calibrate_subject_*.mat 路径（优先级高于 data-dir+subject）")

    ap.add_argument("--actions", type=str, default="REST,OPEN,CLOSE")
    ap.add_argument("--clf", choices=["RLDA","LDA","SVM","RF","logreg"], default="RLDA")
    ap.add_argument("--model-out", type=str, default="./models/myo_model.pkl")

    # 特征相关
    ap.add_argument("--use-feat", action="store_true", help="优先使用 .mat 中保存的特征")
    ap.add_argument("--win", type=int, default=200)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--fftlen", type=int, default=64)
    ap.add_argument("--no-fft", action="store_true")

    args = ap.parse_args()

    # 定位 .mat
    if args.mat:
        mat_path = Path(args.mat)
    else:
        if not args.data_dir or not args.subject:
            raise SystemExit("需要 --data-dir 与 --subject（或直接给 --mat）")
        mat_path = _find_calib_mat(Path(args.data_dir), args.subject)

    actions = [a.strip().upper() for a in args.actions.split(",") if a.strip()]
    cfg = FeatureConfig(win_ms=args.win, step_ms=args.step,
                        use_fft=(not args.no_fft),
                        fftlen=args.fftlen)

    print(f"[TRAIN] loading {mat_path}")
    X, y, thr = load_calib_mat(mat_path, actions, args.use_feat, cfg)

    print(f"[TRAIN] samples={len(y)}, dim={X.shape[1]}, classes={len(set(y))}")
    train_and_save(X, y, Path(args.model_out), actions, thr, cfg, clf_name=args.clf)

if __name__ == "__main__":
    main()