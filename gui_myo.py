# gui_myo.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess, sys, os, json
from pathlib import Path

# ------------------ 配置文件 ------------------
CONFIG_FILE = Path.home() / ".myo_emg_gui" / "config.json"
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

# 允许直接运行 gui_myo.py 时找到 myocode 包
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MYOCODE_DIR = os.path.join(THIS_DIR, "myocode")
PARENT_DIR = os.path.dirname(THIS_DIR)
for p in (PARENT_DIR, MYOCODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from myocode.myo_emg_stream import MYO_ADDR_DEFAULT  # 仅用于默认地址
except Exception:
    MYO_ADDR_DEFAULT = "709CFFAF-2E15-0BF7-5E0D-07720E8111E1"  # 兜底

def _calib_mat_path(outdir: str, subject: str) -> str:
    return os.path.join(outdir, f"calibrate_subject_{subject}.mat")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Myo EMG (Always-On View / Calibrate / Train+Predict)")
        self.geometry("780x560")

        # 子进程
        self.proc_run = None       # 始终用于 main.py 可视化
        self.proc_cal = None
        self.proc_train = None
        self.proc_pred = None

        # Run 恢复标记（Calibrate / Predict 占用设备期间暂停，结束后恢复）
        self._resume_run_after = False

        # 读取配置
        self.cfg = self._load_config()

        # 选项卡
        nb = ttk.Notebook(self)
        self.tab_run   = ttk.Frame(nb)
        self.tab_cal   = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)  # 合并 Train+Predict
        nb.add(self.tab_run,   text="View / Run")
        nb.add(self.tab_cal,   text="Calibrate")
        nb.add(self.tab_train, text="Train & Predict")
        nb.pack(fill="both", expand=True)

        # 构建各页
        self.build_run_tab(self.tab_run)
        self.build_cal_tab(self.tab_cal)
        self.build_train_pred_tab(self.tab_train)

        # 自动应用配置
        self.apply_run_cfg(self.cfg.get("run", {}))
        self.apply_cal_cfg(self.cfg.get("cal", {}))
        self.apply_train_cfg(self.cfg.get("train", {}))
        self.apply_pred_cfg(self.cfg.get("pred", {}))

        # 启动「一直开着」的 EMG 可视化
        self.after(100, self._start_always_on_view)

        # 关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------- 小工具：暂停/恢复 Run --------------------
    def _pause_run_for_device(self):
        """若 Run 正在占用设备，则先停掉，并标记稍后恢复。"""
        if self.proc_run and self.proc_run.poll() is None:
            try:
                self.proc_run.terminate()
            except Exception:
                pass
            self.proc_run = None
            self._resume_run_after = True
        else:
            self._resume_run_after = False

    def _maybe_resume_run(self):
        """根据标记恢复 Run。"""
        if self._resume_run_after:
            self._resume_run_after = False
            try:
                self._start_always_on_view()
            except Exception as e:
                print("[RESUME RUN FAILED]", e)

    def _watch_pred_end(self):
        """轮询等待 Predict 结束，然后自动恢复 Run。"""
        if self.proc_pred and self.proc_pred.poll() is None:
            self.after(500, self._watch_pred_end)
            return
        self._maybe_resume_run()

    # -------------------- 配置持久化（自动存取） --------------------
    def _load_config(self) -> dict:
        if CONFIG_FILE.exists():
            try:
                return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_config(self):
        cfg = {
            "run": {
                "addr": self.addr_var.get(),
                "view": self.view_var.get(),
                "mode": self.mode_var.get(),
                "win":  self.win_var.get(),
                "step": self.step_var.get(),
                "fftlen": self.fftlen_var.get(),
                "usefft": self.usefft_var.get(),
            },
            "cal": {
                "addr": self.c_addr_var.get(),
                "mode": self.c_mode_var.get(),
                "subject": self.subj_var.get(),
                "actions": self.actions_var.get(),
                "trials": self.trials_var.get(),
                "hold": float(self.hold_var.get()),
                "rest": float(self.rest_var.get()),
                "outdir": self.outdir_var.get(),
                "savefeat": self.savefeat_var.get(),
                "win": self.c_win_var.get(),
                "step": self.c_step_var.get(),
                "fftlen": self.c_fftlen_var.get(),
                "usefft": self.c_usefft_var.get(),
                "onset_scalar": float(self.scalar_var.get()),
                "auto_pipe": self.auto_pipe_var.get(),
            },
            "train": {
                "algo": self.t_algo_var.get(),
                "use_saved_feat": self.t_use_saved_var.get(),
                "model_out": self.t_model_out_var.get(),
                "recalc_win": self.t_win_var.get(),
                "recalc_step": self.t_step_var.get(),
                "recalc_fftlen": self.t_fftlen_var.get(),
                "recalc_usefft": self.t_usefft_var.get(),
            },
            "pred": {
                "model": self.model_path_var.get(),
                "smooth_k": self.smoothk_var.get(),
                "mode": self.pred_mode_var.get()
            }
        }
        try:
            CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print("[WARN] 保存配置失败：", e)

    # -------------------- View / Run tab（一直开着） --------------------
    def apply_run_cfg(self, rcfg: dict):
        self.addr_var.set(rcfg.get("addr", MYO_ADDR_DEFAULT))
        self.view_var.set(rcfg.get("view", "raw"))
        self.mode_var.set(rcfg.get("mode", "filtered"))
        self.win_var.set(rcfg.get("win", 200))
        self.step_var.set(rcfg.get("step", 10))
        self.fftlen_var.set(rcfg.get("fftlen", 64))
        self.usefft_var.set(rcfg.get("usefft", True))

    def build_run_tab(self, parent):
        r = 0
        ttk.Label(parent, text="Myo 地址/UUID:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.addr_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.addr_var, width=46).grid(row=r, column=1, columnspan=2, sticky="we", padx=8)
        r += 1

        ttk.Label(parent, text="显示模式:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.view_var = tk.StringVar()
        ttk.Radiobutton(parent, text="Raw (8通道图)", variable=self.view_var, value="raw").grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(parent, text="Feat (特征计算)", variable=self.view_var, value="feat").grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Label(parent, text="Myo 数据模式:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.mode_var = tk.StringVar()
        ttk.Radiobutton(parent, text="Filtered", variable=self.mode_var, value="filtered").grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(parent, text="Raw",      variable=self.mode_var, value="raw").grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Label(parent, text="窗长 ms:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.win_var = tk.IntVar()
        ttk.Entry(parent, textvariable=self.win_var, width=8).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(parent, text="步长 ms:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.step_var = tk.IntVar()
        ttk.Entry(parent, textvariable=self.step_var, width=8).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(parent, text="FFT 长度:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.fftlen_var = tk.IntVar()
        ttk.Entry(parent, textvariable=self.fftlen_var, width=8).grid(row=r, column=1, sticky="w")
        self.usefft_var = tk.BooleanVar()
        ttk.Checkbutton(parent, text="启用频段能量 (FFT bands)", variable=self.usefft_var).grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Button(parent, text="重启视图窗口", command=self._restart_view).grid(row=r, column=1, pady=12)
        ttk.Button(parent, text="关闭视图窗口", command=self.on_stop_run).grid(row=r, column=2)
        r += 1

        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(2, weight=1)

    def _start_always_on_view(self):
        """启动一个只显示 EMG 数据的 main.py 窗口（如已在运行则跳过）"""
        if self.proc_run and self.proc_run.poll() is None:
            return
        addr = (self.addr_var.get() or MYO_ADDR_DEFAULT).strip()
        cmd = [
            sys.executable, os.path.join(MYOCODE_DIR, "main.py"),
            "--addr", addr,
            "--mode", self.mode_var.get(),
            "--view", self.view_var.get(),
            "--win", str(self.win_var.get() or 200),
            "--step", str(self.step_var.get() or 10),
            "--fftlen", str(self.fftlen_var.get() or 64),
        ]
        if not self.usefft_var.get():
            cmd.append("--no-fft")
        try:
            self.proc_run = subprocess.Popen(cmd)
            self._save_config()
        except Exception as e:
            messagebox.showerror("视图启动失败", str(e))

    def _restart_view(self):
        """手动重启视图窗口（修改了参数想立即生效时用）"""
        if self.proc_run and self.proc_run.poll() is None:
            try:
                self.proc_run.terminate()
            except Exception:
                pass
            self.proc_run = None
        self._start_always_on_view()

    def on_stop_run(self):
        if self.proc_run and self.proc_run.poll() is None:
            self.proc_run.terminate()
            self.proc_run = None
            messagebox.showinfo("已停止", "视图窗口已关闭。")
        else:
            messagebox.showinfo("提示", "视图窗口当前未在运行。")

    # -------------------- Calibrate tab --------------------
    def apply_cal_cfg(self, ccfg: dict):
        self.c_addr_var.set(ccfg.get("addr", MYO_ADDR_DEFAULT))
        self.c_mode_var.set(ccfg.get("mode", "filtered"))
        self.subj_var.set(ccfg.get("subject", "1"))
        self.actions_var.set(ccfg.get("actions", "REST,OPEN,CLOSE"))
        self.trials_var.set(ccfg.get("trials", 3))
        self.hold_var.set(ccfg.get("hold", 3.0))
        self.rest_var.set(ccfg.get("rest", 2.0))
        self.outdir_var.set(ccfg.get("outdir", os.path.join(THIS_DIR, "data")))
        self.savefeat_var.set(ccfg.get("savefeat", True))
        self.c_win_var.set(ccfg.get("win", 200))
        self.c_step_var.set(ccfg.get("step", 10))
        self.c_fftlen_var.set(ccfg.get("fftlen", 64))
        self.c_usefft_var.set(ccfg.get("usefft", True))
        self.scalar_var.set(ccfg.get("onset_scalar", 1.3))
        self.auto_pipe_var.set(ccfg.get("auto_pipe", True))

    def build_cal_tab(self, parent):
        ccfg = self.cfg.get("cal", {})
        r = 0
        ttk.Label(parent, text="Myo 地址/UUID:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.c_addr_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.c_addr_var, width=46).grid(row=r, column=1, columnspan=2, sticky="we", padx=8)
        r += 1

        ttk.Label(parent, text="Myo 数据模式:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.c_mode_var = tk.StringVar()
        ttk.Radiobutton(parent, text="Filtered", variable=self.c_mode_var, value="filtered").grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(parent, text="Raw",      variable=self.c_mode_var, value="raw").grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Label(parent, text="Subject:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.subj_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.subj_var, width=10).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(parent, text="动作列表 (逗号分隔):").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.actions_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.actions_var, width=40).grid(row=r, column=1, columnspan=2, sticky="we", padx=8)
        r += 1

        ttk.Label(parent, text="Trials:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.trials_var = tk.IntVar()
        ttk.Entry(parent, textvariable=self.trials_var, width=8).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(parent, text="Hold 秒:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.hold_var = tk.DoubleVar()
        ttk.Entry(parent, textvariable=self.hold_var, width=8).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(parent, text="Rest 秒:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.rest_var = tk.DoubleVar()
        ttk.Entry(parent, textvariable=self.rest_var, width=8).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(parent, text="输出目录:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.outdir_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.outdir_var, width=40).grid(row=r, column=1, columnspan=2, sticky="we", padx=8)
        r += 1

        self.savefeat_var = tk.BooleanVar()
        ttk.Checkbutton(parent, text="同时保存 TD(+FFT) 特征", variable=self.savefeat_var).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(parent, text="窗长/步长/FFT:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.c_win_var = tk.IntVar()
        self.c_step_var = tk.IntVar()
        self.c_fftlen_var = tk.IntVar()
        ttk.Entry(parent, textvariable=self.c_win_var, width=6).grid(row=r, column=1, sticky="w")
        ttk.Entry(parent, textvariable=self.c_step_var, width=6).grid(row=r, column=1, padx=60, sticky="w")
        ttk.Entry(parent, textvariable=self.c_fftlen_var, width=6).grid(row=r, column=1, padx=120, sticky="w")
        self.c_usefft_var = tk.BooleanVar()
        ttk.Checkbutton(parent, text="启用 FFT bands", variable=self.c_usefft_var).grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Label(parent, text="Onset scalar:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.scalar_var = tk.DoubleVar()
        ttk.Entry(parent, textvariable=self.scalar_var, width=8).grid(row=r, column=1, sticky="w")
        r += 1

        # 自动流程
        self.auto_pipe_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Calibrate 完成后自动 Train + Predict", variable=self.auto_pipe_var).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Button(parent, text="Start Calibrate", command=self.on_start_cal).grid(row=r, column=1, pady=12)
        ttk.Button(parent, text="Stop",            command=self.on_stop_cal).grid(row=r, column=2)
        r += 1

        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(2, weight=1)

        self.apply_cal_cfg(ccfg)

    def on_start_cal(self):
        # Calibrate 需要独占设备 → 暂停视图
        self._pause_run_for_device()

        if any(p and p.poll() is None for p in (self.proc_train, self.proc_pred)):
            messagebox.showinfo("提示", "请先停止 Train/Predict。")
            return
        if self.proc_cal and self.proc_cal.poll() is None:
            messagebox.showinfo("提示", "Calibrate 已在运行。")
            return

        outdir = self.outdir_var.get().strip()
        os.makedirs(outdir, exist_ok=True)
        cmd = [
            sys.executable, os.path.join(MYOCODE_DIR, "calibrate.py"),
            "--addr", self.c_addr_var.get().strip(),
            "--mode", self.c_mode_var.get(),
            "--subject", self.subj_var.get().strip(),
            "--actions", self.actions_var.get().strip(),
            "--trials", str(self.trials_var.get()),
            "--hold", str(self.hold_var.get()),
            "--rest", str(self.rest_var.get()),
            "--outdir", outdir,
            "--onset-scalar", str(self.scalar_var.get()),
        ]
        if self.savefeat_var.get():
            cmd.append("--save-feat")
            cmd += ["--win", str(self.c_win_var.get()),
                    "--step", str(self.c_step_var.get()),
                    "--fftlen", str(self.c_fftlen_var.get())]
            if not self.c_usefft_var.get():
                cmd.append("--no-fft")

        try:
            self.proc_cal = subprocess.Popen(cmd)
            self._save_config()
            messagebox.showinfo("已启动", "Calibrate 子进程已启动。可在终端查看进度。")
            if self.auto_pipe_var.get():
                mat = _calib_mat_path(outdir, self.subj_var.get().strip())
                self._watch_cal_then_train(mat)
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            self._maybe_resume_run()

    def _watch_cal_then_train(self, mat_path: str):
        if self.proc_cal and self.proc_cal.poll() is None:
            self.after(500, lambda: self._watch_cal_then_train(mat_path))
            return
        if not self.proc_cal:
            return
        if self.proc_cal.returncode != 0:
            messagebox.showerror("Calibrate 失败", "Calibrate 进程异常退出，已停止自动流程。")
            self._maybe_resume_run()
            return
        if not os.path.isfile(mat_path):
            messagebox.showerror("未找到数据", f"未找到输出文件：\n{mat_path}")
            self._maybe_resume_run()
            return

        # 启动训练（用 Train & Predict 页配置）
        self._start_train_with_mat(mat_path)

        # 若未开启自动管线（只做 Calibrate），此处恢复视图
        if not self.auto_pipe_var.get():
            self._maybe_resume_run()

    def on_stop_cal(self):
        if self.proc_cal and self.proc_cal.poll() is None:
            self.proc_cal.terminate()
            self.proc_cal = None
            messagebox.showinfo("已停止", "Calibrate 子进程已终止。")
            self._maybe_resume_run()
        else:
            messagebox.showinfo("提示", "Calibrate 当前未在运行。")

    # -------------------- Train & Predict tab --------------------
    def apply_train_cfg(self, tcfg: dict):
        # Train
        self.t_algo_var.set(tcfg.get("algo", "logreg"))              # RLDA/LDA/SVM/RF/logreg
        self.t_use_saved_var.set(tcfg.get("use_saved_feat", True))   # True-> --use-feat
        self.t_model_out_var.set(tcfg.get("model_out", os.path.join(THIS_DIR, "myocode", "models", "model_subj1.pkl")))
        self.t_win_var.set(tcfg.get("recalc_win", 200))
        self.t_step_var.set(tcfg.get("recalc_step", 10))
        self.t_fftlen_var.set(tcfg.get("recalc_fftlen", 64))
        self.t_usefft_var.set(tcfg.get("recalc_usefft", True))

    def build_train_pred_tab(self, parent):
        tcfg = self.cfg.get("train", {})
        pcfg = self.cfg.get("pred", {})

        # ------- Train -------
        lf1 = ttk.LabelFrame(parent, text="Train")
        r = 0
        ttk.Label(lf1, text="算法:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.t_algo_var = tk.StringVar()
        ttk.Combobox(lf1, textvariable=self.t_algo_var,
                     values=["RLDA", "LDA", "SVM", "RF", "logreg"],
                     width=12, state="readonly").grid(row=r, column=1, sticky="w")
        r += 1

        self.t_use_saved_var = tk.BooleanVar()
        ttk.Checkbutton(lf1, text="使用已保存的特征（否则从 raw 重算）", variable=self.t_use_saved_var).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(lf1, text="（若重算特征）窗长/步长/FFT:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.t_win_var = tk.IntVar()
        self.t_step_var = tk.IntVar()
        self.t_fftlen_var = tk.IntVar()
        ttk.Entry(lf1, textvariable=self.t_win_var, width=6).grid(row=r, column=1, sticky="w")
        ttk.Entry(lf1, textvariable=self.t_step_var, width=6).grid(row=r, column=1, padx=60, sticky="w")
        ttk.Entry(lf1, textvariable=self.t_fftlen_var, width=6).grid(row=r, column=1, padx=120, sticky="w")
        self.t_usefft_var = tk.BooleanVar()
        ttk.Checkbutton(lf1, text="启用 FFT bands", variable=self.t_usefft_var).grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Label(lf1, text="模型输出:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.t_model_out_var = tk.StringVar()
        out_entry = ttk.Entry(lf1, textvariable=self.t_model_out_var, width=46)
        out_entry.grid(row=r, column=1, sticky="we", padx=8)
        ttk.Button(lf1, text="保存到…", command=self._browse_model_out).grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Button(lf1, text="Start Train (使用 Calibrate 输出)", command=self.on_start_train).grid(row=r, column=1, pady=8)
        ttk.Button(lf1, text="Stop", command=self.on_stop_train).grid(row=r, column=2)
        lf1.grid_columnconfigure(1, weight=1)
        lf1.pack(fill="x", padx=8, pady=8)

        # ------- Predict（精简：不再显示地址/UUID） -------
        lf2 = ttk.LabelFrame(parent, text="Predict")
        r = 0
        ttk.Label(lf2, text="模型文件:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.model_path_var = tk.StringVar()
        m_entry = ttk.Entry(lf2, textvariable=self.model_path_var, width=46)
        m_entry.grid(row=r, column=1, sticky="we", padx=8)
        ttk.Button(lf2, text="选择文件…", command=self._browse_model).grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Label(lf2, text="Myo 数据模式:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.pred_mode_var = tk.StringVar()
        ttk.Radiobutton(lf2, text="Filtered", variable=self.pred_mode_var, value="filtered").grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(lf2, text="Raw",      variable=self.pred_mode_var, value="raw").grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Label(lf2, text="输出平滑窗口 k:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.smoothk_var = tk.IntVar()
        ttk.Entry(lf2, textvariable=self.smoothk_var, width=8).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Button(lf2, text="Start Predict", command=self.on_start_pred).grid(row=r, column=1, pady=8)
        ttk.Button(lf2, text="Stop",          command=self.on_stop_pred).grid(row=r, column=2)
        lf2.grid_columnconfigure(1, weight=1)
        lf2.pack(fill="x", padx=8, pady=8)

        # 应用配置
        self.apply_train_cfg(tcfg)
        self.apply_pred_cfg(pcfg)

    def _browse_model_out(self):
        f = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(self.t_model_out_var.get() or os.path.join(THIS_DIR,"myocode","models")),
            defaultextension=".pkl",
            filetypes=[("Pickle Model","*.pkl"),("All Files","*.*")]
        )
        if f:
            self.t_model_out_var.set(f)
            # 同步到 Predict 页默认
            self.model_path_var.set(f)
            self._save_config()

    # 训练：直接使用 Calibrate 输出的 .mat 文件（训练不独占设备）
    def on_start_train(self):
        if any(p and p.poll() is None for p in (self.proc_cal, self.proc_pred)):
            messagebox.showinfo("提示", "请先停止 Calibrate/Predict。")
            return
        if self.proc_train and self.proc_train.poll() is None:
            messagebox.showinfo("提示", "Train 已在运行。")
            return

        mat = _calib_mat_path(self.outdir_var.get().strip(), self.subj_var.get().strip())
        if not os.path.isfile(mat):
            messagebox.showerror("错误", f"未找到 Calibrate 输出：\n{mat}")
            return

        cmd = [
            sys.executable, os.path.join(MYOCODE_DIR, "train_classifier.py"),
            "--mat", mat,
            "--actions", self.actions_var.get().strip(),
            "--clf", self.t_algo_var.get(),
            "--model-out", self.t_model_out_var.get().strip(),
        ]
        if self.t_use_saved_var.get():
            cmd.append("--use-feat")
        else:
            cmd += ["--win", str(self.t_win_var.get()),
                    "--step", str(self.t_step_var.get()),
                    "--fftlen", str(self.t_fftlen_var.get())]
            if not self.t_usefft_var.get():
                cmd.append("--no-fft")

        try:
            self.proc_train = subprocess.Popen(cmd)
            self._save_config()
            messagebox.showinfo("已启动", "Train 子进程已启动。请在终端观察训练日志。")
        except Exception as e:
            messagebox.showerror("启动失败", str(e))

    # 供自动管线调用：已知 mat 路径
    def _start_train_with_mat(self, mat_path: str):
        if self.proc_train and self.proc_train.poll() is None:
            return
        cmd = [
            sys.executable, os.path.join(MYOCODE_DIR, "train_classifier.py"),
            "--mat", mat_path,
            "--actions", self.actions_var.get().strip(),
            "--clf", self.t_algo_var.get(),
            "--model-out", self.t_model_out_var.get().strip(),
        ]
        if self.t_use_saved_var.get():
            cmd.append("--use-feat")
        else:
            cmd += ["--win", str(self.t_win_var.get()),
                    "--step", str(self.t_step_var.get()),
                    "--fftlen", str(self.t_fftlen_var.get())]
            if not self.t_usefft_var.get():
                cmd.append("--no-fft")

        try:
            self.proc_train = subprocess.Popen(cmd)
            # 训练结束后自动预测
            self._watch_train_then_predict(self.t_model_out_var.get().strip())
        except Exception as e:
            messagebox.showerror("启动训练失败", str(e))

    def on_stop_train(self):
        if self.proc_train and self.proc_train.poll() is None:
            self.proc_train.terminate()
            self.proc_train = None
            messagebox.showinfo("已停止", "Train 子进程已终止。")
        else:
            messagebox.showinfo("提示", "Train 当前未在运行。")

    # -------------------- Predict（精简地址：复用 Run 页地址） --------------------
    def apply_pred_cfg(self, pcfg: dict):
        self.model_path_var.set(pcfg.get("model", os.path.join(THIS_DIR, "myocode", "models", "model_subj1.pkl")))
        self.smoothk_var.set(pcfg.get("smooth_k", 5))
        self.pred_mode_var.set(pcfg.get("mode", "filtered"))

    def _browse_model(self):
        f = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.model_path_var.get() or os.path.join(THIS_DIR,"myocode","models")),
            filetypes=[("Pickle Model","*.pkl"),("All Files","*.*")]
        )
        if f:
            self.model_path_var.set(f)
            self._save_config()

    def on_start_pred(self):
        # Predict 需要独占设备 → 暂停视图
        self._pause_run_for_device()

        if any(p and p.poll() is None for p in (self.proc_cal, self.proc_train)):
            messagebox.showinfo("提示", "请先停止 Calibrate/Train。")
            return
        if self.proc_pred and self.proc_pred.poll() is None:
            messagebox.showinfo("提示", "Predict 已在运行。")
            return

        model = self.model_path_var.get().strip()
        if not os.path.isfile(model):
            messagebox.showerror("错误", f"模型文件不存在：\n{model}")
            self._maybe_resume_run()
            return

        # 复用 Run 页地址/模式
        addr_for_pred = (self.addr_var.get() or MYO_ADDR_DEFAULT).strip()
        mode_for_pred = self.pred_mode_var.get()

        cmd = [
            sys.executable, os.path.join(MYOCODE_DIR, "run_classifier.py"),
            "--model", model,
            "--addr", addr_for_pred,
            "--mode", mode_for_pred,
            "--smooth-k", str(self.smoothk_var.get())
        ]
        try:
            self.proc_pred = subprocess.Popen(cmd)
            self._save_config()
            # 监视 Predict 结束后自动恢复 Run
            self._watch_pred_end()
            messagebox.showinfo("已启动", "Predict 子进程已启动。终端会打印预测结果。")
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            self._maybe_resume_run()

    def on_stop_pred(self):
        if self.proc_pred and self.proc_pred.poll() is None:
            self.proc_pred.terminate()
            self.proc_pred = None
            messagebox.showinfo("已停止", "Predict 子进程已终止。")
            self._maybe_resume_run()
        else:
            messagebox.showinfo("提示", "Predict 当前未在运行。")

    def _watch_train_then_predict(self, model_path: str):
        if self.proc_train and self.proc_train.poll() is None:
            self.after(500, lambda: self._watch_train_then_predict(model_path))
            return
        if not self.proc_train:
            return
        if self.proc_train.returncode != 0:
            messagebox.showerror("训练失败", "训练进程异常退出，已停止自动流程。")
            self._maybe_resume_run()
            return
        if not os.path.isfile(model_path):
            messagebox.showwarning("提示", f"未找到训练输出模型：\n{model_path}")
            self._maybe_resume_run()
            return
        # 自动启动预测（使用 Predict 区域参数，复用 Run 地址）
        self.model_path_var.set(model_path)
        self.on_start_pred()

    # -------------------- close --------------------
    def on_close(self):
        try:
            self._save_config()
        except Exception:
            pass
        for p in (self.proc_run, self.proc_cal, self.proc_train, self.proc_pred):
            if p and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        self.destroy()

if __name__ == "__main__":
    App().mainloop()
