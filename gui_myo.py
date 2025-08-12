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
        self.geometry("860x620")

        # 子进程
        self.proc_run = None       # 一直开着的可视化
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

        # 不再自动启动「一直开着」的 EMG 可视化（改为手动点击按钮启动）

        # 关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # 常量：动作选择（REST 总是自动包含，不在复选项中显示）
    ACTION_CHOICES = ["OPEN", "CLOSE", "PALMUP", "PALMDOWN", "INDEX"]

    # -------------------- 通用子进程/参数工具 --------------------
    def _spawn_subprocess(self, cmd: list[str]) -> subprocess.Popen:
        """统一的子进程启动（便于后续加日志/工作目录/平台差异设置）。"""
        return subprocess.Popen(cmd, start_new_session=True, close_fds=True)

    def _terminate_proc_silent(self, attr_name: str):
        """静默终止指定属性的子进程（若存在且在运行）。"""
        proc = getattr(self, attr_name, None)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        setattr(self, attr_name, None)

    def _stop_proc_with_info(self, attr_name: str, stopped_msg: str, not_running_msg: str) -> bool:
        """带消息的停止动作，返回是否确实停止了一个正在运行的进程。"""
        proc = getattr(self, attr_name, None)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
            setattr(self, attr_name, None)
            messagebox.showinfo("已停止", stopped_msg)
            return True
        else:
            messagebox.showinfo("提示", not_running_msg)
            return False

    @staticmethod
    def _extend_cmd_with_win_fft(cmd: list[str], win: int, step: int, fftlen: int, usefft: bool) -> list[str]:
        """追加统一的窗口/步长/FFT 参数；当未启用 FFT 时追加 "--no-fft"。"""
        cmd += ["--win", str(win), "--step", str(step), "--fftlen", str(fftlen)]
        if not usefft:
            cmd.append("--no-fft")
        return cmd

    # -------------------- 小工具：暂停/恢复 Run --------------------
    def _pause_run_for_device(self):
        if self.proc_run and self.proc_run.poll() is None:
            self._terminate_proc_silent("proc_run")
            self._resume_run_after = True
        else:
            self._resume_run_after = False

    def _maybe_resume_run(self):
        if self._resume_run_after:
            self._resume_run_after = False
            try:
                self._start_always_on_view()
            except Exception as e:
                print("[RESUME RUN FAILED]", e)

    # -------------------- 工具：从复选框生成动作字符串 --------------------
    def _get_selected_actions_str(self) -> str:
        """生成 'REST,OPEN,...' 字符串；REST 总是包含；按固定顺序输出。"""
        selected = [a for a in self.ACTION_CHOICES if self.action_vars.get(a) and self.action_vars[a].get()]
        parts = ["REST"] + selected
        return ",".join(parts)

    def _watch_pred_end(self):
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
                # Level 相关
                "smooth_alpha": self.level_alpha_var.get(),
                "scales": self.scales_var.get(),
                "print_hz": self.print_hz_var.get(),
            },
            "cal": {
                "addr": self.c_addr_var.get(),
                "mode": self.c_mode_var.get(),
                "subject": self.subj_var.get(),
                "actions": self._get_selected_actions_str(),
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
        # Level
        self.level_alpha_var.set(rcfg.get("smooth_alpha", 0.30))
        self.scales_var.set(rcfg.get("scales", "1,1,1,1,1,1,1,1"))
        self.print_hz_var.set(rcfg.get("print_hz", 20))

    def build_run_tab(self, parent):
        r = 0
        ttk.Label(parent, text="Myo 地址/UUID:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.addr_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.addr_var, width=52).grid(row=r, column=1, columnspan=3, sticky="we", padx=8)
        r += 1

        ttk.Label(parent, text="显示模式:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.view_var = tk.StringVar()
        ttk.Radiobutton(parent, text="Raw (8通道图)",   variable=self.view_var, value="raw").grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(parent, text="Feat (特征计算)", variable=self.view_var, value="feat").grid(row=r, column=2, sticky="w")
        ttk.Radiobutton(parent, text="Level (平滑强度)", variable=self.view_var, value="level").grid(row=r, column=3, sticky="w")
        r += 1

        ttk.Label(parent, text="Myo 数据模式:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.mode_var = tk.StringVar()
        ttk.Radiobutton(parent, text="Filtered", variable=self.mode_var, value="filtered").grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(parent, text="Raw",      variable=self.mode_var, value="raw").grid(row=r, column=2, sticky="w")
        r += 1

        # feat / raw 通用参数
        ttk.Label(parent, text="窗长 ms / 步长 ms / FFT:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.win_var = tk.IntVar()
        self.step_var = tk.IntVar()
        self.fftlen_var = tk.IntVar()
        ttk.Entry(parent, textvariable=self.win_var, width=6).grid(row=r, column=1, sticky="w")
        ttk.Entry(parent, textvariable=self.step_var, width=6).grid(row=r, column=1, padx=60, sticky="w")
        ttk.Entry(parent, textvariable=self.fftlen_var, width=6).grid(row=r, column=1, padx=120, sticky="w")
        self.usefft_var = tk.BooleanVar()
        ttk.Checkbutton(parent, text="启用 FFT bands（feat）", variable=self.usefft_var).grid(row=r, column=2, sticky="w")
        r += 1

        # Level 模式参数
        lf = ttk.LabelFrame(parent, text="Level 模式参数（整流 + EMA 平滑 + 通道缩放）")
        rr = 0
        ttk.Label(lf, text="EMA α (0~1):").grid(row=rr, column=0, sticky="e", padx=8, pady=6)
        self.level_alpha_var = tk.DoubleVar()
        ttk.Entry(lf, textvariable=self.level_alpha_var, width=8).grid(row=rr, column=1, sticky="w")
        rr += 1

        ttk.Label(lf, text="打印频率 (Hz):").grid(row=rr, column=0, sticky="e", padx=8, pady=6)
        self.print_hz_var = tk.DoubleVar()
        ttk.Entry(lf, textvariable=self.print_hz_var, width=8).grid(row=rr, column=1, sticky="w")
        rr += 1

        ttk.Label(lf, text="8通道 scaler（逗号分隔）:").grid(row=rr, column=0, sticky="e", padx=8, pady=6)
        self.scales_var = tk.StringVar()
        ttk.Entry(lf, textvariable=self.scales_var, width=40).grid(row=rr, column=1, columnspan=2, sticky="we", padx=8)
        rr += 1

        lf.grid_columnconfigure(1, weight=1)
        lf.grid(row=r, column=0, columnspan=4, sticky="we", padx=8, pady=4)
        r += 1

        ttk.Button(parent, text="重启视图窗口", command=self._restart_view).grid(row=r, column=1, pady=12)
        ttk.Button(parent, text="关闭视图窗口", command=self.on_stop_run).grid(row=r, column=2)
        r += 1

        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(2, weight=1)
        parent.grid_columnconfigure(3, weight=1)

    def _start_always_on_view(self):
        """启动一个只显示 EMG 数据的 main_myo_run.py 窗口（如已在运行则跳过）"""
        if self.proc_run and self.proc_run.poll() is None:
            return
        addr = (self.addr_var.get() or MYO_ADDR_DEFAULT).strip()
        view = self.view_var.get()
        cmd = [
            sys.executable, os.path.join(MYOCODE_DIR, "main_myo_run.py"),
            "--addr", addr,
            "--mode", self.mode_var.get(),
            "--view", view,
        ]
        self._extend_cmd_with_win_fft(
            cmd,
            self.win_var.get() or 200,
            self.step_var.get() or 10,
            self.fftlen_var.get() or 64,
            self.usefft_var.get(),
        )

        # Level 模式追加参数
        if view == "level":
            cmd += [
                "--smooth-alpha", str(self.level_alpha_var.get() or 0.30),
                "--scales", self.scales_var.get() or "1,1,1,1,1,1,1,1",
                "--print-hz", str(self.print_hz_var.get() or 20),
            ]
        try:
            self.proc_run = self._spawn_subprocess(cmd)
            self._save_config()
        except Exception as e:
            messagebox.showerror("视图启动失败", str(e))

    def _restart_view(self):
        """手动重启视图窗口（修改了参数想立即生效时用）"""
        if self.proc_run and self.proc_run.poll() is None:
            self._terminate_proc_silent("proc_run")
        self._start_always_on_view()

    def on_stop_run(self):
        self._stop_proc_with_info("proc_run", "视图窗口已关闭。", "视图窗口当前未在运行。")

    # -------------------- Calibrate tab --------------------
    def apply_cal_cfg(self, ccfg: dict):
        self.c_addr_var.set(ccfg.get("addr", MYO_ADDR_DEFAULT))
        self.c_mode_var.set(ccfg.get("mode", "filtered"))
        self.subj_var.set(ccfg.get("subject", "1"))
        # 解析已保存的动作字符串，设置复选框
        saved_actions = str(ccfg.get("actions", "REST,OPEN,CLOSE")).upper()
        chosen = {a.strip() for a in saved_actions.split(',') if a.strip()}
        # REST 总是默认包含，不用勾
        for a in self.ACTION_CHOICES:
            var = self.action_vars.get(a)
            if var is not None:
                var.set(a in chosen)
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
        ttk.Entry(parent, textvariable=self.c_addr_var, width=52).grid(row=r, column=1, columnspan=3, sticky="we", padx=8)
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

        ttk.Label(parent, text="选择动作 (自动包含 REST):").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        # 动作复选框：OPEN/CLOSE/PALMUP/PALMDOWN/INDEX
        self.action_vars = {}
        c_frame = ttk.Frame(parent)
        c_frame.grid(row=r, column=1, columnspan=3, sticky="w", padx=8)
        c = 0
        for a in self.ACTION_CHOICES:
            var = tk.BooleanVar(value=(a in ["OPEN","CLOSE"]))
            self.action_vars[a] = var
            ttk.Checkbutton(c_frame, text=a, variable=var).grid(row=0, column=c, sticky="w", padx=(0,8))
            c += 1
        r += 1

        ttk.Label(parent, text="Trials / Hold(s) / Rest(s):").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.trials_var = tk.IntVar()
        self.hold_var = tk.DoubleVar()
        self.rest_var = tk.DoubleVar()
        ttk.Entry(parent, textvariable=self.trials_var, width=6).grid(row=r, column=1, sticky="w")
        ttk.Entry(parent, textvariable=self.hold_var, width=6).grid(row=r, column=1, padx=60, sticky="w")
        ttk.Entry(parent, textvariable=self.rest_var, width=6).grid(row=r, column=1, padx=120, sticky="w")
        r += 1

        ttk.Label(parent, text="输出目录:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.outdir_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.outdir_var, width=46).grid(row=r, column=1, columnspan=3, sticky="we", padx=8)
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
        parent.grid_columnconfigure(3, weight=1)

        self.apply_cal_cfg(ccfg)

    def on_start_cal(self):
        self._pause_run_for_device()

        if any(p and p.poll() is None for p in (self.proc_train, self.proc_pred)):
            messagebox.showinfo("提示", "请先停止 Train/Predict。")
            return
        if self.proc_cal and self.proc_cal.poll() is None:
            messagebox.showinfo("提示", "Calibrate 已在运行。")
            return

        outdir = self.outdir_var.get().strip()
        os.makedirs(outdir, exist_ok=True)
        # 由复选框生成动作字符串（自动包含 REST，至少要有一个动作被勾选）
        actions_str = self._get_selected_actions_str()
        if actions_str.strip() == "REST":
            messagebox.showerror("错误", "请至少勾选一个非 REST 的动作。")
            self._maybe_resume_run()
            return

        cmd = [
            sys.executable, os.path.join(MYOCODE_DIR, "calibrate.py"),
            "--addr", self.c_addr_var.get().strip(),
            "--mode", self.c_mode_var.get(),
            "--subject", self.subj_var.get().strip(),
            "--actions", actions_str,
            "--trials", str(self.trials_var.get()),
            "--hold", str(self.hold_var.get()),
            "--rest", str(self.rest_var.get()),
            "--outdir", outdir,
            "--onset-scalar", str(self.scalar_var.get()),
        ]
        if self.savefeat_var.get():
            cmd.append("--save-feat")
            self._extend_cmd_with_win_fft(
                cmd,
                self.c_win_var.get(),
                self.c_step_var.get(),
                self.c_fftlen_var.get(),
                self.c_usefft_var.get(),
            )

        try:
            self.proc_cal = self._spawn_subprocess(cmd)
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

        self._start_train_with_mat(mat_path)

        if not self.auto_pipe_var.get():
            self._maybe_resume_run()

    def on_stop_cal(self):
        did_stop = self._stop_proc_with_info("proc_cal", "Calibrate 子进程已终止。", "Calibrate 当前未在运行。")
        if did_stop:
            self._maybe_resume_run()

    # -------------------- Train & Predict tab --------------------
    def apply_train_cfg(self, tcfg: dict):
        self.t_algo_var.set(tcfg.get("algo", "logreg"))
        self.t_use_saved_var.set(tcfg.get("use_saved_feat", True))
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
        out_entry = ttk.Entry(lf1, textvariable=self.t_model_out_var, width=52)
        out_entry.grid(row=r, column=1, columnspan=2, sticky="we", padx=8)
        ttk.Button(lf1, text="保存到…", command=self._browse_model_out).grid(row=r, column=3, sticky="w")
        r += 1

        ttk.Button(lf1, text="Start Train (使用 Calibrate 输出)", command=self.on_start_train).grid(row=r, column=1, pady=8)
        ttk.Button(lf1, text="Stop", command=self.on_stop_train).grid(row=r, column=2)
        lf1.grid_columnconfigure(1, weight=1)
        lf1.pack(fill="x", padx=8, pady=8)

        # ------- Predict（精简：不再显示地址/UUID，复用 Run 地址） -------
        lf2 = ttk.LabelFrame(parent, text="Predict")
        r = 0
        ttk.Label(lf2, text="模型文件:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.model_path_var = tk.StringVar()
        m_entry = ttk.Entry(lf2, textvariable=self.model_path_var, width=52)
        m_entry.grid(row=r, column=1, columnspan=2, sticky="we", padx=8)
        ttk.Button(lf2, text="选择文件…", command=self._browse_model).grid(row=r, column=3, sticky="w")
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

        # 旧数据 .mat（用于保持其他类，和混合/替换）
        ttk.Label(lf2, text="旧数据 .mat:").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.old_mat_var = tk.StringVar()
        old_entry = ttk.Entry(lf2, textvariable=self.old_mat_var, width=52)
        old_entry.grid(row=r, column=1, columnspan=2, sticky="we", padx=8)
        ttk.Button(lf2, text="选择…", command=self._browse_old_mat).grid(row=r, column=3, sticky="w")
        r += 1

        ttk.Label(lf2, text="最大录制秒数(缓冲):").grid(row=r, column=0, sticky="e", padx=8, pady=6)
        self.adapt_max_sec_var = tk.DoubleVar(value=5.0)
        ttk.Entry(lf2, textvariable=self.adapt_max_sec_var, width=8).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Button(lf2, text="Start Predict", command=self.on_start_pred).grid(row=r, column=1, pady=8)
        ttk.Button(lf2, text="Stop",          command=self.on_stop_pred).grid(row=r, column=2)
        lf2.grid_columnconfigure(1, weight=1)
        lf2.pack(fill="x", padx=8, pady=8)

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

        # Train 使用与 Calibrate 相同的动作集合
        actions_str = self._get_selected_actions_str()

        cmd = [
            sys.executable, os.path.join(MYOCODE_DIR, "train_classifier.py"),
            "--mat", mat,
            "--actions", actions_str,
            "--clf", self.t_algo_var.get(),
            "--model-out", self.t_model_out_var.get().strip(),
        ]
        if self.t_use_saved_var.get():
            cmd.append("--use-feat")
        else:
            self._extend_cmd_with_win_fft(
                cmd,
                self.t_win_var.get(),
                self.t_step_var.get(),
                self.t_fftlen_var.get(),
                self.t_usefft_var.get(),
            )

        try:
            self.proc_train = self._spawn_subprocess(cmd)
            self._save_config()
            messagebox.showinfo("已启动", "Train 子进程已启动。请在终端观察训练日志。")
        except Exception as e:
            messagebox.showerror("启动失败", str(e))

    def _start_train_with_mat(self, mat_path: str):
        if self.proc_train and self.proc_train.poll() is None:
            return
        actions_str = self._get_selected_actions_str()

        cmd = [
            sys.executable, os.path.join(MYOCODE_DIR, "train_classifier.py"),
            "--mat", mat_path,
            "--actions", actions_str,
            "--clf", self.t_algo_var.get(),
            "--model-out", self.t_model_out_var.get().strip(),
        ]
        if self.t_use_saved_var.get():
            cmd.append("--use-feat")
        else:
            self._extend_cmd_with_win_fft(
                cmd,
                self.t_win_var.get(),
                self.t_step_var.get(),
                self.t_fftlen_var.get(),
                self.t_usefft_var.get(),
            )

        try:
            self.proc_train = self._spawn_subprocess(cmd)
            self._watch_train_then_predict(self.t_model_out_var.get().strip())
        except Exception as e:
            messagebox.showerror("启动训练失败", str(e))

    def on_stop_train(self):
        self._stop_proc_with_info("proc_train", "Train 子进程已终止。", "Train 当前未在运行。")

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

    def _browse_old_mat(self):
        f = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.old_mat_var.get() or os.path.join(THIS_DIR,"data")),
            filetypes=[("MAT File","*.mat"),("All Files","*.*")]
        )
        if f:
            self.old_mat_var.set(f)
            self._save_config()

    def on_start_pred(self):
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

        addr_for_pred = (self.addr_var.get() or MYO_ADDR_DEFAULT).strip()
        mode_for_pred = self.pred_mode_var.get()

        # 如果未手动选择旧 .mat，则尝试用 Calibrate 页的输出目录+subject 自动推断
        if not self.old_mat_var.get().strip():
            try:
                auto_mat = _calib_mat_path(self.outdir_var.get().strip(), self.subj_var.get().strip())
                if os.path.isfile(auto_mat):
                    self.old_mat_var.set(auto_mat)
            except Exception:
                pass

        cmd = [
            sys.executable, os.path.join(MYOCODE_DIR, "run_classifier.py"),
            "--model", model,
            "--addr", addr_for_pred,
            "--mode", mode_for_pred,
            "--smooth-k", str(self.smoothk_var.get()),
            "--show-gui"
        ]
        if self.old_mat_var.get().strip():
            cmd += ["--old-mat", self.old_mat_var.get().strip(), "--old-use-feat"]
        if self.adapt_max_sec_var.get():
            cmd += ["--adapt-max-sec", str(self.adapt_max_sec_var.get())]
        try:
            self.proc_pred = self._spawn_subprocess(cmd)
            self._save_config()
            self._watch_pred_end()
            messagebox.showinfo("已启动", "Predict 子进程已启动。终端会打印预测结果。")
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            self._maybe_resume_run()

    def on_stop_pred(self):
        did_stop = self._stop_proc_with_info("proc_pred", "Predict 子进程已终止。", "Predict 当前未在运行。")
        if did_stop:
            self._maybe_resume_run()

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
        self.model_path_var.set(model_path)
        self.on_start_pred()

    # -------------------- close --------------------
    def on_close(self):
        try:
            self._save_config()
        except Exception:
            pass
        for attr in ("proc_run", "proc_cal", "proc_train", "proc_pred"):
            self._terminate_proc_silent(attr)
        self.destroy()

if __name__ == "__main__":
    App().mainloop()