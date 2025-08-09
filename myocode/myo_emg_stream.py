# myo_emg_streamer.py
import asyncio, struct, time
from collections import deque
from bleak import BleakClient

# macOS 下 Myo 的 CoreBluetooth UUID（换成你的）
MYO_ADDR_DEFAULT = "709CFFAF-2E15-0BF7-5E0D-07720E8111E1"

# Myo 固定 GATT
CMD  = "d5060401-a904-deb9-4748-2c7f4a124842"
IMU  = "d5060402-a904-deb9-4748-2c7f4a124842"
EMG0 = "d5060105-a904-deb9-4748-2c7f4a124842"
EMG1 = "d5060205-a904-deb9-4748-2c7f4a124842"
EMG2 = "d5060305-a904-deb9-4748-2c7f4a124842"
EMG3 = "d5060405-a904-deb9-4748-2c7f4a124842"

# 命令
SET_NONE      = bytearray(b"\x01\x03\x00\x00\x00")
DONT_SLEEP    = bytearray(b"\x09\x01\x01")
LOCK_HOLD     = bytearray(b"\x0a\x01\x02")
SET_EMG_IMU_F = bytearray(b"\x01\x03\x02\x01\x00")  # 过滤 EMG+IMU
SET_EMG_IMU_R = bytearray(b"\x01\x03\x03\x01\x00")  # 原始 EMG+IMU

class MyoEMGStreamer:
    """
    连接 Myo，订阅 4 路 EMG。向上游持续产出： (ts_ns, (ch1..ch8))
    每次通知 16 int8 = 2 帧 -> 拆成两条产出。
    """
    def __init__(self, addr: str = MYO_ADDR_DEFAULT, mode: str = "filtered", queue_max: int = 4096):
        assert mode in ("filtered", "raw")
        self.addr = addr
        self.mode = mode
        self._client: BleakClient | None = None
        self._q: asyncio.Queue[tuple[int, tuple[int, ...]]] = asyncio.Queue(maxsize=queue_max)
        self._started = False
        self._notified_frames = 0

    # ---- 下游消费：异步生成器 ----
    async def stream(self):
        """
        使用示例：
            async with MyoEMGStreamer(...) as s:
                async for ts, chs in s.stream():
                    ...   # chs 是长度为8的 tuple[int]
        """
        while True:
            item = await self._q.get()
            yield item

    # ---- 连接与订阅 ----
    async def __aenter__(self):
        self._client = BleakClient(self.addr, timeout=20)
        await self._client.__aenter__()

        # 初始化顺序
        await self._client.write_gatt_char(CMD, SET_NONE)
        await self._client.write_gatt_char(CMD, DONT_SLEEP)
        await self._client.write_gatt_char(CMD, LOCK_HOLD)
        await self._client.write_gatt_char(CMD, SET_EMG_IMU_F if self.mode == "filtered" else SET_EMG_IMU_R)

        # 订阅四路
        for ch in (EMG0, EMG1, EMG2, EMG3):
            await self._client.start_notify(ch, self._on_emg)

        # 1 秒内没数据则自动切另一种模式再试
        await asyncio.sleep(1.0)
        if self._notified_frames == 0:
            self.mode = "raw" if self.mode == "filtered" else "filtered"
            await self._client.write_gatt_char(CMD, SET_EMG_IMU_R if self.mode == "raw" else SET_EMG_IMU_F)

        self._started = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client:
            # 停止通知/关流
            for ch in (EMG0, EMG1, EMG2, EMG3):
                try:
                    await self._client.stop_notify(ch)
                except:
                    pass
            try:
                await self._client.write_gatt_char(CMD, SET_NONE)
            except:
                pass
            await self._client.__aexit__(exc_type, exc, tb)
        self._client = None
        self._started = False

    # ---- 通知回调：推入队列 ----
    def _on_emg(self, _sender, data: bytearray):
        vals = struct.unpack("<16b", data)  # 2 帧
        ts = time.time_ns()
        a = tuple(int(x) for x in vals[:8])
        b = tuple(int(x) for x in vals[8:])
        # 尽量不丢数据；队列满则丢最旧一条（避免阻塞回调）
        self._try_put((ts, a))
        self._try_put((ts, b))
        self._notified_frames += 2

    def _try_put(self, item):
        try:
            self._q.put_nowait(item)
        except asyncio.QueueFull:
            # 丢掉最旧一条，再放入
            try:
                _ = self._q.get_nowait()
            except:
                pass
            try:
                self._q.put_nowait(item)
            except:
                pass
