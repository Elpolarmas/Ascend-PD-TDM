"""DCMI per-metric latency test."""
import ctypes, time, os, sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

METRICS = {
    1: "DDR",
    2: "AICore",
    3: "AICPU",
    4: "CtrlCPU",
    5: "DDR BW",
    6: "HBM Usage",
    10: "HBM BW",
    12: "VectorCore",
    13: "NPU Overall",
}

dcmi = ctypes.CDLL("/usr/local/dcmi/libdcmi.so")
dcmi.dcmi_init()
dcmi.dcmi_get_device_utilization_rate.restype = ctypes.c_int
dcmi.dcmi_get_device_utilization_rate.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)
]

N = 50  # small N to keep total time reasonable
print(f"DCMI per-metric latency ({N} calls each):\n")
print(f"  {'Metric':15s}  {'Avg(ms)':>10s}  {'Value':>6s}  {'Ret':>5s}")
print(f"  {'-'*15}  {'-'*10}  {'-'*6}  {'-'*5}")

for type_id, name in sorted(METRICS.items()):
    rate = ctypes.c_uint(0)
    # Warmup
    for _ in range(3):
        dcmi.dcmi_get_device_utilization_rate(0, 0, type_id, ctypes.byref(rate))

    start = time.perf_counter()
    for _ in range(N):
        ret = dcmi.dcmi_get_device_utilization_rate(0, 0, type_id, ctypes.byref(rate))
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / N * 1000
    print(f"  {name:15s}  {avg_ms:10.2f}  {rate.value:5d}%  {ret:5d}")

print("\nDone.")
os._exit(0)
