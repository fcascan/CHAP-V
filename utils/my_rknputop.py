# -*- coding: utf-8 -*-
"""my_rknputop.py
by fcascan 2025
"""
import re
import time

SLEEP_TIME = 0.5

def get_npu_load():
    try:
        with open("/sys/kernel/debug/rknpu/load", "r") as f:
            rkload = f.read()
        return [int(pct) for _, pct in re.findall(r"Core([0-9]+):\s*([0-9]+)%", rkload)]
    except Exception as e:
        print(f"[ERROR] Cannot read NPU load: {e}")
        return []

def log_npu_usage():
    while True:
        npu_load = get_npu_load()
        npu_load_str = ", ".join([f"NPU_CORE_{i} Load = {load}%" for i, load in enumerate(npu_load)])
        print(f"[RKNPUTOP] {npu_load_str}")
        time.sleep(SLEEP_TIME)
