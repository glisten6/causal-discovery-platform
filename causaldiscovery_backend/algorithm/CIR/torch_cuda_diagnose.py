#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
torch_cuda_diagnose.py
Quick diagnostics for PyTorch CUDA availability.
Run with:  python torch_cuda_diagnose.py
"""

from __future__ import annotations
import os, sys, platform, subprocess, json, shutil, textwrap

def run(cmd):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True, timeout=20)
        return p.stdout.strip()
    except Exception as e:
        return f"<error: {e}>"

def find_dlls():
    # Try to locate common CUDA DLLs on Windows
    dlls = ["cudart64_*.dll", "cublas64_*.dll", "cudnn64*.dll"]
    hits = {}
    search_dirs = os.environ.get("PATH","").split(os.pathsep)
    for dll in dlls:
        found = []
        for d in search_dirs:
            try:
                for name in os.listdir(d):
                    if name.lower().startswith(dll.split('*')[0].lower()) and name.lower().endswith(dll.split('*')[-1].lower()):
                        found.append(os.path.join(d, name))
            except Exception:
                pass
        hits[dll] = found
    return hits

def main():
    info = {}
    info["python"] = {
        "version": sys.version.replace("\n"," "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "arch": platform.machine(),
        "cwd": os.getcwd(),
    }
    info["env"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_PATH": os.environ.get("CUDA_PATH"),
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
        "PATH_contains_CUDA": any("cuda" in (p.lower()) for p in os.environ.get("PATH","").split(os.pathsep)),
    }

    # Try importing torch
    torch_block = {}
    try:
        import torch
        torch_block["import_ok"] = True
        torch_block["torch_version"] = torch.__version__
        torch_block["compiled_with_cuda"] = bool(torch.version.cuda)
        torch_block["torch_version_cuda"] = torch.version.cuda
        cudnn_v = None
        try:
            cudnn_v = torch.backends.cudnn.version()
        except Exception:
            pass
        torch_block["cudnn_version"] = cudnn_v
        torch_block["is_available"] = torch.cuda.is_available()
        torch_block["device_count"] = torch.cuda.device_count()
        torch_block["current_device"] = torch.cuda.current_device() if torch_block["is_available"] else None

        devices = []
        for i in range(torch_block["device_count"]):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_GB": round(props.total_memory / (1024**3), 2),
                "multi_processor_count": getattr(props, "multi_processor_count", None),
                "major": getattr(props, "major", None),
                "minor": getattr(props, "minor", None),
                "arch": f"sm_{getattr(props, 'major', '?')}{getattr(props, 'minor', '?')}",
            })
        torch_block["devices"] = devices
    except Exception as e:
        torch_block["import_ok"] = False
        torch_block["error"] = repr(e)
    info["torch"] = torch_block

    # System-level commands
    info["system"] = {
        "nvidia_smi": run("nvidia-smi"),
        "nvcc_version": run("nvcc --version"),
        "where_nvcc_or_which": run("where nvcc" if os.name == "nt" else "which nvcc"),
    }

    # DLL presence check (Windows)
    if os.name == "nt":
        info["dlls"] = find_dlls()

    print("\n==== PyTorch CUDA Diagnostics ====\n")
    print(json.dumps(info, indent=2, ensure_ascii=False))

    print("\n---- Quick guidance ----")
    tips = []
    if not info["torch"].get("import_ok", False):
        tips.append("Torch 导入失败：请确认当前环境已安装 PyTorch。")
    else:
        if not info["torch"].get("compiled_with_cuda", False):
            tips.append("当前 PyTorch 不是 CUDA 版本（CPU-only）。请安装带 CUDA 的 PyTorch 发行版。")
        if info["torch"].get("compiled_with_cuda", False) and not info["torch"].get("is_available", False):
            tips.append("PyTorch 是 CUDA 版，但 torch.cuda.is_available() 为 False：可能是驱动/CUDA 运行时缺失或版本不匹配。")

    if "NVIDIA-SMI" not in (info["system"]["nvidia_smi"] or ""):
        tips.append("无法运行 nvidia-smi：NVIDIA 显卡驱动可能未安装或未正确安装。")

    if os.name == "nt" and not any(info.get("dlls", {}).values()):
        tips.append("未在 PATH 中发现常见 CUDA/cuDNN DLL：请检查 CUDA Toolkit/cuDNN 是否安装并加入 PATH。")

    if tips:
        for i, t in enumerate(tips, 1):
            print(f"{i}. {t}")
    else:
        print("看起来一切正常。")

if __name__ == "__main__":
    main()
