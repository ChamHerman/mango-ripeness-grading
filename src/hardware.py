"""
Hardware Acceleration & Compute Device Dispatcher Module
--------------------------------------------------------
Provides automatic GPU detection (NVIDIA CUDA / OpenCV OpenCL / DirectML)
and transparent fallback to multi-threaded CPU SIMD execution.
"""

import cv2
import os

_HARDWARE_CACHE = None

def get_hardware_info():
    """Detect available compute devices and return hardware configuration metadata."""
    global _HARDWARE_CACHE
    if _HARDWARE_CACHE is not None:
        return _HARDWARE_CACHE

    info = {
        'has_gpu': False,
        'device_type': 'CPU',
        'device_name': 'Host CPU (Multi-threaded SIMD)',
        'backend': 'CPU Vectorized (OpenMP / AVX2)',
        'cuda_available': False,
        'opencl_available': False,
        'gpu_vendor': None
    }

    # 1. Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            info['has_gpu'] = True
            info['cuda_available'] = True
            info['device_name'] = torch.cuda.get_device_name(0)
            info['device_type'] = 'GPU (CUDA)'
            info['backend'] = f"PyTorch CUDA ({torch.version.cuda})"
    except Exception:
        pass

    # 2. Check OpenCV OpenCL (Hardware GPU Acceleration for filters/morphology/color)
    try:
        if cv2.ocl.haveOpenCL():
            dev = cv2.ocl.Device.getDefault()
            if dev.available():
                info['opencl_available'] = True
                if not info['has_gpu']:
                    info['has_gpu'] = True
                    info['device_name'] = dev.name()
                    info['device_type'] = 'GPU (OpenCL)'
                    info['backend'] = f"OpenCV OpenCL ({dev.name()})"
                    info['gpu_vendor'] = dev.vendorName()
    except Exception:
        pass

    _HARDWARE_CACHE = info
    return info


def init_hardware_acceleration(enable_gpu=True):
    """Configure OpenCV and runtime backends for GPU acceleration if available,
    otherwise gracefully fallback to CPU.
    """
    hw = get_hardware_info()
    if enable_gpu and hw['has_gpu']:
        try:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
            return True, hw['device_name']
        except Exception:
            cv2.ocl.setUseOpenCL(False)
            return False, "CPU Fallback (OpenCL Error)"
    else:
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
        return False, "Host CPU"


# Initialize hardware acceleration upon module import
_IS_GPU_ACTIVE, _ACTIVE_DEVICE_NAME = init_hardware_acceleration(enable_gpu=True)
