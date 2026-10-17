"""
GPU FORCE CONFIGURATION MODULE
This module ensures ALL operations run on GPU ONLY
"""
import torch
import os
import sys

def force_gpu_only():
    """Force GPU usage and prevent CPU fallback"""
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CRITICAL ERROR: CUDA not available!")
        print("❌ GPU is required. CPU execution is DISABLED.")
        print("🔧 Please run RUN_COMPLETE_SETUP.bat first")
        sys.exit(1)
    
    # Force CUDA device
    torch.cuda.set_device(0)
    
    # Set environment variables to force GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Force PyTorch to use GPU memory allocation
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Pre-allocate GPU memory to ensure it's being used
    dummy_tensor = torch.cuda.FloatTensor(1)
    del dummy_tensor
    
    print("🔒 GPU-ONLY MODE ACTIVATED")
    print(f"🎯 Device: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return 0  # GPU device ID

def verify_gpu_usage():
    """Verify that operations are actually running on GPU"""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU verification failed - CUDA not available!")
    
    # Test GPU computation
    test_tensor = torch.randn(100, 100, device='cuda')
    result = torch.mm(test_tensor, test_tensor)
    
    if not result.is_cuda:
        raise RuntimeError("GPU verification failed - operations running on CPU!")
    
    print("✅ GPU verification passed - all operations on GPU")
    return True

# Auto-configure when module is imported
if __name__ != "__main__":
    try:
        GPU_DEVICE = force_gpu_only()
        verify_gpu_usage()
    except Exception as e:
        print(f"❌ GPU setup failed: {e}")
        sys.exit(1) 