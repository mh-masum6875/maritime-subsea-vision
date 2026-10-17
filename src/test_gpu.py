import torch

print("=" * 50)
print("GPU TEST SCRIPT")
print("=" * 50)

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print("✅ GPU computation test passed!")
    
    print(f"✅ GPU is ready for use!")
else:
    print("❌ CUDA not available. Install PyTorch with CUDA support:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("=" * 50) 