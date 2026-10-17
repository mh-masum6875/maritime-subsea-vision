# GPU FORCE MODULE - ALWAYS USE GPU
from force_gpu_config import GPU_DEVICE, verify_gpu_usage

import torch
import time
import os
from pathlib import Path
from ultralytics import YOLOE

def check_gpu_setup():
    """Display GPU setup information - GPU GUARANTEED"""
    print("=" * 60)
    print("🚢 VESSEL DETECTION WITH GUARANTEED GPU ACCELERATION")
    print("=" * 60)
    
    # Verify GPU is working (will exit if not available)
    verify_gpu_usage()
    
    # Display GPU info
    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"🔧 CUDA available: True (FORCED)")
    print(f"🎯 CUDA version: {torch.version.cuda}")
    print(f"🖥️  GPU device: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("🔒 GPU-ONLY MODE - CPU DISABLED")
    
    print("=" * 60)
    return GPU_DEVICE

def run_vessel_detection():
    """Run vessel detection on video files - GPU ONLY"""
    
    # Check GPU setup (guaranteed GPU)
    device = check_gpu_setup()
    
    # Initialize YOLOE model
    print("🔄 Loading YOLOE model on GPU...")
    try:
        model = YOLOE("yoloe-11l-seg-pf.pt")
        print("✅ Model loaded successfully on GPU")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Set vessel classes
    vessel_names = [
        "barge", "cargo ship", "container ship",
        "cruise ship", "ferry", "river boat",
        "sailboat", "yacht"
    ]
    
    try:
        model.set_classes(vessel_names, model.get_text_pe(vessel_names))
        print(f"🎯 Set detection classes: {', '.join(vessel_names)}")
    except Exception as e:
        print(f"⚠️  Could not set custom classes: {e}")
        print("📝 Using default model classes")
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path('.').glob(f'*{ext}'))
    
    if not video_files:
        print("❌ No video files found in current directory")
        return
    
    print(f"📹 Found {len(video_files)} video files:")
    for i, video in enumerate(video_files, 1):
        print(f"   {i}. {video.name}")
    
    # Process each video
    processed_count = 0
    for video_file in video_files:
        if "Vessel" in video_file.name or video_file.name.startswith(('vessel', 'Vessel')):
            print(f"\n🎬 Processing: {video_file.name}")
            
            try:
                start_time = time.time()
                
                # Clear GPU memory before processing
                torch.cuda.empty_cache()
                
                # Run detection - FORCED GPU ONLY
                results = model.predict(
                    str(video_file),
                    save=True,
                    project="runs/predict",
                    name=f"{video_file.stem}_gpu_results",
                    device=device,          # FORCED GPU device
                    verbose=True,
                    conf=0.25,             # Confidence threshold
                    iou=0.45,              # IoU threshold
                    show_labels=True,
                    show_conf=True
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                print(f"✅ Processing completed in {processing_time:.2f} seconds")
                print(f"📁 Results saved to: runs/predict/{video_file.stem}_gpu_results")
                
                # GPU memory usage
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
                print(f"🔋 Peak GPU memory usage: {gpu_memory:.2f} GB")
                
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing {video_file.name}: {e}")
                continue
    
    print("\n" + "=" * 60)
    print(f"🎉 VESSEL DETECTION COMPLETE! Processed {processed_count} videos on GPU")
    print("=" * 60)

if __name__ == "__main__":
    run_vessel_detection() 