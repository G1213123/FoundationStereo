#!/usr/bin/env python3
"""Convenient launcher for Unity YOLO inference"""

import os

if __name__ == "__main__":
    print("🚀 Unity YOLO Launcher")
    print("=" * 50)
    print()
    print("📁 Folder Structure:")
    print("   scripts/")
    print("   ├── unity_yolo/          # Unity + YOLO scripts")
    print("   │   ├── test_yolo_best_only.py   # Main inference script")
    print("   │   └── ...              # Other Unity/YOLO scripts")
    print("   └── foundation_stereo/   # Foundation Stereo scripts")
    print()
    print("🎯 To run Unity YOLO inference:")
    print("   python scripts/unity_yolo/test_yolo_best_only.py")
    print()
    print("📊 Results will be saved to: ./results/")
    print()
    
    # Ask user if they want to run it
    choice = input("Run Unity YOLO inference now? (y/n): ").lower().strip()
    if choice in ['y', 'yes']:
        print()
        os.system("python scripts/unity_yolo/test_yolo_best_only.py")
    else:
        print("✅ Use the command above to run when ready!")
