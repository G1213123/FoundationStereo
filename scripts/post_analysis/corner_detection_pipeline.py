#!/usr/bin/env python3
"""
Corner Detection Pipeline
Combines depth edge analysis and 3D edge extraction to detect box corners.
Outputs only the final best match corner and distance comparison with ground truth.
"""

import corner_detection_lib as cdl

def main():
    """Main entry point."""
    config = cdl.Config()
    
    # Auto-detect raw_dir if not specified
    if config.raw_dir is None:
        search_roots = [
            './scripts/run_files/batch_outputs',
        ]
        config.raw_dir = cdl.auto_detect_raw_dir(search_roots)
        if config.raw_dir is None:
            print("ERROR: Could not auto-detect raw_dir. Please specify it in the Config class.")
            return 1
        print(f"Auto-detected raw_dir: {config.raw_dir}\n")
    
    try:
        result = cdl.run_pipeline(config)
        return 0 if result is not None else 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
