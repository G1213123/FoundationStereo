#!/usr/bin/env python3
"""
Batch 3D Block Coordinate Extraction
Process multiple Unity sequences to extract 3D coordinates of all detected blocks
"""

import os
import argparse
import subprocess
import sys
from dotenv import load_dotenv
import json

def find_available_sequences(unity_data_dir):
    """Find all available sequences in Unity data directory"""
    sequences = []
    if os.path.exists(unity_data_dir):
        for item in os.listdir(unity_data_dir):
            if item.startswith('sequence.') and os.path.isdir(os.path.join(unity_data_dir, item)):
                sequences.append(item)
    return sorted(sequences)

def process_sequence(sequence, step, min_depth, max_depth, conf_threshold, python_exe, script_path):
    """Process a single sequence and return results"""
    output_dir = f"./output_3d_blocks/batch_{sequence}_{step}"
    
    cmd = [
        python_exe, script_path,
        '--sequence', sequence,
        '--step', step,
        '--min_depth', str(min_depth),
        '--max_depth', str(max_depth),
        '--conf_threshold', str(conf_threshold),
        '--output_dir', output_dir
    ]
    
    print(f"🎬 Processing {sequence}/{step}...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Extract results from output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'Block 0: Position=' in line:
                    # Extract position coordinates
                    start = line.find('Position=(') + 10
                    end = line.find(')m,')
                    if start > 9 and end > start:
                        coords_str = line[start:end]
                        coords = [float(x.strip()) for x in coords_str.split(',')]
                        confidence_start = line.find('Confidence=') + 11
                        confidence = float(line[confidence_start:])
                        return {
                            'sequence': sequence,
                            'step': step,
                            'success': True,
                            'position': coords,
                            'confidence': confidence,
                            'output_dir': output_dir
                        }
            
            # If no blocks found
            return {
                'sequence': sequence,
                'step': step,
                'success': True,
                'position': None,
                'confidence': None,
                'output_dir': output_dir
            }
        else:
            return {
                'sequence': sequence,
                'step': step,
                'success': False,
                'error': result.stderr,
                'output_dir': output_dir
            }
    except subprocess.TimeoutExpired:
        return {
            'sequence': sequence,
            'step': step,
            'success': False,
            'error': 'Timeout after 5 minutes',
            'output_dir': output_dir
        }
    except Exception as e:
        return {
            'sequence': sequence,
            'step': step,
            'success': False,
            'error': str(e),
            'output_dir': output_dir
        }

def main():
    parser = argparse.ArgumentParser(description='Batch 3D Block Coordinate Extraction')
    parser.add_argument('--step', type=str, default='step0', help='Unity step to process')
    parser.add_argument('--max_sequences', type=int, default=10, help='Maximum number of sequences to process')
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='YOLO confidence threshold')
    parser.add_argument('--min_depth', type=float, default=0.01, help='Minimum depth (meters)')
    parser.add_argument('--max_depth', type=float, default=50.0, help='Maximum depth (meters)')
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Get paths
    unity_project_path = os.getenv('UNITY_PROJECT_PATH')
    unity_dataset_name = os.getenv('UNITY_DATASET_NAME', 'solo_9')
    python_exe = os.getenv('PYTHON_ENV', sys.executable)
    
    if not unity_project_path:
        print("❌ UNITY_PROJECT_PATH not found in .env file")
        return 1
    
    unity_data_dir = os.path.join(unity_project_path, unity_dataset_name)
    
    print(f"🎯 Batch 3D Block Coordinate Extraction")
    print(f"📁 Unity Data: {unity_data_dir}")
    print(f"🎬 Processing step: {args.step}")
    print(f"🎯 Confidence threshold: {args.conf_threshold}")
    print(f"📏 Depth range: {args.min_depth} - {args.max_depth} meters")
    print("="*60)
    
    # Find available sequences
    sequences = find_available_sequences(unity_data_dir)
    if not sequences:
        print(f"❌ No sequences found in {unity_data_dir}")
        return 1
    
    # Limit sequences
    sequences = sequences[:args.max_sequences]
    print(f"📋 Found {len(sequences)} sequences: {sequences}")
    
    # Process each sequence
    script_path = os.path.join(os.path.dirname(__file__), 'run_3d_integration.py')
    results = []
    
    for sequence in sequences:
        result = process_sequence(sequence, args.step, args.min_depth, args.max_depth, 
                                args.conf_threshold, python_exe, script_path)
        results.append(result)
        
        if result['success']:
            if result['position']:
                x, y, z = result['position']
                print(f"✅ {sequence}: Position=({x:.3f}, {y:.3f}, {z:.3f})m, Confidence={result['confidence']:.3f}")
            else:
                print(f"⚠️  {sequence}: No blocks detected")
        else:
            print(f"❌ {sequence}: Failed - {result.get('error', 'Unknown error')}")
    
    # Save summary results
    summary_file = f"./output_3d_blocks/batch_summary_{unity_dataset_name}_{args.step}.json"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*60)
    print(f"📊 BATCH PROCESSING SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['success']]
    with_blocks = [r for r in successful if r['position']]
    
    print(f"📈 Total sequences processed: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"🧱 With detected blocks: {len(with_blocks)}")
    print(f"📁 Summary saved to: {summary_file}")
    
    if with_blocks:
        print("\n🗺️  BLOCK POSITIONS MAP:")
        print("-" * 50)
        for result in with_blocks:
            x, y, z = result['position']
            print(f"{result['sequence']:>12}: ({x:>6.3f}, {y:>6.3f}, {z:>6.3f})m  conf={result['confidence']:.3f}")
    
    return 0

if __name__ == "__main__":
    exit(main())
