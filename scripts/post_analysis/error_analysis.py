#!/usr/bin/env python3
"""
Error Analysis Script
Reads batch_summary.txt and performs trend analysis (linear vs exponential) on Error vs Distance.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def parse_batch_summary(file_path):
    """
    Parses the batch_summary.txt file to extract sequence data.
    Returns a list of tuples: (seq_id, error_dist, cam_dist)
    """
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the start of the data table
    start_idx = -1
    for i, line in enumerate(lines):
        if "Per-Sequence Results:" in line:
            start_idx = i + 2 # Skip header lines
            break
    
    if start_idx == -1:
        print("Error: Could not find data table in summary file.")
        return []

    # Parse data lines
    for line in lines[start_idx:]:
        line = line.strip()
        if not line or line.startswith('-') or "Sequence ID" in line:
            continue
            
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 3:
            seq_id = parts[0]
            try:
                error_dist = float(parts[1])
                cam_dist_str = parts[2]
                if cam_dist_str == "N/A":
                    cam_dist = None
                else:
                    cam_dist = float(cam_dist_str)
                
                if cam_dist is not None:
                    data.append((seq_id, error_dist, cam_dist))
            except ValueError:
                continue
                
    return data

def analyze_and_plot(data, output_dir, show_fits=None):
    """
    Performs trend analysis and plots the results.
    """
    if not data:
        print("No valid data to analyze.")
        return

    if show_fits is None:
        show_fits = ['Linear', 'Exponential', 'Inverse', 'Quadratic']
    
    # Normalize input to title case
    show_fits = [f.title() for f in show_fits]

    seq_ids, errors, cam_dists = zip(*data)
    errors = np.array(errors)
    cam_dists = np.array(cam_dists)

    r2_scores = {}
    
    # Linear Regression: y = mx + c
    if 'Linear' in show_fits:
        z_linear = np.polyfit(cam_dists, errors, 1)
        p_linear = np.poly1d(z_linear)
        y_pred_linear = p_linear(cam_dists)
        r2_linear = r2_score(errors, y_pred_linear)
        r2_scores['Linear'] = r2_linear
        print(f"Linear Fit R^2: {r2_linear:.4f}")

    # Exponential Regression: y = a * e^(bx)
    if 'Exponential' in show_fits:
        log_errors = np.log(errors + 1e-9)
        z_exp = np.polyfit(cam_dists, log_errors, 1)
        b_exp, ln_a_exp = z_exp
        a_exp = np.exp(ln_a_exp)
        y_pred_exp = a_exp * np.exp(b_exp * cam_dists)
        r2_exp = r2_score(errors, y_pred_exp)
        r2_scores['Exponential'] = r2_exp
        print(f"Exponential Fit R^2: {r2_exp:.4f}")

    # Inverse Regression: y = a / x + b
    if 'Inverse' in show_fits:
        inv_cam_dists = 1.0 / cam_dists
        z_inv = np.polyfit(inv_cam_dists, errors, 1)
        a_inv, b_inv = z_inv
        y_pred_inv = a_inv * inv_cam_dists + b_inv
        r2_inv = r2_score(errors, y_pred_inv)
        r2_scores['Inverse'] = r2_inv
        print(f"Inverse Fit R^2: {r2_inv:.4f}")

    # Quadratic Regression: y = ax^2 + bx + c
    if 'Quadratic' in show_fits:
        z_quad = np.polyfit(cam_dists, errors, 2)
        p_quad = np.poly1d(z_quad)
        y_pred_quad = p_quad(cam_dists)
        r2_quad = r2_score(errors, y_pred_quad)
        r2_scores['Quadratic'] = r2_quad
        print(f"Quadratic Fit R^2: {r2_quad:.4f}")

    if r2_scores:
        better_fit = max(r2_scores, key=r2_scores.get)
        print(f"Best fit among selected: {better_fit}")
    else:
        better_fit = "None"

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(cam_dists, errors, c='blue', alpha=0.6, edgecolors='w', s=80, label='Data Points')

    # Generate smooth lines for plotting
    x_range = np.linspace(min(cam_dists), max(cam_dists), 100)
    
    if 'Linear' in show_fits:
        plt.plot(x_range, p_linear(x_range), "r--", alpha=0.8, 
                 label=f'Linear: y={z_linear[0]:.4f}x + {z_linear[1]:.4f} ($R^2$={r2_scores["Linear"]:.2f})')

    if 'Exponential' in show_fits:
        plt.plot(x_range, a_exp * np.exp(b_exp * x_range), "g-.", alpha=0.8, 
                 label=f'Exp: y={a_exp:.4f}e^({b_exp:.4f}x) ($R^2$={r2_scores["Exponential"]:.2f})')

    if 'Inverse' in show_fits:
        plt.plot(x_range, a_inv / x_range + b_inv, "m:", alpha=0.8, 
                 label=f'Inverse: y={a_inv:.4f}/x + {b_inv:.4f} ($R^2$={r2_scores["Inverse"]:.2f})')

    if 'Quadratic' in show_fits:
        plt.plot(x_range, p_quad(x_range), "c-", alpha=0.8, 
                 label=f'Quad: y={z_quad[0]:.4f}x^2 + {z_quad[1]:.4f}x + {z_quad[2]:.4f} ($R^2$={r2_scores["Quadratic"]:.2f})')

    plt.title(f'Corner Detection Error vs Camera Distance\nBest Fit: {better_fit}')
    plt.xlabel('Camera to Ground Truth Distance (m)')
    plt.ylabel('Error Distance (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_path = os.path.join(output_dir, "error_trend_analysis.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Error Trend Analysis')
    parser.add_argument('--summary_file', type=str, default='./scripts/run_files/batch_macro_detection/batch_summary.txt', help='Path to batch_summary.txt')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save the plot (default: same as summary file)')
    parser.add_argument('--show_fits', nargs='+', default=['Quadratic'], 
                        choices=['Linear', 'Exponential', 'Inverse', 'Quadratic'],
                        help='List of fits to display on the graph (default: all)')

    args = parser.parse_args()

    if not os.path.exists(args.summary_file):
        print(f"Error: File not found at {args.summary_file}")
        return

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.summary_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Reading summary from: {args.summary_file}")
    data = parse_batch_summary(args.summary_file)
    print(f"Found {len(data)} data points.")

    analyze_and_plot(data, args.output_dir, args.show_fits)

if __name__ == "__main__":
    main()
