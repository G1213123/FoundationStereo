#!/usr/bin/env python3
"""
Compare depth maps between FoundationStereo demo output and our integration
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def compare_depth_maps():
    """Compare the depth maps to identify scale differences"""
    
    # Load both depth maps
    demo_depth_file = "test_outputs/unity_seq49_cm_scale/depth_meter.npy"
    integration_depth_file = "output_3d_blocks/solo_9_sequence.0_step0/depth_map.npy"
    
    if not os.path.exists(demo_depth_file):
        print(f"❌ Demo depth file not found: {demo_depth_file}")
        return
    
    if not os.path.exists(integration_depth_file):
        print(f"❌ Integration depth file not found: {integration_depth_file}")
        return
    
    print("📊 Loading depth maps for comparison...")
    
    # Load depth maps
    demo_depth = np.load(demo_depth_file)
    integration_depth = np.load(integration_depth_file)
    
    print(f"Demo depth shape: {demo_depth.shape}")
    print(f"Integration depth shape: {integration_depth.shape}")
    
    # Remove infinite/invalid values for statistics
    demo_finite = demo_depth[np.isfinite(demo_depth)]
    integration_finite = integration_depth[np.isfinite(integration_depth)]
    
    print("\n📈 DEPTH STATISTICS COMPARISON:")
    print("=" * 50)
    
    print("🔵 Demo (FoundationStereo):")
    print(f"   Min: {demo_finite.min():.3f}m")
    print(f"   Max: {demo_finite.max():.3f}m") 
    print(f"   Mean: {demo_finite.mean():.3f}m")
    print(f"   Median: {np.median(demo_finite):.3f}m")
    print(f"   Std: {demo_finite.std():.3f}m")
    
    print("\n🟢 Integration (Our Script):")
    print(f"   Min: {integration_finite.min():.6f}m")
    print(f"   Max: {integration_finite.max():.6f}m")
    print(f"   Mean: {integration_finite.mean():.6f}m")
    print(f"   Median: {np.median(integration_finite):.6f}m")
    print(f"   Std: {integration_finite.std():.6f}m")
    
    # Calculate scale factor
    demo_median = np.median(demo_finite)
    integration_median = np.median(integration_finite)
    scale_factor = demo_median / integration_median
    
    print(f"\n📏 SCALE ANALYSIS:")
    print(f"   Demo median / Integration median = {scale_factor:.2f}")
    print(f"   This suggests a {scale_factor:.0f}x scale difference")
    
    # Check if it's a unit conversion issue
    if abs(scale_factor - 100) < 10:
        print(f"   💡 This looks like a meters ↔ centimeters conversion!")
        print(f"   Demo depth might be in meters, integration in centimeters")
    elif abs(scale_factor - 1000) < 100:
        print(f"   💡 This looks like a meters ↔ millimeters conversion!")
        print(f"   Demo depth might be in meters, integration in millimeters")
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Depth Map Comparison: Demo vs Integration', fontsize=16)
    
    # Demo depth visualizations
    demo_vis = demo_depth.copy()
    demo_vis[~np.isfinite(demo_vis)] = 0
    
    # Use percentile clipping for better visualization
    demo_p5, demo_p95 = np.percentile(demo_finite, [5, 95])
    demo_clipped = np.clip(demo_vis, demo_p5, demo_p95)
    
    # Integration depth visualizations  
    integration_vis = integration_depth.copy()
    integration_vis[~np.isfinite(integration_vis)] = 0
    
    integration_p5, integration_p95 = np.percentile(integration_finite, [5, 95])
    integration_clipped = np.clip(integration_vis, integration_p5, integration_p95)
    
    # Plot demo depth
    im1 = axes[0,0].imshow(demo_clipped, cmap='plasma')
    axes[0,0].set_title(f'Demo Depth Map\n({demo_p5:.1f} - {demo_p95:.1f}m)')
    plt.colorbar(im1, ax=axes[0,0], label='Depth (m)')
    
    # Plot integration depth
    im2 = axes[1,0].imshow(integration_clipped, cmap='plasma')
    axes[1,0].set_title(f'Integration Depth Map\n({integration_p5:.3f} - {integration_p95:.3f}m)')
    plt.colorbar(im2, ax=axes[1,0], label='Depth (m)')
    
    # Histograms
    axes[0,1].hist(demo_finite, bins=50, alpha=0.7, color='blue')
    axes[0,1].set_title('Demo Depth Distribution')
    axes[0,1].set_xlabel('Depth (m)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_yscale('log')
    
    axes[1,1].hist(integration_finite, bins=50, alpha=0.7, color='green')
    axes[1,1].set_title('Integration Depth Distribution')
    axes[1,1].set_xlabel('Depth (m)')
    axes[1,1].set_ylabel('Count')
    
    # Scaled comparison
    integration_scaled = integration_depth * scale_factor
    integration_scaled_finite = integration_scaled[np.isfinite(integration_scaled)]
    
    axes[0,2].hist(demo_finite, bins=50, alpha=0.7, color='blue', label='Demo')
    axes[0,2].hist(integration_scaled_finite, bins=50, alpha=0.7, color='green', label=f'Integration × {scale_factor:.0f}')
    axes[0,2].set_title('Scaled Comparison')
    axes[0,2].set_xlabel('Depth (m)')
    axes[0,2].set_ylabel('Count')
    axes[0,2].legend()
    axes[0,2].set_yscale('log')
    
    # Show integration depth in the same scale as demo
    integration_scaled_clipped = np.clip(integration_scaled, demo_p5, demo_p95)
    im3 = axes[1,2].imshow(integration_scaled_clipped, cmap='plasma')
    axes[1,2].set_title(f'Integration Scaled to Demo Range\n(× {scale_factor:.0f})')
    plt.colorbar(im3, ax=axes[1,2], label='Depth (m)')
    
    plt.tight_layout()
    plt.savefig('depth_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Comparison visualization saved to: depth_comparison.png")

if __name__ == "__main__":
    compare_depth_maps()
