# -*- coding: utf-8 -*-
# scripts/resolve_and_simulate_assemblies.py

import os
import sys
import subprocess

project_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from generate_synthetic_set import get_ldraw_part_path

ASSEMBLY_MAP = {
    "56823c50": ["56823", "37609"],
    "15301c01": ["15301", "15302"],
    "298c02": ["298", "299"],
    "54930c02": ["54930", "15302"],
    "73590c03a": ["73590"]
}

venv_python = os.path.join(project_root, ".venv", "bin", "python")
blender_path = "/Applications/Blender.app/Contents/MacOS/Blender"
if not os.path.exists(blender_path):
    blender_path = "blender"

def main():
    print("=== RESOLVING AND SIMULATING ASSEMBLY COMPONENTS ===")
    
    # 1. Gather all components to simulate
    components_to_simulate = []
    for assembly, components in ASSEMBLY_MAP.items():
        print(f"Assembly {assembly} splits into: {components}")
        for comp in components:
            # Ensure the LDraw file is downloaded/available
            path = get_ldraw_part_path(comp)
            if path and os.path.exists(path):
                print(f"  ✓ Component {comp} LDraw mesh is ready at: {path}")
                components_to_simulate.append(comp)
            else:
                print(f"  ✗ Failed to find or download LDraw mesh for component: {comp}")
                
    components_to_simulate = sorted(list(set(components_to_simulate)))
    if not components_to_simulate:
        print("No components found to simulate.")
        return

    print(f"\nTotal unique components to simulate: {len(components_to_simulate)}")
    
    # 2. Simulate each component in parallel using simulate_stable_poses.py
    for comp in components_to_simulate:
        print(f"\n>>> Running physics simulation for component: {comp}")
        cmd = [
            blender_path, "--background", "--python", 
            os.path.join(project_root, "scripts", "simulate_stable_poses.py"), 
            "--", "--part", comp, "--save_db"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  ✓ Simulation finished successfully for: {comp}")
        else:
            print(f"  ✗ Simulation failed for: {comp}")
            print(res.stderr)

    # 3. Run post-processing for these parts
    print("\n>>> Running dimensions and silhouette post-processing for new components...")
    parts_arg = ",".join(components_to_simulate)
    subprocess.run([venv_python, os.path.join(project_root, "scripts", "populate_stable_pose_dims.py"), "--parts", parts_arg], check=True)
    subprocess.run([venv_python, os.path.join(project_root, "scripts", "populate_silhouette_areas.py"), "--parts", parts_arg], check=True)
    
    # 4. Synchronize all caches
    print("\n>>> Synchronizing all project caches...")
    subprocess.run([venv_python, os.path.join(project_root, "scratch", "sync_caches_only.py")], check=True)
    
    print("\n=== ASSEMBLY COMPONENTS SIMULATION AND CACHE SYNC COMPLETE ===")

if __name__ == "__main__":
    main()
