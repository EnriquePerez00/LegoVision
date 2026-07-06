"""
Utilities para paralelización y optimización de rendering.
- Detecta RAM/CPU disponible (sin psutil)
- Calcula workers dinámicamente (manteniendo 20% libre)
- Divide frames entre workers
"""
import os
import subprocess

def get_system_resources():
    """Devuelve (cpu_cores, available_ram_gb) usando comandos nativos de macOS"""
    # CPU cores
    result = subprocess.run(["sysctl", "-n", "hw.logicalcpu"], capture_output=True, text=True)
    cpu_cores = int(result.stdout.strip())
    
    # RAM disponible en bytes
    result = subprocess.run(["vm_stat"], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    free_pages = 0
    for line in lines:
        if "Pages free:" in line:
            free_pages = int(line.split()[-1].replace(".", ""))
            break
    
    available_ram_gb = (free_pages * 4096) / (1024**3)  # 4KB per page on macOS
    
    return cpu_cores, available_ram_gb

def calculate_optimal_workers(reserve_pct=0.20):
    """
    Calcula workers óptimos manteniendo reserve_pct (20%) libre.
    """
    cpu_cores, ram_gb = get_system_resources()
    
    # Cada worker Blender EEVEE usa ~1.5 cores y ~3-4GB RAM
    cpu_for_workers = int(cpu_cores * (1 - reserve_pct))
    ram_for_workers = int(ram_gb * (1 - reserve_pct))
    
    workers_by_cpu = max(2, cpu_for_workers // 2)  # 2 cores/worker
    workers_by_ram = max(2, ram_for_workers // 4)   # 4GB/worker
    
    optimal_workers = min(workers_by_cpu, workers_by_ram)
    
    # Máximo 4 workers, mínimo 2
    optimal_workers = min(4, max(2, optimal_workers))
    
    print(f"[OPTIMIZE] Available: {cpu_cores} cores, {ram_gb:.1f}GB RAM")
    print(f"[OPTIMIZE] Recommended workers: {optimal_workers}")
    
    return optimal_workers

def split_frame_ranges(total_frames, num_workers):
    """Divide frames entre workers de forma equilibrada"""
    ranges = []
    frames_per_worker = total_frames // num_workers
    remainder = total_frames % num_workers
    
    start = 0
    for i in range(num_workers):
        end = start + frames_per_worker + (1 if i < remainder else 0)
        ranges.append((start, end))
        start = end
    
    return ranges

if __name__ == "__main__":
    workers = calculate_optimal_workers()
    ranges = split_frame_ranges(2000, workers)
    print(f"\nFrame ranges para {workers} workers:")
    for i, (s, e) in enumerate(ranges):
        print(f"  Worker {i}: frames {s}-{e-1} ({e-s} frames)")
