"""
merge_worker_metadata.py
Une los `dataset_metadata_workerNN.json` generados por cada worker en
un único `dataset_metadata.json` consolidado.
"""
import argparse
import glob
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True,
                        help="Directorio que contiene dataset_metadata_workerNN.json")
    parser.add_argument("--set_id", default="75078-1")
    parser.add_argument("--total_frames", type=int, required=True)
    args = parser.parse_args()

    pattern = os.path.join(args.output_dir, "dataset_metadata_worker*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"❌ No se encontraron metadatas worker en {pattern}")
        sys.exit(1)

    print(f"🔍 Encontrados {len(files)} archivos de workers")
    all_frames = []
    total_saved = 0
    total_skipped = 0
    cameras = set()
    workers = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_frames.extend(data.get("frames", []))
        total_saved += data.get("total_frames_saved", 0)
        total_skipped += data.get("skipped", 0)
        cameras.add(data.get("camera"))
        workers.append({
            "worker_id": data.get("worker_id"),
            "start_frame": data.get("start_frame"),
            "end_frame": data.get("end_frame"),
            "saved": data.get("total_frames_saved"),
            "skipped": data.get("skipped"),
        })
        print(f"  Worker {data.get('worker_id')}: "
              f"frames {data.get('start_frame')}-{data.get('end_frame')}, "
              f"saved={data.get('total_frames_saved')}, skipped={data.get('skipped')}")

    # Ordenar por frame index
    all_frames.sort(key=lambda f: f.get("frame", 0))

    consolidated = {
        "set_id": args.set_id,
        "camera": list(cameras)[0] if len(cameras) == 1 else list(cameras),
        "total_frames_planned": args.total_frames,
        "total_frames_saved": total_saved,
        "total_skipped": total_skipped,
        "num_workers": len(files),
        "workers_summary": workers,
        "frames": all_frames,
    }

    out_path = os.path.join(args.output_dir, "dataset_metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Metadata consolidada: {out_path}")
    print(f"   Total frames: {total_saved}/{args.total_frames}")
    print(f"   Skipped: {total_skipped}")


if __name__ == "__main__":
    main()
