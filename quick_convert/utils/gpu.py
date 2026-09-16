import subprocess
from collections import Counter


def get_available_gpu() -> int:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    process_counts = Counter()

    for line in process_result.stdout.strip().splitlines():
        if not line.strip():
            continue

        gpu_uuid, _pid = (x.strip() for x in line.split(","))
        process_counts[gpu_uuid] += 1

    available = []

    for line in gpu_result.stdout.strip().splitlines():
        index, uuid, total, used = (x.strip() for x in line.split(","))

        # Categorically exclude GPUs with running compute processes.
        if process_counts[uuid] > 0:
            continue

        total = int(total)
        used = int(used)

        available.append(
            (
                int(index),
                (total - used) / total,
            )
        )

    if not available:
        raise RuntimeError("No unused GPUs are available.")

    # Highest proportion of free VRAM first.
    available.sort(key=lambda gpu: gpu[1], reverse=True)

    return available[0][0]
