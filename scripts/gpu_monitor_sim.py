#!/usr/bin/env python3
"""
gpu_monitor_sim.py - Simulated GPU monitoring dashboard.

Simulates nvidia-smi output format for learning and demonstration.
On a real cluster, replace simulated data with nvidia-smi queries.

Real usage on cluster:
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

Usage:
    python3 gpu_monitor_sim.py              # live dashboard
    python3 gpu_monitor_sim.py --once       # single snapshot
"""

import argparse
import time
import random
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich import box

console = Console()

# Simulated NVIDIA GPU data (fictional specs for demo purposes)
GPUS = [
    {"index": 0, "name": "NVIDIA GPU [simulated]", "mem_total": 40960},
    {"index": 1, "name": "NVIDIA GPU [simulated]", "mem_total": 40960},
    {"index": 2, "name": "NVIDIA GPU [simulated]", "mem_total": 40960},
    {"index": 3, "name": "NVIDIA GPU [simulated]", "mem_total": 40960},
]

def simulate_gpu_metrics(gpu):
    """Generate realistic GPU metrics."""
    util = random.randint(0, 100)
    mem_used = int(gpu["mem_total"] * util / 100 * random.uniform(0.8, 1.2))
    mem_used = min(mem_used, gpu["mem_total"])
    temp = random.randint(30, 85)
    power = random.randint(50, 400)
    return {
        "util": util,
        "mem_used": mem_used,
        "mem_total": gpu["mem_total"],
        "temp": temp,
        "power": power,
    }

def make_gpu_table(metrics_list):
    """Build GPU status table."""
    table = Table(box=box.ROUNDED, show_header=True, padding=(0, 1))
    table.add_column("GPU", width=4, style="dim")
    table.add_column("Name", width=28)
    table.add_column("Util%", width=8)
    table.add_column("GPU Bar", width=22)
    table.add_column("VRAM Used", width=12)
    table.add_column("VRAM Bar", width=22)
    table.add_column("Temp", width=6)
    table.add_column("Power", width=8)

    for gpu, metrics in zip(GPUS, metrics_list):
        util = metrics["util"]
        gpu_bar_len = int(util / 5)
        gpu_color = "red" if util > 90 else "yellow" if util > 60 else "green"
        gpu_bar = f"[{gpu_color}]" + "█" * gpu_bar_len + f"[/{gpu_color}]" + "░" * (20 - gpu_bar_len)

        mem_pct = metrics["mem_used"] / metrics["mem_total"] * 100
        mem_bar_len = int(mem_pct / 5)
        mem_color = "red" if mem_pct > 90 else "yellow" if mem_pct > 70 else "cyan"
        mem_bar = f"[{mem_color}]" + "█" * mem_bar_len + f"[/{mem_color}]" + "░" * (20 - mem_bar_len)

        temp_color = "red" if metrics["temp"] > 80 else "yellow" if metrics["temp"] > 70 else "green"

        table.add_row(
            str(gpu["index"]),
            gpu["name"],
            f"[{gpu_color}]{util}%[/{gpu_color}]",
            gpu_bar,
            f"{metrics['mem_used']:,} MB",
            mem_bar,
            f"[{temp_color}]{metrics['temp']}C[/{temp_color}]",
            f"{metrics['power']}W",
        )

    return table

def make_slurm_info():
    """Simulate Slurm GPU allocation info."""
    table = Table(box=box.SIMPLE, show_header=True, padding=(0,1))
    table.add_column("JobID", width=10)
    table.add_column("User", width=15)
    table.add_column("GPU", width=5)
    table.add_column("Runtime", width=10)
    table.add_column("Software", width=15)

    jobs = [
        ("12345", "researcher1", "0", "02:34:12", "PyTorch"),
        ("12346", "researcher2", "1", "05:12:44", "TensorFlow"),
        ("12347", "researcher3", "2", "00:45:03", "GROMACS-GPU"),
        ("12348", "researcher4", "3", "11:02:18", "AlphaFold2"),
    ]
    for job in jobs:
        table.add_row(*job)

    return Panel(table, title="[bold]GPU Job Allocations (Slurm)[/bold]", border_style="blue")

def render():
    """Render full dashboard."""
    metrics = [simulate_gpu_metrics(g) for g in GPUS]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = Text(
        f"GPU Monitoring Dashboard | node-gpu01 | {now} | [simulated data]",
        style="bold white", justify="center"
    )

    avg_util = sum(m["util"] for m in metrics) / len(metrics)
    summary = Text(
        f"4 GPUs | Avg utilization: {avg_util:.1f}% | Driver: 535.104.05 | CUDA: 12.2",
        style="dim", justify="center"
    )

    gpu_panel = Panel(
        make_gpu_table(metrics),
        title="[bold]GPU Status (simulated NVIDIA GPU data)[/bold]",
        border_style="green"
    )

    from rich.console import Group
    return Group(header, summary, gpu_panel, make_slurm_info())

def main():
    parser = argparse.ArgumentParser(description="GPU monitoring dashboard")
    parser.add_argument("--once", action="store_true", help="Single snapshot")
    parser.add_argument("--interval", type=int, default=2)
    args = parser.parse_args()

    if args.once:
        console.print(render())
        return

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            live.update(render())
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
