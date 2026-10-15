"""Export LeRobot's logged training metrics to CSV and JSON (no extra packages)."""

import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean


def read_metrics(log: str) -> list[dict]:
    pattern = re.compile(r"\b(loss|grdn|lr|updt_s|data_s|mem_gb):([-+\d.eE]+)")
    frequency = re.search(r"['\"]log_freq['\"]:\s*(\d+)", log)
    log_freq = int(frequency[1]) if frequency else None
    rows = []
    for line in log.splitlines():
        row = {key: float(value) for key, value in pattern.findall(line)}
        step = re.search(r"\bstep:(\d+(?:\.\d+)?)([KMB]?)\b", line)
        if step and "loss" in row:
            # LeRobot rounds steps to 1K/2K. The progress bar preserves the
            # exact count, so prefer that when it is on the same log line.
            progress = re.search(r"\|\s*(\d+)/\d+\s*\[", line)
            scale = {"": 1, "K": 1000, "M": 1_000_000, "B": 1_000_000_000}
            count = int(float(step[1]) * scale[step[2]])
            if step[2] and progress:
                count = int(progress[1])
                # tqdm can lag one iteration behind the metric. Align its
                # count to the configured logging interval when available.
                if log_freq:
                    count = ((count + log_freq - 1) // log_freq) * log_freq
            row = {"step": count, **row}
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="把训练日志中的 loss 等指标导出为 CSV 和 JSON")
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    rows = read_metrics(args.log.read_text(encoding="utf-8"))
    if not rows:
        parser.error("日志中还没有训练指标")
    csv_path = args.log.with_suffix(".metrics.csv")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    # Upstream logs are already averages across log_freq steps. Compare the
    # first/last ten records as a useful trend, not as a task success metric.
    window = min(10, len(rows))
    summary = {
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "logged_records": len(rows),
        "window_records": window,
        "first_window_mean_loss": mean(row["loss"] for row in rows[:window]),
        "last_window_mean_loss": mean(row["loss"] for row in rows[-window:]),
        "last_logged_loss": rows[-1]["loss"],
        "peak_logged_memory_gb": max(row.get("mem_gb", 0) for row in rows),
    }
    summary_path = args.log.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()
