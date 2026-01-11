from __future__ import annotations

import csv
import heapq
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from task1 import NPuzzleProblem, NPuzzleState
from pddl_gen import generate_problem_pddl

# =========================
# Config 
# =========================

N = 4  # 15-puzzle = 4x4

# Scaling parameter: scramble depth k
KS = [5, 10, 15, 20, 25, 30]  # increase as you like

# Repeat per k (different random instances)
SEEDS = [0, 1, 2]  # e.g., [0..9] for more stable averages

# Time limits (seconds)
ASTAR_TIME_LIMIT_SEC: Optional[float] = 30.0
PLANNER_TIME_LIMIT_SEC: Optional[float] = 30.0

# Fast Downward (WSL)
FD_PY = "/home/bobozbyyy/downward/fast-downward.py"
PLANNER_SEARCH = "lazy_greedy([ff()])"  # stable baseline; you can try others later

# Output
OUT_CSV = "task3_results.csv"
RUN_DIR = Path("task3_runs")
PLAN_CANDIDATES = ["sas_plan", "sas_plan.1", "sas_plan.2"]


# =========================
# A* (Task 2.1) with metrics
# =========================

def manhattan(prob: NPuzzleProblem, s: NPuzzleState) -> int:
    n = prob.n
    dist = 0
    for idx, v in enumerate(s.board):
        if v == 0:
            continue
        goal_idx = v - 1
        r1, c1 = divmod(idx, n)
        r2, c2 = divmod(goal_idx, n)
        dist += abs(r1 - r2) + abs(c1 - c2)
    return dist


@dataclass
class AStarMetrics:
    solved: bool
    sol_len: Optional[int]
    runtime_sec: float
    expanded: int
    generated: int
    max_frontier: int
    explored_size: int
    branching_avg: float
    branching_min: int
    branching_max: int
    note: str = ""


def astar_with_metrics(prob: NPuzzleProblem, start: NPuzzleState, time_limit: Optional[float]) -> AStarMetrics:
    t0 = time.perf_counter()

    if prob.is_goal(start):
        return AStarMetrics(True, 0, 0.0, 0, 1, 1, 0, 0.0, 0, 0)

    g: Dict[NPuzzleState, int] = {start: 0}
    closed: set[NPuzzleState] = set()
    came_from: Dict[NPuzzleState, Tuple[NPuzzleState, str]] = {}

    # pq items: (f, tie, state)
    pq: List[Tuple[int, int, NPuzzleState]] = []
    tie = 0
    heapq.heappush(pq, (manhattan(prob, start), tie, start))

    expanded = 0
    generated = 1
    max_frontier = 1

    # branching stats over expansions
    branch_counts: List[int] = []

    while pq:
        if time_limit is not None and (time.perf_counter() - t0) > time_limit:
            return AStarMetrics(
                solved=False,
                sol_len=None,
                runtime_sec=time.perf_counter() - t0,
                expanded=expanded,
                generated=generated,
                max_frontier=max_frontier,
                explored_size=len(closed),
                branching_avg=(sum(branch_counts) / len(branch_counts)) if branch_counts else 0.0,
                branching_min=min(branch_counts) if branch_counts else 0,
                branching_max=max(branch_counts) if branch_counts else 0,
                note=f"TIMEOUT>{time_limit}s",
            )

        _, _, cur = heapq.heappop(pq)
        if cur in closed:
            continue

        # Goal check
        if prob.is_goal(cur):
            dt = time.perf_counter() - t0
            branch_avg = (sum(branch_counts) / len(branch_counts)) if branch_counts else 0.0
            return AStarMetrics(
                solved=True,
                sol_len=g[cur],
                runtime_sec=dt,
                expanded=expanded,
                generated=generated,
                max_frontier=max_frontier,
                explored_size=len(closed),
                branching_avg=branch_avg,
                branching_min=min(branch_counts) if branch_counts else 0,
                branching_max=max(branch_counts) if branch_counts else 0,
            )

        closed.add(cur)
        expanded += 1

        # branching factor measured as number of legal moves from expanded node
        legal = prob.legal_moves(cur)
        branch_counts.append(len(legal))

        for mv in legal:
            nxt = prob.apply_move(cur, mv)
            tentative = g[cur] + 1
            if nxt in closed:
                continue
            if tentative < g.get(nxt, 10**18):
                g[nxt] = tentative
                came_from[nxt] = (cur, mv)
                tie += 1
                f = tentative + manhattan(prob, nxt)
                heapq.heappush(pq, (f, tie, nxt))
                generated += 1

        if len(pq) > max_frontier:
            max_frontier = len(pq)

    dt = time.perf_counter() - t0
    branch_avg = (sum(branch_counts) / len(branch_counts)) if branch_counts else 0.0
    return AStarMetrics(
        solved=False,
        sol_len=None,
        runtime_sec=dt,
        expanded=expanded,
        generated=generated,
        max_frontier=max_frontier,
        explored_size=len(closed),
        branching_avg=branch_avg,
        branching_min=min(branch_counts) if branch_counts else 0,
        branching_max=max(branch_counts) if branch_counts else 0,
        note="NO_SOLUTION",
    )


# =========================
# PDDL Planner (Task 2.2) with integration
# =========================

def parse_plan(plan_path: Path) -> List[Tuple[str, str, str, str]]:
    steps: List[Tuple[str, str, str, str]] = []
    if not plan_path.exists():
        return steps
    for line in plan_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip().lower()
        if not line or line.startswith(";"):
            continue
        if "(" not in line:
            continue
        if ":" in line and line.split(":", 1)[0].strip().isdigit():
            line = line.split(":", 1)[1].strip()
        line = line.strip("()")
        parts = line.split()
        if len(parts) == 4 and parts[0] == "move":
            steps.append((parts[0], parts[1], parts[2], parts[3]))
    return steps


def tile_move_to_blank_direction(n: int, frm: int, to: int) -> str:
    fr, fc = divmod(frm, n)
    tr, tc = divmod(to, n)
    dr, dc = fr - tr, fc - tc
    if dr == -1 and dc == 0:
        return "UP"
    if dr == 1 and dc == 0:
        return "DOWN"
    if dr == 0 and dc == -1:
        return "LEFT"
    if dr == 0 and dc == 1:
        return "RIGHT"
    return "UNKNOWN"


def clean_plans(run_dir: Path):
    for name in PLAN_CANDIDATES:
        p = run_dir / name
        if p.exists():
            p.unlink()


def find_plan_file(run_dir: Path) -> Optional[Path]:
    for name in PLAN_CANDIDATES:
        p = run_dir / name
        if p.exists():
            return p
    return None


def extract_fd_times(log: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Try to extract:
      INFO Planner time: Xs
    Return (planner_time, search_time) if found.
    Search time sometimes appears like: [t=...]
    We keep it simple and just parse Planner time.
    """
    planner_time = None
    m = re.search(r"Planner time:\s*([0-9.]+)s", log)
    if m:
        planner_time = float(m.group(1))
    return planner_time, None


@dataclass
class PlannerMetrics:
    solved: bool
    plan_len: Optional[int]
    runtime_sec: float
    planner_time_sec: Optional[float]
    note: str = ""


def planner_run(prob: NPuzzleProblem, initial: NPuzzleState, domain_src: Path, k: int, seed: int,
                time_limit: Optional[float]) -> PlannerMetrics:
    RUN_DIR.mkdir(exist_ok=True)

    run_dir = RUN_DIR / f"k{k}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy domain into run_dir
    (run_dir / "pddl_domain.pddl").write_text(domain_src.read_text(encoding="utf-8"), encoding="utf-8")

    # Write problem into run_dir
    (run_dir / "problem.pddl").write_text(generate_problem_pddl(prob, initial, f"npuz_k{k}_s{seed}"), encoding="utf-8")

    clean_plans(run_dir)

    cmd = [
        "wsl", "--exec",
        "python3", FD_PY,
        "pddl_domain.pddl",
        "problem.pddl",
        "--search", PLANNER_SEARCH,
    ]

    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True, timeout=time_limit)
        dt = time.perf_counter() - t0
        log = (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return PlannerMetrics(False, None, time_limit or 0.0, None, note=f"TIMEOUT>{time_limit}s")
    except Exception as e:
        return PlannerMetrics(False, None, 0.0, None, note=f"EXCEPTION:{e}")

    plan_file = find_plan_file(run_dir)
    planner_time, _ = extract_fd_times(log)

    # save log for reproducibility
    (run_dir / "fd_log.txt").write_text(log, encoding="utf-8")

    if p.returncode != 0 or plan_file is None:
        return PlannerMetrics(False, None, dt, planner_time, note=f"FD_CODE={p.returncode}")

    steps = parse_plan(plan_file)
    if not steps:
        return PlannerMetrics(False, None, dt, planner_time, note="EMPTY_PLAN")

    # Optionally convert to blank moves (for debugging/consistency)
    _blank_moves = []
    for _, _tile, frm, to in steps:
        frm_i = int(frm[1:])
        to_i = int(to[1:])
        _blank_moves.append(tile_move_to_blank_direction(prob.n, frm_i, to_i))

    return PlannerMetrics(True, len(steps), dt, planner_time, note=plan_file.name)


# =========================
# Main experiment loop
# =========================

def main():
    prob = NPuzzleProblem(N)
    domain_path = Path("pddl_domain.pddl")
    if not domain_path.exists():
        raise FileNotFoundError("pddl_domain.pddl not found in current directory.")

    rows: List[Dict[str, object]] = []

    print("=== Task 3 Experiments ===")
    print("KS:", KS)
    print("SEEDS:", SEEDS)
    print("A* time limit:", ASTAR_TIME_LIMIT_SEC)
    print("Planner time limit:", PLANNER_TIME_LIMIT_SEC)
    print("Planner search:", PLANNER_SEARCH)
    print()

    for k in KS:
        for seed in SEEDS:
            initial = prob.scramble_from_goal(k=k, seed=seed)

            # ---------- A* ----------
            print(f"[A*] k={k} seed={seed} ...")
            a = astar_with_metrics(prob, initial, ASTAR_TIME_LIMIT_SEC)
            print(f"     solved={a.solved} len={a.sol_len} time={a.runtime_sec:.3f}s "
                  f"expanded={a.expanded} generated={a.generated} "
                  f"branch(avg/min/max)={a.branching_avg:.2f}/{a.branching_min}/{a.branching_max} "
                  f"max_frontier={a.max_frontier} explored={a.explored_size} note={a.note}")

            rows.append({
                "k": k,
                "seed": seed,
                "method": "A* (Manhattan)",
                "solved": a.solved,
                "solution_len": a.sol_len if a.solved else "",
                "runtime_sec": round(a.runtime_sec, 6),
                "expanded": a.expanded,
                "generated": a.generated,
                "branching_avg": round(a.branching_avg, 4),
                "branching_min": a.branching_min,
                "branching_max": a.branching_max,
                "max_frontier": a.max_frontier,
                "explored_size": a.explored_size,
                "planner_time_sec": "",
                "note": a.note,
            })

            # ---------- PDDL Planner ----------
            print(f"[PDDL] k={k} seed={seed} ...")
            pm = planner_run(prob, initial, domain_path, k, seed, PLANNER_TIME_LIMIT_SEC)
            print(f"     solved={pm.solved} len={pm.plan_len} time={pm.runtime_sec:.3f}s "
                  f"planner_time={pm.planner_time_sec} note={pm.note}")

            rows.append({
                "k": k,
                "seed": seed,
                "method": f"PDDL-FD ({PLANNER_SEARCH})",
                "solved": pm.solved,
                "solution_len": pm.plan_len if pm.solved else "",
                "runtime_sec": round(pm.runtime_sec, 6),
                "expanded": "",
                "generated": "",
                "branching_avg": "",
                "branching_min": "",
                "branching_max": "",
                "max_frontier": "",
                "explored_size": "",
                "planner_time_sec": pm.planner_time_sec if pm.planner_time_sec is not None else "",
                "note": pm.note,
            })

            print()

    # Write CSV
    headers = [
        "k", "seed", "method", "solved", "solution_len", "runtime_sec",
        "expanded", "generated",
        "branching_avg", "branching_min", "branching_max",
        "max_frontier", "explored_size",
        "planner_time_sec", "note"
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Done. Results saved to {OUT_CSV}")
    print(f"Planner run artifacts/logs saved under: {RUN_DIR.resolve()}")


if __name__ == "__main__":
    main()
