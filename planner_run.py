from __future__ import annotations
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

from task1 import NPuzzleProblem
from pddl_gen import generate_problem_pddl


# ======== Config ========

FD_PY = "/home/bobozbyyy/downward/fast-downward.py"

SEARCH = "lazy_greedy([ff()])"

# Fast Downward's plan
PLAN_CANDIDATES = ["sas_plan", "sas_plan.1", "sas_plan.2"]


def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def tail_lines(text: str, n: int = 120) -> str:
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "\n".join(lines[-n:])


def parse_plan(plan_path: Path) -> List[Tuple[str, str, str, str]]:
    """
    Parse a plan file like:
      0: (move t5 p6 p10)
      1: (move t11 p14 p15)
    Return list of tuples: (action, tile, from, to)
    """
    steps: List[Tuple[str, str, str, str]] = []
    if not plan_path.exists():
        return steps
    for line in plan_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip().lower()
        if not line or line.startswith(";"):
            continue
        if "(" not in line:
            continue
        # strip "0:" prefix if exists
        if ":" in line and line.split(":", 1)[0].strip().isdigit():
            line = line.split(":", 1)[1].strip()
        # now line like "(move t5 p6 p10)"
        line = line.strip("()")
        parts = line.split()
        if len(parts) == 4 and parts[0] == "move":
            steps.append((parts[0], parts[1], parts[2], parts[3]))
    return steps


def tile_move_to_blank_direction(n: int, frm: int, to: int) -> str:
    """
    Our PDDL move: tile moves from frm -> to (to is blank).
    The blank moves opposite direction: blank goes to frm.
    We want output as blank-move direction (UP/DOWN/LEFT/RIGHT) consistent with Task1.
    """
    fr, fc = divmod(frm, n)
    tr, tc = divmod(to, n)

    # blank moves from "to" -> "from"
    br, bc = tr, tc
    nbr, nbc = fr, fc
    dr, dc = nbr - br, nbc - bc

    if dr == -1 and dc == 0:
        return "UP"
    if dr == 1 and dc == 0:
        return "DOWN"
    if dr == 0 and dc == -1:
        return "LEFT"
    if dr == 0 and dc == 1:
        return "RIGHT"
    return "UNKNOWN"


def wsl_can_see_file(path: Path) -> bool:
    """
    Quick check: can WSL see this Windows-side file?
    We ask WSL to `ls` it.
    """
    cmd = ["wsl", "--exec", "bash", "-lc", f"ls -1 {path.as_posix()} >/dev/null 2>&1; echo $?"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    # stdout should be "0" if exists, but be robust:
    return p.stdout.strip().endswith("0")


def run_planner(domain: Path, problem: Path) -> Tuple[int, float, str]:

    t0 = time.perf_counter()

    cmd = [
    "wsl", "--exec",
    "python3", FD_PY,
    str(domain),
    str(problem),
    "--search", SEARCH,
    ]


    print("CMD_DEBUG:", cmd)
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, dt, out


def find_plan_file() -> Path | None:
    for name in PLAN_CANDIDATES:
        p = Path(name)
        if p.exists():
            return p
    return None


def main():
    prob = NPuzzleProblem(4)
    initial = prob.scramble_from_goal(k=20, seed=42)

    domain_path = Path("pddl_domain.pddl")
    problem_path = Path("problem.pddl")
    plan_path = Path("sas_plan")  # we request this name

    # Basic sanity checks
    if not domain_path.exists():
        print(f"[ERROR] Domain file not found: {domain_path.resolve()}")
        return

    write_text(problem_path, generate_problem_pddl(prob, initial, "npuz1"))
    if not problem_path.exists():
        print(f"[ERROR] Failed to write problem file: {problem_path.resolve()}")
        return

    print("[INFO] Domain:", domain_path.resolve())
    print("[INFO] Problem:", problem_path.resolve())
    # optional: check if WSL can see these (helps diagnose path issues)
    # If this check fails, planner will fail later anyway.
    # Note: This is conservative; if it returns False, we'll still try planner and show logs.
    try:
        ok_domain = wsl_can_see_file(domain_path)
        ok_problem = wsl_can_see_file(problem_path)
        print(f"[INFO] WSL sees domain? {ok_domain}")
        print(f"[INFO] WSL sees problem? {ok_problem}")
    except Exception as e:
        print("[WARN] WSL visibility check skipped due to error:", e)

    # Clean old plan files to avoid reading stale plans
    for name in PLAN_CANDIDATES:
     p = Path(name)
     if p.exists():
        p.unlink()

    code, runtime, log = run_planner(domain_path, problem_path)

    print("planner return code:", code)
    print("planner runtime_sec:", f"{runtime:.6f}")

    plan_file = find_plan_file()
    if plan_file is None:
        print("No plan found (sas_plan not generated). Showing last log lines:")
        print(tail_lines(log, 120))
        return

    steps = parse_plan(plan_file)
    if not steps:
        print(f"[WARN] Plan file exists but could not be parsed: {plan_file.resolve()}")
        print("Showing last log lines:")
        print(tail_lines(log, 120))
        return

    moves = []
    for _, tile, frm, to in steps:
        frm_i = int(frm[1:])  # "p6" -> 6
        to_i = int(to[1:])
        moves.append(tile_move_to_blank_direction(prob.n, frm_i, to_i))

    print("Plan file:", plan_file.name)
    print("Plan length:", len(steps))
    print("Moves (blank directions):", moves)


if __name__ == "__main__":
    main()
