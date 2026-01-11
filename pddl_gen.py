from __future__ import annotations
from typing import List, Tuple
from task1 import NPuzzleProblem, NPuzzleState

def pos_name(i: int) -> str:
    return f"p{i}"

def tile_name(v: int) -> str:
    # v in 1..(n*n-1)
    return f"t{v}"

def neighbors(n: int, i: int) -> List[int]:
    r, c = divmod(i, n)
    out = []
    if r > 0: out.append((r-1)*n + c)
    if r < n-1: out.append((r+1)*n + c)
    if c > 0: out.append(r*n + (c-1))
    if c < n-1: out.append(r*n + (c+1))
    return out

def generate_problem_pddl(
    prob: NPuzzleProblem,
    initial: NPuzzleState,
    problem_name: str = "puz",
) -> str:
    n = prob.n
    N = n * n

    # Objects
    pos_objs = " ".join(pos_name(i) for i in range(N))
    tile_objs = " ".join(tile_name(v) for v in range(1, N))

    # Init facts
    init_facts: List[str] = []

    # adjacency (undirected) — we add both directions
    for i in range(N):
        for j in neighbors(n, i):
            init_facts.append(f"(adj {pos_name(i)} {pos_name(j)})")

    # tiles placement + blank
    blank_i = initial.board.index(0)
    init_facts.append(f"(blank {pos_name(blank_i)})")
    for i, v in enumerate(initial.board):
        if v == 0:
            continue
        init_facts.append(f"(at {tile_name(v)} {pos_name(i)})")

    # Goal: tiles in goal positions and blank at last (because your Task1 goal is that)
    goal_facts: List[str] = []
    for i, v in enumerate(prob.goal):
        if v == 0:
            goal_facts.append(f"(blank {pos_name(i)})")
        else:
            goal_facts.append(f"(at {tile_name(v)} {pos_name(i)})")

    init_block = "\n    ".join(init_facts)
    goal_block = "\n      ".join(goal_facts)

    return f"""(define (problem {problem_name})
  (:domain npuzzle)
  (:objects
    {tile_objs} - tile
    {pos_objs} - pos
  )
  (:init
    {init_block}
  )
  (:goal (and
      {goal_block}
  ))
)
"""

if __name__ == "__main__":
    prob = NPuzzleProblem(4)
    s = prob.scramble_from_goal(k=20, seed=42)
    print(generate_problem_pddl(prob, s, "demo"))
