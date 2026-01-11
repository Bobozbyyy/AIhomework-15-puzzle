from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple
import heapq
import time

# task1.py 's definition
from task1 import NPuzzleProblem, NPuzzleState, i_to_rc


Heuristic = Callable[[NPuzzleState], int]


@dataclass(frozen=True)
class Node:
    state: NPuzzleState
    parent: Optional["Node"]
    action: Optional[str]   # "UP"/"DOWN"/"LEFT"/"RIGHT"
    g: int                  # path cost from start


@dataclass
class AStarResult:
    success: bool
    solution_moves: List[str]
    solution_cost: int
    expanded: int
    generated: int
    max_frontier_size: int
    explored_size: int
    runtime_sec: float


def make_manhattan_heuristic(prob: NPuzzleProblem) -> Heuristic:
    """
    Manhattan distance heuristic for N-puzzle:
    sum over tiles (excluding 0): |r-r_goal| + |c-c_goal|
    """
    n = prob.n
    goal_pos: Dict[int, Tuple[int, int]] = {}
    for idx, tile in enumerate(prob.goal):
        if tile != 0:
            goal_pos[tile] = i_to_rc(idx, n)

    def h(state: NPuzzleState) -> int:
        dist = 0
        for idx, tile in enumerate(state.board):
            if tile == 0:
                continue
            r, c = i_to_rc(idx, n)
            gr, gc = goal_pos[tile]
            dist += abs(r - gr) + abs(c - gc)
        return dist

    return h


def reconstruct_moves(goal_node: Node) -> List[str]:
    moves: List[str] = []
    cur = goal_node
    while cur.parent is not None:
        # cur.action is the move used to reach cur from parent
        moves.append(cur.action)  # type: ignore[arg-type]
        cur = cur.parent
    moves.reverse()
    return moves


def astar(
    prob: NPuzzleProblem,
    initial: NPuzzleState,
    heuristic: Heuristic,
) -> AStarResult:
    """
    A* with:
    - duplicate elimination
    - NO reopening (once in explored/closed, never reinsert)
    - frontier replacement if better g found for same state (UCS-style)
    """
    t0 = time.perf_counter()

    # explored: set of states already expanded
    explored: Set[NPuzzleState] = set()

    # frontier as heap of (f, tie, node)
    # tie is a counter to avoid comparing Node objects
    frontier_heap: List[Tuple[int, int, Node]] = []
    tie = 0

    # best g-value currently known for states IN frontier
    frontier_best_g: Dict[NPuzzleState, int] = {}

    start = Node(state=initial, parent=None, action=None, g=0)
    f0 = start.g + heuristic(start.state)
    heapq.heappush(frontier_heap, (f0, tie, start))
    frontier_best_g[start.state] = 0
    tie += 1

    expanded = 0
    generated = 0
    max_frontier_size = 1

    while True:
        if not frontier_heap:
            # Empty?(frontier) then return failure
            return AStarResult(
                success=False,
                solution_moves=[],
                solution_cost=-1,
                expanded=expanded,
                generated=generated,
                max_frontier_size=max_frontier_size,
                explored_size=len(explored),
                runtime_sec=time.perf_counter() - t0,
            )

        # n <- Pop(frontier)
        f, _, node = heapq.heappop(frontier_heap)

        # "Stale entry" check (lazy deletion):
        # If this state is no longer in frontier_best_g with the same g,
        # it means we have already replaced it with a better one.
        best_g = frontier_best_g.get(node.state)
        if best_g is None or node.g != best_g:
            continue  # skip outdated node

        # GoalTest immediately after pop
        if prob.is_goal(node.state):
            moves = reconstruct_moves(node)
            return AStarResult(
                success=True,
                solution_moves=moves,
                solution_cost=node.g,
                expanded=expanded,
                generated=generated,
                max_frontier_size=max_frontier_size,
                explored_size=len(explored),
                runtime_sec=time.perf_counter() - t0,
            )

        # explored <- explored ∪ n.State
        explored.add(node.state)
        expanded += 1

        # Once we expand it, it is no longer considered in frontier
        frontier_best_g.pop(node.state, None)

        # for each action a in problem.Actions(n.State) do
        for action_tuple, child_state, step_cost in prob.successors(node.state):
            generated += 1

            # action_tuple is like ('DOWN',) in task1.py
            # normalize it to a string:
            if isinstance(action_tuple, tuple):
                action = action_tuple[0]
            else:
                action = str(action_tuple)

            child_g = node.g + step_cost
            child_node = Node(
                state=child_state,
                parent=node,
                action=action,
                g=child_g,
            )

            # if n'.State ∉ explored ∪ States(frontier) then Insert
            if (child_state not in explored) and (child_state not in frontier_best_g):
                child_f = child_g + heuristic(child_state)
                heapq.heappush(frontier_heap, (child_f, tie, child_node))
                frontier_best_g[child_state] = child_g
                tie += 1

            # else if exists n'' in frontier with same state and g(n') < g(n'') then replace
            elif child_state in frontier_best_g:
                old_g = frontier_best_g[child_state]
                if child_g < old_g:
                    # "replace" in frontier:
                    # update best_g, push new entry; old entry becomes stale and will be skipped
                    frontier_best_g[child_state] = child_g
                    child_f = child_g + heuristic(child_state)
                    heapq.heappush(frontier_heap, (child_f, tie, child_node))
                    tie += 1

            # else: child_state in explored -> do NOTHING (NO reopening)

        if len(frontier_best_g) > max_frontier_size:
            max_frontier_size = len(frontier_best_g)


def run_demo(n: int = 4, k: int = 20, seed: int = 42):
    """
    Small runner to verify A* works with Task 1.
    """
    prob = NPuzzleProblem(n)
    h = make_manhattan_heuristic(prob)

    initial = prob.scramble_from_goal(k=k, seed=seed)
    print("Initial state:")
    print(prob.pretty(initial))
    print("Solvable?", prob.is_solvable(initial))
    print()

    res = astar(prob, initial, h)
    print("A* success:", res.success)
    if res.success:
        print("Solution cost:", res.solution_cost)
        print("Moves:", res.solution_moves)
    print("expanded:", res.expanded)
    print("generated:", res.generated)
    print("max_frontier_size:", res.max_frontier_size)
    print("explored_size:", res.explored_size)
    print("runtime_sec:", f"{res.runtime_sec:.6f}")


if __name__ == "__main__":
    run_demo(n=4, k=20, seed=42)
