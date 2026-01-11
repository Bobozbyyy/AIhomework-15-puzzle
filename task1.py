from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Dict
import random
import time

print("running once")

Move = str  # "UP", "DOWN", "LEFT", "RIGHT"

MOVES: Tuple[Move, ...] = ("UP", "DOWN", "LEFT", "RIGHT")


def rc_to_i(r: int, c: int, n: int) -> int:
    return r * n + c


def i_to_rc(i: int, n: int) -> Tuple[int, int]:
    return divmod(i, n)


@dataclass(frozen=True)
class NPuzzleState:
    """
    Immutable state:
    board: length n*n, contains numbers 0..n*n-1, 0 is blank.
    """
    n: int
    board: Tuple[int, ...]

    def __post_init__(self):
        if len(self.board) != self.n * self.n:
            raise ValueError("board length must be n*n")
        # Basic validation 
        s = set(self.board)
        if s != set(range(self.n * self.n)):
            raise ValueError("board must contain exactly numbers 0..n*n-1")

    def blank_index(self) -> int:
        return self.board.index(0)


class NPuzzleProblem:
    """
      modeling :
    - state representation
    - goal test
    - successor function
    - solvability test
    - instance generation
    """

    def __init__(self, n: int):
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.goal = tuple(list(range(1, n * n)) + [0])

    # ---------- Parsing / formatting ----------

    def from_list(self, tiles: List[int]) -> NPuzzleState:
        return NPuzzleState(self.n, tuple(tiles))

    def pretty(self, state: NPuzzleState) -> str:
        n = state.n
        lines = []
        for r in range(n):
            row = []
            for c in range(n):
                v = state.board[rc_to_i(r, c, n)]
                row.append(" ." if v == 0 else f"{v:2d}")
            lines.append(" ".join(row))
        return "\n".join(lines)

    # ---------- Goal & validity ----------

    def is_goal(self, state: NPuzzleState) -> bool:
        return state.board == self.goal

    # ---------- Moves & successors ----------

    def legal_moves(self, state: NPuzzleState) -> List[Move]:
        n = state.n
        b = state.blank_index()
        r, c = i_to_rc(b, n)

        moves = []
        if r > 0:
            moves.append("UP")
        if r < n - 1:
            moves.append("DOWN")
        if c > 0:
            moves.append("LEFT")
        if c < n - 1:
            moves.append("RIGHT")
        return moves

    def apply_move(self, state: NPuzzleState, move: Move) -> NPuzzleState:
        """
        Move the blank in the given direction by swapping blank with neighboring tile.
        """
        n = state.n
        b = state.blank_index()
        r, c = i_to_rc(b, n)

        if move == "UP":
            nr, nc = r - 1, c
        elif move == "DOWN":
            nr, nc = r + 1, c
        elif move == "LEFT":
            nr, nc = r, c - 1
        elif move == "RIGHT":
            nr, nc = r, c + 1
        else:
            raise ValueError(f"Unknown move: {move}")

        # Check legality
        if not (0 <= nr < n and 0 <= nc < n):
            raise ValueError(f"Illegal move {move} from blank at {(r, c)}")

        nb = rc_to_i(nr, nc, n)

        new_board = list(state.board)
        new_board[b], new_board[nb] = new_board[nb], new_board[b]
        return NPuzzleState(n, tuple(new_board))

    def successors(self, state: NPuzzleState) -> List[Tuple[Tuple[Move], NPuzzleState, int]]:
        """
        For search algorithms later:
        returns list of (action, next_state, step_cost)
        Here action is (move,) and cost is 1.
        """
        succ = []
        for m in self.legal_moves(state):
            ns = self.apply_move(state, m)
            succ.append(((m,), ns, 1))
        return succ

    # ---------- Solvability ----------

    def inversion_count(self, board: Tuple[int, ...]) -> int:
        """
        Count inversions ignoring 0.
        """
        arr = [x for x in board if x != 0]
        inv = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    inv += 1
        return inv

    def is_solvable(self, state: NPuzzleState) -> bool:
        """
        Standard N-puzzle solvability using inversion:
        - If n is odd: solvable iff inversion count is even.
        - If n is even: solvable iff (inversions + blank_row_from_bottom) is even,
          where blank_row_from_bottom counts from 1 at bottom row.
        """
        n = state.n
        inv = self.inversion_count(state.board)
        blank_i = state.blank_index()
        blank_r, _ = i_to_rc(blank_i, n)
        blank_row_from_bottom = n - blank_r  # bottom row => 1

        if n % 2 == 1:
            return inv % 2 == 0
        else:
            return (inv + blank_row_from_bottom) % 2 == 1

    # ---------- Instance generation ----------

    def goal_state(self) -> NPuzzleState:
        return NPuzzleState(self.n, self.goal)

    def random_state(self, seed: Optional[int] = None) -> NPuzzleState:
        """
        Generate a random solvable state by shuffling until solvable.
        """
        if seed is not None:
            random.seed(seed)
        tiles = list(range(self.n * self.n))
        while True:
            random.shuffle(tiles)
            s = self.from_list(tiles)
            if self.is_solvable(s):
                return s

    def scramble_from_goal(self, k: int, seed: Optional[int] = None) -> NPuzzleState:
        """
        Generate a guaranteed-solvable instance by applying k random legal moves from the goal.
        Great for Task 3 scaling: k = 10,20,30,...
        """
        if seed is not None:
            random.seed(seed)

        s = self.goal_state()
        last_move = None
        opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

        for _ in range(k):
            moves = self.legal_moves(s)
            # avoid immediately undoing last move 
            if last_move is not None and opposite[last_move] in moves and len(moves) > 1:
                moves.remove(opposite[last_move])
            m = random.choice(moves)
            s = self.apply_move(s, m)
            last_move = m
        return s


def demo():
    n = 4
    prob = NPuzzleProblem(n)

    print("Goal state:")
    g = prob.goal_state()
    print(prob.pretty(g))
    print("Solvable?", prob.is_solvable(g))
    print()

    print("Scrambled instance (k=20):")
    s = prob.scramble_from_goal(k=20, seed=42)
    print(prob.pretty(s))
    print("Solvable?", prob.is_solvable(s))
    print()

    print("Successors (one step):")
    succ = prob.successors(s)
    print("Legal moves:", [a[0] for a, _, _ in succ])
    # show first successor
    a, ns, cost = succ[0]
    print(f"\nApply move {a[0]}, cost={cost}")
    print(prob.pretty(ns))


if __name__ == "__main__":
    demo()
