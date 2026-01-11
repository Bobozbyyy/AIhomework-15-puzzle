# 15-Puzzle: A* and PDDL

This project solves the 15-puzzle problem using two AI approaches:
A* search and PDDL-based planning.


## Files

- `task1.py`: A* implementation for the 15-puzzle  
- `planner_run.py`: PDDL planner integration  
- `pddl_domain.pddl`: PDDL domain definition  
- `pddl_gen.py`: PDDL problem generator  
- `task3_experiment.py`: experimental evaluation  
- `task3_results.csv`: experimental results  

## Requirements

- Python 3.8.8 or higher  
- Windows with WSL (Ubuntu 22.04)  
- Fast Downward planner  

All Python code uses only the standard library.

## How to Run

### A* Search

python task1.py
This runs A* with the Manhattan distance heuristic.

PDDL Planner
python planner_run.py
This generates a PDDL problem, runs the planner, and parses the solution.

Experiments (Task 3)
python task3_experiment.py
This runs both methods on multiple puzzle instances and saves the results to:
task3_results.csv