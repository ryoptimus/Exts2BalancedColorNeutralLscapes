# Gradescope Setup — CSE 568 Project 02: Balanced Colorings and Neutral Landscapes

## Files included in this autograder source

- `tests/grade_project2.py`
- `graphs/`
- `requirements.txt`
- `setup.sh`
- `run_autograder`

This source package intentionally does **not** include `student_solution.py`. Gradescope should import the file submitted by the student.

## Instructor setup

1. Create a Gradescope programming assignment.
2. Upload the autograder source zip containing:
   - `tests/`
   - `graphs/`
   - `requirements.txt`
   - `setup.sh`
   - `run_autograder`
3. Configure the student submission to require one file named:

```text
student_solution.py
```

## Student-facing requirements

Students should submit a `student_solution.py` file containing at least:

```python
def eval_balance(graph, individual):
    ...

def eval_graph(graph, individual):
    ...

def one_neutral(graph, individual):
    ...
```

`eval_graph` is the helper copied from Project 01. It is not directly assigned its own score here, but `one_neutral` depends on it or on equivalent logic.

The grader passes graph objects with the same linked-list representation used in the zyBooks notebook:

```python
graph.n      # number of vertices
graph.k      # number of colors
graph.graph  # adjacency-list heads
```

## Expected encoding

Use the Project 01 bitstring encoding:

```text
bits_per_vertex = ceil(log2(k))
```

Each vertex color is decoded from a consecutive binary chunk, most-significant bit first. For example, with `k = 3`, each vertex uses 2 bits:

```text
00 -> 0
01 -> 1
10 -> 2
11 -> invalid
```

Invalid color encodings should receive fitness `0.0`.

## Scoring

The autograder gives 100 total points:

- Question 1, `eval_balance`: 50 points
- Question 3, `one_neutral`: 50 points

Question 2, `eval_graph`, is treated as a helper for Question 3.

## Notes from the zyBooks port

The uploaded Project 2 zyBooks zip contained a corrupted `graphs/graph_1.txt` placeholder. Because Project 2 reuses the Project 01 graph fixtures and `graph_2.txt` through `graph_4.txt` matched, this package restores `graph_1.txt` from the completed Project 01 reference.

The zyBooks notebook's Question 1 cell in the uploaded file contains NetworkX-style fields such as `graph.nodes` and `graph.edges`. The provided notebook graph class does not expose those fields, so the autograder tests the mathematical specification from the markdown instead.
