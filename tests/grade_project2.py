#!/usr/bin/env python3
"""Gradescope grader for CSE 568 Project 02.

The grader imports /autograder/submission/student_solution.py and tests:
  - eval_balance(graph, individual)
  - one_neutral(graph, individual)

It writes /autograder/results/results.json using Gradescope's JSON format.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

RESULTS_DIR = os.environ.get("GRADESCOPE_RESULTS_DIR", "/autograder/results")
SOURCE_DIR = os.environ.get("GRADESCOPE_SOURCE_DIR", "/autograder/source")
SUBMISSION_DIR = os.environ.get("GRADESCOPE_SUBMISSION_DIR", "/autograder/submission")

# Make local execution easy while preserving Gradescope behavior.
if not os.path.isdir(SOURCE_DIR):
    SOURCE_DIR = os.getcwd()
if not os.path.isdir(SUBMISSION_DIR):
    SUBMISSION_DIR = os.getcwd()

for path in [SUBMISSION_DIR, SOURCE_DIR, os.path.join(SOURCE_DIR, "tests")]:
    if path not in sys.path:
        sys.path.insert(0, path)


class AdjNode:
    def __init__(self, vertex: int):
        self.vertex = vertex
        self.next = None


class Graph:
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self.graph = [None] * n
        self.color_assignment = [None] * n

    def get_node(self, s: int):
        return self.graph[s]

    def add_edge(self, src: int, dest: int) -> None:
        node = AdjNode(dest)
        node.next = self.graph[src]
        self.graph[src] = node

        node = AdjNode(src)
        node.next = self.graph[dest]
        self.graph[dest] = node


def load_graph(relative_path: str) -> Graph:
    path = os.path.join(SOURCE_DIR, relative_path)
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    n = int(lines[0])
    k = int(lines[1])
    graph = Graph(n, k)

    for line in lines[2:]:
        src, dest = map(int, line.split())
        graph.add_edge(src, dest)

    return graph


def unique_edges(graph: Graph) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for src in range(graph.n):
        node = graph.graph[src]
        while node is not None:
            dest = node.vertex
            if src < dest:
                edges.append((src, dest))
            node = node.next
    return edges


def bits_per_vertex(k: int) -> int:
    return math.ceil(math.log2(k))


def colors_to_bits(colors: Sequence[int], k: int) -> List[int]:
    width = bits_per_vertex(k)
    bits: List[int] = []
    for color in colors:
        if color < 0 or color >= 2 ** width:
            raise ValueError(f"Color {color} cannot be represented with {width} bits")
        bits.extend(int(bit) for bit in format(color, f"0{width}b"))
    return bits


def decode_colors(graph: Graph, individual: Sequence[int]) -> List[int]:
    width = bits_per_vertex(graph.k)
    expected_len = graph.n * width
    if len(individual) != expected_len:
        raise ValueError(f"Expected individual of length {expected_len}, got {len(individual)}")

    colors: List[int] = []
    for vertex in range(graph.n):
        start = vertex * width
        chunk = individual[start : start + width]
        colors.append(int("".join(str(int(bit)) for bit in chunk), 2))
    return colors


def reference_eval_graph(graph: Graph, individual: Sequence[int]) -> float:
    try:
        colors = decode_colors(graph, individual)
    except Exception:
        return 0.0

    if any(color < 0 or color >= graph.k for color in colors):
        return 0.0

    edges = unique_edges(graph)
    if not edges:
        return 0.0

    satisfied = sum(1 for src, dest in edges if colors[src] != colors[dest])
    return satisfied / len(edges)


def reference_eval_balance(graph: Graph, individual: Sequence[int]) -> float:
    try:
        colors = decode_colors(graph, individual)
    except Exception:
        return 0.0

    if any(color < 0 or color >= graph.k for color in colors):
        return 0.0

    edge_score = reference_eval_graph(graph, individual)
    balance_product = 1.0
    for color in range(graph.k):
        balance_product *= colors.count(color) / graph.n

    return edge_score * balance_product


def reference_one_neutral(graph: Graph, individual: Sequence[int]) -> float:
    colors = decode_colors(graph, individual)
    if any(color < 0 or color >= graph.k for color in colors):
        return 0.0

    original_fitness = reference_eval_graph(graph, individual)
    total_mutations = graph.n * (graph.k - 1)
    neutral_mutations = 0

    for vertex, old_color in enumerate(colors):
        for new_color in range(graph.k):
            if new_color == old_color:
                continue
            mutant_colors = list(colors)
            mutant_colors[vertex] = new_color
            mutant = colors_to_bits(mutant_colors, graph.k)
            if math.isclose(reference_eval_graph(graph, mutant), original_fitness, rel_tol=1e-12, abs_tol=1e-12):
                neutral_mutations += 1

    return neutral_mutations / total_mutations if total_mutations else 0.0


def numeric_value(value):
    """Accept DEAP-style (x,), one-item lists, numpy scalars, or plain floats."""
    if isinstance(value, (tuple, list)):
        if not value:
            raise AssertionError("Returned an empty tuple/list")
        value = value[0]
    try:
        return float(value)
    except Exception as exc:
        raise AssertionError(f"Return value {value!r} is not numeric") from exc


def assert_close(actual, expected: float, *, label: str, tol: float = 1e-9) -> None:
    actual_float = numeric_value(actual)
    if not math.isclose(actual_float, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual_float}")


@dataclass
class GradedCheck:
    name: str
    max_score: float
    func: Callable[[object], None]


def check_eval_balance_sample(student) -> None:
    graph = load_graph("graphs/graph_1.txt")
    individual = colors_to_bits([1, 0, 2, 1], graph.k)
    expected = 0.03125
    assert_close(student.eval_balance(graph, individual), expected, label="eval_balance sample")


def check_eval_balance_graph2(student) -> None:
    graph = load_graph("graphs/graph_2.txt")
    individual = colors_to_bits([0, 1, 2, 0, 1], graph.k)
    expected = reference_eval_balance(graph, individual)
    assert_close(student.eval_balance(graph, individual), expected, label="eval_balance graph_2")


def check_eval_balance_invalid_encoding(student) -> None:
    graph = load_graph("graphs/graph_1.txt")
    # With k = 3 and 2 bits per vertex, color 3 is invalid.
    invalid_individual = colors_to_bits([3, 0, 1, 2], 4)
    assert_close(student.eval_balance(graph, invalid_individual), 0.0, label="eval_balance invalid encoding")


def check_eval_balance_zero_color_bucket(student) -> None:
    graph = load_graph("graphs/graph_2.txt")
    # This uses only colors 0 and 1 on a 3-color graph, so the product term should be zero.
    individual = colors_to_bits([0, 1, 0, 1, 0], graph.k)
    assert_close(student.eval_balance(graph, individual), 0.0, label="eval_balance missing color bucket")


def check_one_neutral_sample(student) -> None:
    graph = load_graph("graphs/graph_1.txt")
    individual = colors_to_bits([1, 0, 2, 1], graph.k)
    expected = 0.25
    assert_close(student.one_neutral(graph, individual), expected, label="one_neutral sample")


def check_one_neutral_graph2(student) -> None:
    graph = load_graph("graphs/graph_2.txt")
    individual = colors_to_bits([0, 1, 2, 0, 1], graph.k)
    expected = reference_one_neutral(graph, individual)
    assert_close(student.one_neutral(graph, individual), expected, label="one_neutral graph_2")


def check_one_neutral_graph3(student) -> None:
    graph = load_graph("graphs/graph_3.txt")
    individual = colors_to_bits([0, 1, 2, 3, 1, 2, 3, 0, 1, 2, 3], graph.k)
    expected = reference_one_neutral(graph, individual)
    assert_close(student.one_neutral(graph, individual), expected, label="one_neutral graph_3")


def build_checks() -> List[GradedCheck]:
    return [
        GradedCheck("Q1 eval_balance sample from notebook", 15, check_eval_balance_sample),
        GradedCheck("Q1 eval_balance graph_2 formula", 15, check_eval_balance_graph2),
        GradedCheck("Q1 eval_balance invalid color encoding", 10, check_eval_balance_invalid_encoding),
        GradedCheck("Q1 eval_balance missing color bucket", 10, check_eval_balance_zero_color_bucket),
        GradedCheck("Q3 one_neutral sample from notebook", 20, check_one_neutral_sample),
        GradedCheck("Q3 one_neutral graph_2", 15, check_one_neutral_graph2),
        GradedCheck("Q3 one_neutral graph_3", 15, check_one_neutral_graph3),
    ]


def result_entry(name: str, score: float, max_score: float, output: str) -> dict:
    return {
        "name": name,
        "score": score,
        "max_score": max_score,
        "output": output,
        "visibility": "visible",
    }


def write_results(payload: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    tests = []
    total_score = 0.0
    max_score = 100.0

    try:
        student = importlib.import_module("student_solution")
    except Exception:
        output = "Could not import student_solution.py. Remove unguarded notebook/demo code and make sure the file is named student_solution.py.\n\n"
        output += traceback.format_exc(limit=20)
        payload = {
            "score": 0,
            "visibility": "visible",
            "stdout_visibility": "visible",
            "output": output,
            "tests": [result_entry("Import student_solution.py", 0, max_score, output)],
        }
        write_results(payload)
        print(output)
        return 0

    for check in build_checks():
        try:
            check.func(student)
        except Exception:
            output = traceback.format_exc(limit=8)
            tests.append(result_entry(check.name, 0, check.max_score, output))
            print(f"FAIL: {check.name}\n{output}")
        else:
            tests.append(result_entry(check.name, check.max_score, check.max_score, "Passed"))
            total_score += check.max_score
            print(f"PASS: {check.name}")

    payload = {
        "score": total_score,
        "visibility": "visible",
        "stdout_visibility": "visible",
        "output": f"Project 02 autograder score: {total_score:.1f} / {max_score:.1f}",
        "tests": tests,
    }
    write_results(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
