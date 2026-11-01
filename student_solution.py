import os, random
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class AdjNode:
    def __init__(self, vertex):
        self.vertex = vertex
        self.next = None

@dataclass
class Graph:
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self.graph = [None] * n

    def add_edge(self, u: int, v: int):
        # Add v to u's adjacency list
        new_node = AdjNode(v)
        new_node.next = self.graph[u]
        self.graph[u] = new_node

        # Add u to v's adjacency list
        new_node = AdjNode(u)
        new_node.next = self.graph[v]
        self.graph[v] = new_node


def init_graph(fp):
    lines = []
    with open(fp, "r", encoding="utf-8") as f:
        for l in f:
            lines.append(l)

    n = int(lines[0])
    k = int(lines[1])

    g = Graph(n, k)

    for line in lines[2:]:
        u, v = line.split(" ")
        g.add_edge(int(u), int(v))

    return g

def unique_edges(graph):
    edges: List[Tuple[int, int]] = []
    for src in range(graph.n):
        node = graph.graph[src]
        while node is not None:
            dest = node.vertex
            if src < dest:
                edges.append((src, dest))
            node = node.next
    return edges

# Use to check how many of each color for fitness
def get_dcolors(graph):
    width = get_color_len(graph)
    combos = []
    for d in range(graph.k):
        combos.append(d)

    return combos

def get_color_cts(dcol, dmap):
    ccts = [0] * len(dcol)
    for d in dmap:
        ccts[d] += 1

    return ccts

# Retrieve bit count for color
def get_color_len(graph):
    b_l = max(1, (graph.k - 1).bit_length())
    return b_l

# Returns dict with node-color mapping
def decode_individual(graph, individual):
    width = get_color_len(graph)

    if len(individual) != graph.n * width:
        return None

    dmap = []
    for node in range(graph.n):
        start = node * width

        # Extract color bits
        bits = individual[start:start + width]

        color = 0

        for bit in bits:
            if bit not in (0, 1):
                return None

            color = (color << 1) | bit

        # Reject invalid color values
        if color >= graph.k:
            return None

        dmap.append(color)

    return dmap

def eval_balance(graph, individual):

    # The balanced-coloring fitness is:
    # F(C(G)) = (sum over edges delta(i, j) / m) * product over colors c of (count(c) / n)

    dmap = decode_individual(graph, individual)
    if dmap is None:
        return (0.0,)

    satisfied = 0

    edges = unique_edges(graph)

    for u, v in edges:
        if dmap[u] != dmap[v]:
            satisfied += 1

    total_edges = len(edges)

    if total_edges == 0:
        return (1.0,)

    dcol = get_dcolors(graph)
    ccts = get_color_cts(dcol, dmap)
    if 0 in ccts:
        return (0.0,)

    prod = 1.0
    for c in ccts:
        prod *= (c / graph.n)

    fitness = satisfied / total_edges * prod
    return (fitness,)

def eval_graph(graph, individual):
    dmap = decode_individual(graph, individual)

    if dmap is None:
        return (0.0,)

    satisfied = 0

    edges = unique_edges(graph)

    for u, v in edges:
        if dmap[u] != dmap[v]:
            satisfied += 1

    total_edges = len(edges)

    if total_edges == 0:
        return (1.0,)

    fitness = satisfied / total_edges

    return (fitness,)

def one_neutral(graph, individual):
    dmap = decode_individual(graph, individual)

    if dmap is None:
        return 0.0

    # Compute original fitness
    og_fitness = eval_graph(graph, individual)[0]

    neutral_ct = 0
    total_muts = graph.n * (graph.k - 1)

    if total_muts == 0:
        return 0.0

    # Number of bits representing each color
    width = max(1, (graph.k - 1).bit_length())

    # Try changing one vertex at a time
    for d in range(graph.n):

        og_c = dmap[d]

        # Try every valid color
        for n_c in range(graph.k):

            # Skip the vertex's existing color
            if n_c == og_c:
                continue

            # Make a copy of original individual
            mutant = individual.copy()

            # Convert the new color into binary bits
            binary = format(n_c, f"0{width}b")
            n_bits = [int(bit) for bit in binary]

            start = d * width
            mutant[start:start + width] = n_bits

            m_fitness = eval_graph(graph, mutant)[0]

            if m_fitness == og_fitness:
                neutral_ct += 1

    return neutral_ct / total_muts

def run_ga(fp, population_size=300, ngen=200, cxpb=0.8, mutpb=0.3):
    from deap import algorithms, base, creator, tools

    graph = init_graph(fp)
    width = get_color_len(graph)

    if not hasattr(creator, "GraphColorFitnessMax"):
        creator.create("GraphColorFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "GraphColorIndividual"):
        creator.create("GraphColorIndividual", list, fitness=creator.GraphColorFitnessMax)

    def make_individual():
        genome = []
        for _ in range(graph.n):
            genome.extend(encode_color(random.randrange(graph.k), width))
        return creator.GraphColorIndividual(genome)

    def encode_color(color, width):
        return [int(bit)for bit in format(color, f"0{width}b")]

    def mate_by_node(ind1, ind2):
        # Swap complete color genes
        for node in range(graph.n):
            if random.random() < 0.5:
                start = node * width
                end = start + width
                ind1[start:end], ind2[start:end] = ind2[start:end], ind1[start:end]
        return ind1, ind2

    def mutate_color(individual):
        # Mutate one node per selected individual
        for node in range(graph.n):
            if random.random() < 1 / graph.n:
                start = node * width
                individual[start:start + width] = encode_color(random.randrange(graph.k), width)
        return (individual,)

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_graph, graph)
    toolbox.register("mate", mate_by_node)
    toolbox.register("mutate", mutate_color)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    best_ever = tools.HallOfFame(1)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=best_ever, verbose=False,)

    best_individual = best_ever[0]
    print(f"Best individual: {best_individual}")
    coloring = decode_individual(graph, best_individual)
    fitness = eval_graph(graph, best_individual)
    bc_fitness = eval_balance(graph, best_individual)
    os_neut = one_neutral(graph, best_individual)

    print(f"Color assignment: {coloring}")
    print(f"Balanced-color fitness: {bc_fitness[0]}")
    print(f"One-step neutrality: {os_neut}")

def main():
    gdir = Path("graphs")
    gpath = os.getcwd() / gdir
    for f in gpath.iterdir():
        run_ga(f)

if __name__ == "__main__":
    main()
