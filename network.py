"""
network.py — Node/Pipe dataclasses, incidence matrix, connectivity check.
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Node:
    id:           str
    label:        str
    node_type:    str            # 'source' | 'junction' | 'offtake'
    elevation:    float = 0.0
    pressure:     float = 0.0   # Pa(a)
    demand:       float = 0.0   # Sm³/s  (positive = consumption)
    min_pressure: float = 0.0   # Pa(a)
    x_coord:      float = 0.0
    y_coord:      float = 0.0

    def is_source(self):   return self.node_type == 'source'
    def is_junction(self): return self.node_type == 'junction'
    def is_offtake(self):  return self.node_type == 'offtake'


@dataclass
class Pipe:
    id:        str
    from_node: str
    to_node:   str
    length:    float
    diameter:  float            # internal [m]
    roughness: float = 7e-6    # PE80 default [m]
    status:    str   = 'open'
    flow:      float = 0.0     # Sm³/s  (set by solver)

    @property
    def area(self): return math.pi * self.diameter**2 / 4


def build_incidence_matrix(nodes: List[Node], pipes: List[Pipe]) -> np.ndarray:
    """A[i,j] = +1 if pipe j leaves node i, -1 if enters, 0 otherwise."""
    node_idx = {n.id: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(pipes)), dtype=float)
    for j, p in enumerate(pipes):
        if p.from_node not in node_idx:
            raise ValueError(f"Pipe {p.id}: from_node '{p.from_node}' not in node list")
        if p.to_node not in node_idx:
            raise ValueError(f"Pipe {p.id}: to_node '{p.to_node}' not in node list")
        A[node_idx[p.from_node], j] = +1.0
        A[node_idx[p.to_node],   j] = -1.0
    return A


def check_connectivity(nodes: List[Node], pipes: List[Pipe]) -> Dict:
    """BFS from all source nodes.  Handles pipes referencing unknown nodes gracefully."""
    valid_ids = {n.id for n in nodes}
    adj: Dict[str, List[str]] = {n.id: [] for n in nodes}

    for pipe in pipes:
        if pipe.status != 'open':
            continue
        # Only add edge if both nodes are known
        if pipe.from_node in valid_ids and pipe.to_node in valid_ids:
            adj[pipe.from_node].append(pipe.to_node)
            adj[pipe.to_node].append(pipe.from_node)

    sources  = {n.id for n in nodes if n.is_source()}
    visited  = set(sources)
    queue    = list(sources)
    while queue:
        cur = queue.pop(0)
        for nb in adj.get(cur, []):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    isolated = valid_ids - visited
    return {'connected': len(isolated) == 0,
            'reachable': visited, 'isolated': isolated}


def partition_nodes(nodes: List[Node]) -> Tuple[List[int], List[int]]:
    src = [i for i, n in enumerate(nodes) if n.is_source()]
    unk = [i for i, n in enumerate(nodes) if not n.is_source()]
    return src, unk


def straight_line_distance(n1: Node, n2: Node) -> float:
    return math.sqrt((n1.x_coord-n2.x_coord)**2 + (n1.y_coord-n2.y_coord)**2)
