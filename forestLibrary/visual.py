import numpy as np
from pygel3d import graph, gl_display as gl
from forestLibrary.lsystem_utils import realize, swap_verts_array_YZ
from forestLibrary.forest import Forest
from forestLibrary.tree import Tree

def build_forest_graph(forest: Forest, grid_spacing: float = 5.0) -> graph.Graph:
    g = graph.Graph()

    def inner(self, i, j):
        tree = forest.grid[i, j]
        if tree is None:
            return

        verts, edges, _ = realize(tree.lsystem)
        verts = swap_verts_array_YZ(verts)
        #offset = np.array([i * grid_spacing, tree.height_mod, j * grid_spacing], dtype=float)
        offset = np.array([i * grid_spacing, 0.0, j * grid_spacing], dtype=float)
        base   = len(g.nodes())
            
        for v in verts:
            g.add_node(v + offset)
        for child, parent in edges:
            g.connect_nodes(base + child, base + parent)
            
    forest.go_through_forest(inner)

    return g

def build_tree(champions: list[Tree]) -> graph.Graph:
    g = graph.Graph()
    for i, tree in enumerate(champions):
        verts, edges, _ = realize(tree.lsystem)
        base   = len(g.nodes())

        offset = np.array([i*4.0, 0.0, 0.0], dtype=float)
        for v in verts:
            g.add_node(v + offset)
        for child, parent in edges:
            g.connect_nodes(base + child, base + parent)
    return g
