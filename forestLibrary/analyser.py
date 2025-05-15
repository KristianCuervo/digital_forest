from .forest import Forest
from .tree import Tree
from operator import attrgetter
from collections import Counter


class Analyse:
    def __init__(self, forest: Forest):
        self.forest = forest

    def population_distribution(self) -> dict[str, int]:
        """
        Returns a dict mapping species → number of currently living trees.
        """
        # Refresh the internal gene_pools so it reflects the current grid
        self.forest.update_gene_pools()

        # gene_pools is a species→[Tree, …] of living trees
        return { species: len(pool)
                 for species, pool in self.forest.gene_pools.items() }

    def reason_of_death(self) -> dict[str, dict[str, int]]:
        """
        Returns a nested dict:
            species → { 'Age': n1, 'Shadow': n2, 'Size': n3 }
        counting why each tree died.
        """
        result: dict[str, dict[str,int]] = {}
        for species, dead_list in self.forest.graveyard.items():
            # tally up the death_reason attribute
            counts = Counter(t.death_reason for t in dead_list)
            result[species] = {
                'Age':    counts.get('Age', 0),
                'Shadow': counts.get('Shadow', 0),
                'Size':   counts.get('Size', 0),
            }
        return result
                

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def plot_forest_grid(forest, save_path="forest.png"):
    """
    Plots the forest grid as a 2D colored map.
    Each tile is colored based on the tree species: oak, birch, pine, shrub.
    Empty tiles (None) are white.
    Oak and birch have similar greens.
    
    In headless environments, optionally saves the figure to disk instead of displaying.
    
    Parameters:
    -----------
    grid : 2D numpy.ndarray
        Array of tree instances or None. Each tree must have `genes['species']` attribute.
    save_path : str or None
        File path to save the figure (e.g., 'forest.png'). If None, returns the figure object.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure (if save_path is None)
    """
    species_colors = {
        'oak': '#228B22',    # forest green
        'birch': '#2E8B57',  # sea green (similar to oak)
        'pine': '#8B4513',   # saddle brown
        'shrub': '#ADFF2F'   # green yellow
    }
    empty_color = '#FFFFFF'  # white for empty tiles
    water_color = '#0000FF'  # blue for water (if applicable)
    grid = forest.grid
    n, m = grid.shape
    img = np.zeros((n, m, 4))
    for i in range(1, n):
        for j in range(1, m):
            tree = grid[i, j]
            if tree is None and forest.noise_grid[i, j] < 0:
                color = water_color
                print("water")
            elif tree is None and forest.noise_grid[i, j] >= 0:
                color = empty_color
            else:
                species = tree.genes.get('species')
                color = species_colors.get(species, '#CCCCCC')
            img[i, j] = mcolors.to_rgba(color)

    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    print("something")

    patches = [
        mpatches.Patch(color=color, label=species.capitalize())
        for species, color in species_colors.items()
    ]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig