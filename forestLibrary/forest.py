import numpy as np
from numpy import ndindex
from .tree import Tree
from .noise import Noise
import random as random
from .geneticAlgorithm import GeneticAlgorithm
from .species_genes import SPECIES_DEFAULT_PARAMS, reduced_SPECIES, get_species_params
from .species import HondaTree, ShrubTree, PineTree

from .graveyard import Graveyard

SPECIES_CLASS = {
    "honda"      : HondaTree,
    "oak"        : HondaTree,
    "birch"      : HondaTree,
    "shrub"      : ShrubTree,
    "pine"       : PineTree
}



class Forest:
    def __init__(self, size:int, initial_population:float=0.5, spawn_probability:float=0.15, species_subset: list[str] | None = None, scenario:str="polar"):
        seed = 12345
        self.rnd = np.random.default_rng(seed)   
        # Forest is a grid of trees with boundaries of None values
        self.gen = 0

        self.scenario = scenario
        print("The climate zone is", self.scenario)
        
        self.size = size # Tree Size
        self.grid = np.empty((size+2, size+2), dtype=object)

        self.noise = Noise(sigma=3, N=90, seed=seed) # Noise is used to create a height map for the trees
        noise_vertical_scale = 100
        vertical_offset = 0.05
        self.noise_grid = noise_vertical_scale * (self.noise.compute_turbulence_grid(0.01, size+2, 3) + vertical_offset)
        #self.noise_grid = np.ones((self.size+2, self.size+2))
        #print(self.noise_grid)

        # Forest is spawned on grid with random tree species
        self.initial_population = initial_population
        self.active_species = species_subset or list(SPECIES_CLASS.keys())
        self.initial_spawn()

        # Sunlight grid is a grid of sunlight values for each tree at their current growth
        self.sunlight_grid = None
        self.get_gene_pools = None
        self.shadow_kernel = np.array([ [0.05, 0.1, 0.05],
                                        [0.1,  0,   0.1],
                                        [0.05, 0.1, 0.05]])

        # Spawn probability is the probability of spawning a new tree in an empty cell
        # Genetic algorithm is used to create new trees from the gene pools
        self.spawn_probability = spawn_probability
        
        self.genetic_algorithm = GeneticAlgorithm(mutation_rate=0.1, mutation_strength=0.25)
        
        # Collectors for data analysis and gene graphs
        self.graveyard = {s: [] for s in list(SPECIES_CLASS.keys())} # all dead trees
        self.champions = {s: [] for s in list(SPECIES_CLASS.keys())} # most fit trees every few generations


        
        self.death_pool = np.empty((size+2, size+2), dtype=object)
    
    def initial_spawn(self):
        """
        Randomly spawns trees in the forest. 
        It should be investigated whether the trees should also start at
        age 1 or at random ages.
        """

        def inner(self, i, j):
            # 1 --> self.size+1 are the non-boundary cells
            
            if self.noise_grid[i,j] >= 0 and self.rnd.random() < self.initial_population:
                # Given a wanted population probability distribution, spawn random trees
                species_name = random.choice(self.active_species)
                genes = get_species_params(species_name, param_dict=reduced_SPECIES)
                self.grid[i, j] = SPECIES_CLASS[species_name](height_mod=self.noise_grid[i,j], genes=genes)
        self.go_through_forest(inner)

    def update_sunlight(self):
        """
        Returns a grid of sunlight values for each tree in the forest.
        The sunlight value is determined by the distance to the nearest tree
        and the angle of the sun. This is required to calculate the shadow.
        """
        sunlight_grid = np.zeros((self.size+2, self.size+2))
        def inner(self, i, j):
            if self.grid[i, j] is not None:
                    sunlight_grid[i, j] = self.grid[i, j].sunlight

        self.go_through_forest(inner)

        self.sunlight_grid =  sunlight_grid

    def update_shadows(self):
        """
        Returns total shadow cast at tree (x, y) by neighbouring trees
        in a 3x3 grid.
        A kernel multiplied by the sunlight of the neighbours gives an 
        approximation of the shadow cast by the neighbours.
        """
        self.update_sunlight()

        def inner(self, i, j):
            if self.grid[i, j] is not None:
                self.grid[i, j].shadow = self.get_shadow(i,j)
        
        self.go_through_forest(inner)
    
    def get_shadow(self, i, j):
        return np.sum(self.shadow_kernel*self.sunlight_grid[i-1:i+2, j-1:j+2])
    
    def death_or_growth(self):
        """
        Determines whether each tree in the forest survives or dies.
        The trees that die are removed from the forest.
        """

        def inner(self, i, j):
            if self.grid[i, j] is not None:
                if self.grid[i, j].survival_roll(simulation_year=self.gen, scenario=self.scenario) == False:
                    self.graveyard[self.grid[i,j].genes['species']].append(self.grid[i,j]) # Collects the tree instance in the graveyard
                    self.death_pool[i, j] = self.grid[i, j] # adds the final tree state before its death to a pool for rendering
                    self.grid[i, j] = None # Kills the tree instance
                else:
                    self.grid[i, j].grow(climate_zone=self.scenario)
        self.go_through_forest(inner)


    def update_gene_pools(self):
        """
        Returns seperate lists of the instances of each species in the forest.
        """
        gene_pools = {}
        def inner(self, i, j):
            if self.grid[i, j] is not None:
                species = self.grid[i, j].genes['species'] # this is a string of the species name 
                if species not in gene_pools:
                    gene_pools[species] = [] # creates a new list for the species
                gene_pools[species].append(self.grid[i, j]) # appends tree instance to list   
             
        self.go_through_forest(inner)

        self.gene_pools = gene_pools
    
    def spawn_new_trees(self):
        """
        Spawns new trees in the forest based on the gene pools and genetic algorithm.
        The trees are spawned in empty cells with adaptive reproduciton.
        """
        k_reproductive_rate = 0.4 # Constant which affects the dynamical system. This could be changed into a tree property.
        self.update_gene_pools()

        for species, gene_pool in self.gene_pools.items():
            if len(gene_pool) >= 2: # Needs two parents to breed
                n_offspring = int(np.ceil(k_reproductive_rate * len(gene_pool)))
                children = self.genetic_algorithm.generate_children(gene_pool, n_offspring) # Creates n amount of children from the gene_pool
                for child in children:
                    empty_cells = np.argwhere(self.grid[1:-1, 1:-1] == None) # Look at current empty cells
                    if len(empty_cells) > 0: # 
                        cell = random.choice(empty_cells) # Picks a random cell to spawn on 
                        self.grid[cell[0], cell[1]] = SPECIES_CLASS[species](height_mod=self.noise_grid[cell[0],cell[1]], genes=child) # Spawns a child tree on tile
                    else:
                        break # No empty cells to spawn a tree
            
            elif len(gene_pool) < 2:
                print(f"Not enough trees to breed {species} trees. Only {len(gene_pool)} trees available.")

    
    def record_champions(self):
        """
        Every 25 generations, pick the fittest mature tree of each species
        and store it in self.champions[species] as (gen, tree).
        """
        # make sure gene_pools is up to date
        self.update_gene_pools()

        for species, pool in self.gene_pools.items():

            # filter only mature trees
            mature_trees = [
                t for t in pool
                if hasattr(t, 'age') 
                   and t.age >= 10
            ]
            if not mature_trees:
                # no candidates this time
                continue

            from operator import attrgetter

            # pick the one with highest fitness()
            best = max(mature_trees, key=attrgetter('fitness'))
            # store a snapshot (generation, tree)
            self.champions[species].append((self.gen, best))


    def reached_termination(self, i, j):
        tree_in_terminal_state = self.death_pool[i,j]
        if tree_in_terminal_state is not None:
            self.death_pool[i,j] = None
            return tree_in_terminal_state
        else:
            return None

          
    def step(self):
        """
        Runs one step of the simulation.
        """
        self.update_shadows()
        self.death_or_growth()
        self.spawn_new_trees()

        if self.gen % 10 == 0:
            self.record_champions()
        self.gen += 1

    
    def go_through_forest(self, func):
        for i, j in ndindex(self.size, self.size):
            func(self, i+1, j+1)
