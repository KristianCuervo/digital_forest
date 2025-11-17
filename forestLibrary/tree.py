import numpy as np
from .lsystem_utils import realize


class Tree:
    def __init__(self, genes, height_mod, axiom=None):
        self.genes   = genes
        self.height_mod = 10*height_mod
        self.lsystem = axiom if axiom else [('A', 1.0, 0.2)]
        self.age     = 1

        # geometry & ecology state (updated after each grow)
        self.height  = 0.0
        self.width   = 0.0
        self.sunlight = 0.0
        self.shadow   = 0.0
        self.survival_requirement = 0.0

        # compute initial geometry + sunlight
        self._update_geometry()
        #self.sunlight = self.sunlight_intake()
        self.fitness = 0.0


        # ---- DEATH INFO -----
        self.year_of_death = None
        self.death_reason = None
    
     # ----------------- L-SYSTEM  -----------------

    def __repr__(self):
        return self.genes['species']
    
    def production_rule(self, sym):
        """Must be overridden by subclasses."""
        return [sym]             # default: no rewrite
    
    def grow(self, climate_zone):
        if self.age <= 10: 
            grown_lsystem = []
            for sym in self.lsystem:
                grown_lsystem += self.production_rule(sym)
            self.lsystem = grown_lsystem
            self._update_geometry()
            self.sunlight = self.sunlight_intake(scenario=climate_zone)
        self.age += 1      # Age still increases if it stops growing

    def excess_sunlight(self):
        if self.sunlight != self.shadow:
            return self.sunlight / self.shadow
        else:
            return 0
    
    def _update_geometry(self):
        verts, edges, radii = realize(self.lsystem)
        if verts.size == 0:
            self.height = self.width = 0.0
            return

        # Y is the vertical axis in SpaceTurtle
        y_vals = verts[:, 1]
        self.height = float(y_vals.max() - y_vals.min())

        # Width = max horizontal spread in X–Z plane
        x_vals = verts[:, 0]
        z_vals = verts[:, 2]
        x_span = x_vals.max() - x_vals.min()
        z_span = z_vals.max() - z_vals.min()
        self.width = float(max(x_span, z_span))

        # Y is the vertical axis
        y_vals = verts[:,1]
        self.height = float(y_vals.max() - y_vals.min())

        # width in the X‑Z plane
        x_vals = verts[:,0]
        z_vals = verts[:,2]
        self.width = float(max(x_vals.max()-x_vals.min(),
                            z_vals.max()-z_vals.min()))

    # ----------------- FITNESS  -----------------
    def sunlight_intake(self, scenario):
        """
        Fitness function of each of the trees.
        S(h, w) = alpha*h + beta*w + gamma*sqrt(h*w)
        where h is the height and w is the width of the tree. 
        alpha --> tall factor
        beta --> wide factor
        gamma --> square factor
        """
        alpha = None
        beta = None
        gamma = None
        if scenario == "tropical":
            # The light comes from high up in the summer, penetrates far into the forest and benefits wide trees. 
            alpha = 0.5
            beta = 1.75
            gamma = 1.0

        elif scenario == "temperate":
            # Intermediate lighting condition. Doesn't prioritise tall nor wide trees
            alpha = 1.5
            beta =  1.0
            gamma = 1.75
            pass

        elif scenario == "polar":
            # The light comes from a shallow angle in winter. It is better to be taller at this point.
            alpha = 2.5
            beta = 0.5
            gamma = 0.5
            pass
        
        return alpha*(self.height + self.height_mod) + beta*self.width + gamma*np.sqrt((np.abs(self.height + self.height_mod))*self.width)
    

    # ----------------- DEATH ROLLS -----------------
    def old_age_death_roll(self):
        """
        The tree dies with an increasing probability as it ages.
        """

        chance_of_death = (self.age / 100)

        if np.random.rand() < chance_of_death:
            return True
        return False
    
    def survival_roll(self, simulation_year, scenario):
        """
        The tree dies if it does not meet the survival requirements.
        The larger a tree is, the more sunlight it needs to survive. 
        """
        # Parameters for the survival roll
        if scenario == "polar":
            a = 0.5
            b = 0.6
        
        if scenario == "temperate":
            a = 0.4
            b = 0.4
        
        if scenario == "tropical":
            a = 0.4
            b = 0.3
        
        effective_size = (self.height * (self.width)**2)
        
        self.survival_requirement = a*self.shadow + b*effective_size
        self.fitness = self.sunlight - self.survival_requirement

        
        # The tree dies if it does not get enough sunlight
        if self.age >= 1:
            if self.fitness < 0:
                self.year_of_death = simulation_year
                if self.shadow > self.sunlight:
                    self.death_reason = "Shadow"
                else:
                    self.death_reason = "Size"
                return False
            
            # If the tree is not dead, check if it dies from old age
            if self.old_age_death_roll():
                self.year_of_death = simulation_year
                self.death_reason = "Age"
                return False
        
        # Tree survives
        return True




