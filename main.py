import time
from forestLibrary.forest import Forest
from forestLibrary.visual import build_forest_graph
from pygel3d import gl_display as gl

def main():

    total_generations = 200
    delay            = 0.05
    spacing          = 4.0

    # 1) Create viewer once (no display yet)
    viewer = gl.Viewer()

    # 2) Set up your forest
    forest = Forest(size=9, initial_population=0.5, spawn_probability=0.25, species_subset=['honda', 'pine','shrub'], 
                    scenario='temperate')


    # Simulation loop
    for gen in range(total_generations):
        forest.step()                                         # update sim
        g = build_forest_graph(forest, grid_spacing=spacing)  # rebuild graph

        viewer.display(
            g,
            mode='w',           
            smooth=True,
            bg_col=[1, 1, 1],   
            reset_view=False,   
            once=True          
        ) # display graph

        time.sleep(delay) # frame rate 

if __name__ == "__main__":
    main()
