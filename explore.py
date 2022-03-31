# Explore Hichem's code

# Generate some data

import igraph
from moo.data_generation import ExpConfig, DataGenerator
from moo.data_generation import ExpConfig, DataGenerator
from moo.contestant import get_best_community_solutions, draw_best_community_solutions
import moo.contestant as contestant
from moo.communities import run_parallel_communities


import matplotlib.pyplot as plt


import pandas as pd


# Define an experiment configuration instance (parameters for data generation)
expconfig = ExpConfig(
    L=100, U=500,
    NumEdges=1000, ML=0.4, MU=0.4,
    BC=0.1, NumGraphs=30,
    shuffle=True, # Shuffle labels (or no)
    seed=1234  # For reproducibility (this is the default, but can be changed)
    )
print(expconfig) # Print parameters, or access individually, e.g., expconfig.NumEdges

expgen = DataGenerator(expconfig=expconfig) # Pass defined parameters


# Copy the iterator so we can use it for serial and parallel 
datagen = expgen.generate_data() # datagen is an iterator
datagenJoblib = expgen.generate_data() # datagen is an iterator


algos = [
    contestant.ComDetMultiLevel(), # Multi-Level approach
    contestant.ComDetEdgeBetweenness(), # EdgeBetweenness approach
    contestant.ComDetWalkTrap(), # WalkTrap approach
    contestant.ComDetFastGreedy(), # FastGreedy approach
]


# graphlist = list(datagenMaster)
# graphlist2 = list(datagenMaster2)

# Quick check - are the number of vertices in matching pairs the same
# for gp in zip(graphlist, graphlist2):
#     g0count = gp[0].vcount()
#     g1count = gp[1].vcount()
#     print(g0count, g1count)


# quit()


# Run in paralllel
joblibresults = run_parallel_communities(datagenJoblib, algos, n_jobs=7)


df_joblibresults = pd.DataFrame(joblibresults) 
print(df_joblibresults.shape)
df_joblibresults.head()

df_joblibresults.to_excel("joblib.xlsx")

results = [] # Holds results of contestants
for g_idx, graph in enumerate(datagen):
    print(f'Processing Graph {g_idx+1}')
    for algo in algos:
        print(f'\tUsing algoithm {algo.name_}')
        result = algo.detect_communities(graph=graph).get_results()
        # Result is a list of dictionaries, each dictionary stores the metrics of one iteration (see code for details)
        for r in result: # Appending graph index to results, for debugging purposes
            r['graph_idx'] = g_idx + 1
        results.extend(result)


# Optional: Convert results into a dataframe (to use pandas capabilities)
df_contestants = pd.DataFrame(results) # Column names are inferred from the dictionaries' keys
print(df_contestants.shape)
df_contestants.head()

df_contestants.to_excel("series.xlsx")


# Looks like multilevel isn't reproducible, but the others are?  Check this

