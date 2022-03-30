# Explore Hichem's code

# Generate some data

import igraph
from moo.data_generation import ExpConfig, DataGenerator
from moo.data_generation import ExpConfig, DataGenerator
from moo.contestant import get_best_community_solutions, draw_best_community_solutions
import moo.contestant as contestant
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import itertools

import pandas as pd


# Define an experiment configuration instance (parameters for data generation)
expconfig = ExpConfig(
    L=100, U=500,
    NumEdges=1000, ML=0.4, MU=0.4,
    BC=0.1, NumGraphs=5,
    shuffle=True, # Shuffle labels (or no)
    seed=None # For reproducibility (this is the default, but can be changed)
    )
print(expconfig) # Print parameters, or access individually, e.g., expconfig.NumEdges

expgen = DataGenerator(expconfig=expconfig) # Pass defined parameters
print(expgen)
datagenMaster = expgen.generate_data() # datagen is an iterator

datagen, datagenJoblib = itertools.tee(datagenMaster)


algos = [
    contestant.ComDetMultiLevel(), # Multi-Level approach
    contestant.ComDetEdgeBetweenness(), # EdgeBetweenness approach
    contestant.ComDetWalkTrap(), # WalkTrap approach
    contestant.ComDetFastGreedy(), # FastGreedy approach
    
]



combinedList = itertools.product(enumerate(datagenJoblib), algos)

def runalgo(c):
    # Extract elements we need
    ig, algo = c
    i, g  = ig
    #print(i, algo)
    result = algo.detect_communities(graph=g).get_results()
    for r in result: 
        r['graph_idx'] = i + 1

    return(r)


print("Parallel")
joblibresults = Parallel(n_jobs = 7) (delayed(runalgo)(c) for c in combinedList)
print("Done parallel")

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
