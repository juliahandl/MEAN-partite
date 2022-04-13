import os
import sys
# workaround of relative/absolute imports when importing from within the package and from external files
dirname = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.realpath(os.path.join(dirname, '..'))
sys.path.insert(0, module_path)

from random import seed
import numpy as np
from sklearn.metrics import adjusted_rand_score
import igraph
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.indicators.hv import Hypervolume
from moo.contestant import CommunityDetector
import sknetwork
import cdlib
import skbio

class MultiCriteriaProblem(ElementwiseProblem):
    """
    Specializes a pymoo problem
    """
    def __init__(self, mode, graph):
        
        # Problem-specific arguments: bipartite graph
        assert isinstance(graph, igraph.Graph), "graph must be of type igraph.Graph"
        assert graph.is_bipartite(return_types=False), "graph must be a bipartite graph"
        assert graph.is_connected(), "graph must be fully connected (one connected component)"
        assert len(graph.vs), "graph must not be empty"
        self.graph_ = graph

        assert mode == "3d" or mode == "2d", "mode needs to be either '3d' or '2d'"
        self.mode_ = mode # 3d or 2d (see paper)
                
        # Base class arguments: inferred from the graph or explicitly from the moo problem type (2d, 3d)
        # num design variables, num objectives, num constraints, lower/upper bounds for design variables
        self.n_var_ = len(self.graph_.vs) # Number of design variables (vertices)
        self.n_obj_ = 3 if self.mode_=="3d" else 2  # Number of objectives
        self.n_constr_ = 0 # Number of constraints (no constraints)
        self.xl_ = np.zeros(self.n_var_) # Lower bound for design variables (0)
        self.xu_ = self.graph_.indegree()  # Upper bound for design variables
        # It determines the number of possible edges for mutation (upper bound)
        # from the adjacency list)

        self.vertices_ = graph.vs['VX']
        self.groundtruth_ = graph.vs['GT']
        self.proj0_ = [i for i,val in enumerate(self.vertices_) if val==0] # Vertex indices (1st mode)
        self.proj1_ = [i for i,val in enumerate(self.vertices_) if val==1] # Vertex indices (2nd mode)
   
        
        # type_Var, # (optional) A type hint for the user what variable should be optimized.
        
        # Other probleme specific computed parameters
        self.adj_list_ = self.graph_.get_adjlist() # Adjacency list
        self.graph_proj1_, self.graph_proj2_ = self.graph_.bipartite_projection(multiplicity=True) # Graph projecttion into two one-mode graphs
        # self.binary_links = np.full(self.n_var_, -1)

        super().__init__(n_var = self.n_var_,
                         n_obj = self.n_obj_,
                         n_constr = self.n_constr_,
                         xl = self.xl_, # Lower/Upper bounds for decision variables(using node degrees)
                         xu = self.xu_,
                         )

    def _evaluate(self, x, out, *args, **kwargs): # Mutivariate fitness function
        
        sol_edges = []
        # Allow self loops in encoding which are interpreted as "no edges"
        for i in range(0, self.n_var_):
            if x[i] > 0:
                sol_edges.append([i,self.adj_list_[i][x[i]-1]]) # x are the solutions we are looking for
         
        # Construct bipartite graph
        t = igraph.Graph.Bipartite(self.graph_.vs['VX'], sol_edges)
        
        # Decoding: Identify unconnected components to define communities
        c = t.clusters()
        m = c.membership
        num_clusters = max(c.membership) + 1

        if self.mode_ == "3d":
            proj0_labels=[m[i] for i in self.proj0_] # Community memberships for the 1st two-mode projected graph
            proj1_labels=[m[i] for i in self.proj1_] # Community memberships for the 2nd two-mode projected graph
            # Evaluate both projections with respect to those communities
            modularity_score_1 = self.graph_proj1_.modularity(proj0_labels, weights=self.graph_proj1_.es['weight'])
            modularity_score_2 = self.graph_proj2_.modularity(proj1_labels, weights=self.graph_proj2_.es['weight'])
            out["F"] = [-modularity_score_1, -modularity_score_2, num_clusters]
        elif self.mode_ == "2d":
            modularity_score = self.graph_.modularity(m)
            out["F"] = [-modularity_score, num_clusters]
        
    def __str__(self) -> str:
        return f"<MultiCriteriaProblem: mode='{self.mode_}', graph: (V={len(self.graph_.vs)}, "\
             f"E={len(self.graph_.es)}), n_var:{self.n_var_}, n_obj:{self.n_obj_}, "\
                 f"n_constr{self.n_constr_}>"

 # mode, GA population size and a pymoo termination criterion
  # Not used
class ComDetMultiCriteria(CommunityDetector):
    def __init__(
        self, name="multicriteria",
        params={'mode': '3d', 'popsize': 50, 'termination': None, 'save_history': True, 'seed': None},
        min_num_clusters=1, max_num_clusters=15
        ):

        self.name_ = name
        
        super().__init__(self.name_)
        self.params_ = params
                
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here 

    def detect_communities(self, graph, y=None):
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self

    def __detect_communitites(self):
        # Actual community detection code. The steps are the following:
        # 1. Initialize the problem
        # 2. Initialize the GA population (required for the algorithm definition)
        # 3. Define (implement) the algorithm
        # 4. Define termination criteria
        # 5. Optimize (needs the problem, the algorithm (including the initial population),
        # and the termination)
        
        # print('Initializing the problem...', end='')
        self.init_problem()
        # print('Done')
        
        # print('Initializing the population...', end='')
        self.initialize_pop()
        # print('Done')
        
        # print('Defining the algorithm...', end='')
        self.define_algo()
        # print('Done')
        
        # print('Defining the termination criterion...', end='')
        self.define_termination()
        # print('Done')
        
        # print('Optimizing...', end='')
        self.optimize()
        # print('Done')
        
        # print('Collating results...', end='')
        self.collate_results()
        # print('Done with all steps')

    def recursive_links(self, node_index, node_origin, x, temp_edges, binary_links, a, mst):
        # print(node_index,node_origin)
        # Draw link 
        if x[node_index] != -1:
            #print("I have already been here - going backwards")
            return

        # Set link to originating node - # Decode corresponding to full adjacency matrix
        for j in range(0,len(a[node_index])):        
            if a[node_index][j] == node_origin:
                x[node_index] = j+1
                binary_links[node_index] = node_origin
                temp_edges.append([node_index,node_origin]) # translation from node indices to indices in the adj list
                    
        # Start recursion on all neighbors in MST
        for i in range(0,len(mst[node_index])):
            self.recursive_links(mst[node_index][i],node_index, x, temp_edges, binary_links, a, mst)
        
        return

    def init_problem(self):
        self.problem_ = MultiCriteriaProblem(mode=self.params_['mode'], graph=self.graph_)
        
    def initialize_pop(self):
        popsize = self.params_['popsize']
        n_var = self.problem_.n_var_
        binary_links = np.full(n_var, -1)

        # The initial generation of individuals is built by computing the MST
        # of the graph, then introducnig some diversity
        # 1. Initial individual for the Evolutionary ALgorithm (based on the MST)
        t = self.graph_.spanning_tree(weights = self.graph_.edge_betweenness()) # MST as a graph
        mst = t.get_adjlist() # Adjacency list for the MST

        x = list(np.full(n_var, -1)) # Solution to evaluate

        adj_list = self.problem_.adj_list_ # Adjacency list of the original graph

        temp_edges = []
        for i in range(0,n_var):
            if x[i] == -1:
                x[i] = 0
                for j in range(0,len(mst[i])):
                    self.recursive_links(mst[i][j],i, x, temp_edges, binary_links, adj_list, mst) # translate MST into a genome
        
        # 2. Duplicate the individual to make a poulation      
        pop = np.tile(x, (popsize, 1)) # (identical genomes/solutions)
        c = self.graph_.clusters()
        ctr = 0

        # 3. Diversity in the initial generation
        for i in range(len(c),1+min(50,popsize,n_var)): #?

            # Use different greedy solutions for diversity
            test = self.graph_.community_fastgreedy()
            test = test.as_clustering(i)

            test=test.membership
            for j in range(0,n_var):
                if pop[ctr][j] != 0 and test[adj_list[j][pop[ctr][j]-1]] != test[j]: # Removing edges crossing communities
                    pop[ctr][j] = 0
            ctr = ctr +1
       
        self.pop_ = pop # Initial generation

    def define_algo(self):
        self.algorithm_ = NSGA2(
            pop_size=self.params_['popsize'],
            n_offsprings=self.params_['popsize'],
            sampling=self.pop_,
            crossover=get_crossover("int_ux", prob=0.3), # HParams to test
            mutation=get_mutation("int_pm"), # HParams to test
            eliminate_duplicates=True,
        )
        # Popsize, number of generations more important that the above HParams
        # Check hypervolume for convergence test to avoid long running times (early stopping)
    
    def define_termination(self):
        # Define termination here. For now, it is passed in params but can be changed in the future
        termination = self.params_['termination']
        self.termination_ = termination if termination is not None else get_termination("n_gen", 1000)
        # print(self.termination_)

    def optimize(self):
        # Finally, we are solving the problem with the algorithm 
        # and termination we have defined
        self.res_ = minimize(
            self.problem_,
            self.algorithm_,
            self.termination_,
            seed=self.params_['seed'],
            save_history=self.params_['save_history'],
            verbose=False, # True
        )

    def collate_results(self):
        # Collate results and eliminate duplicates
        X = self.res_.X
        n_var = self.problem_.n_var_
        vertices = self.problem_.vertices_
        groundtruth = self.problem_.groundtruth_
        adj_list = self.problem_.adj_list_
        proj0 = self.problem_.proj0_
        proj1 = self.problem_.proj1_

        temp_results = [] # Before removing duplicates
        badj = make_badj(self.graph_)
        for n in range(0,len(X)):
            sol_edges = []
            for i in range(0,n_var):
                if X[n][i] > 0:
                    sol_edges.append([i,adj_list[i][X[n][i]-1]])
            t = igraph.Graph.Bipartite(vertices,sol_edges)
            # igraph.summary(t)
            c= t.clusters()

            m = c.membership
            modularity_score = self.graph_.modularity(m)
            adj_rand_index = adjusted_rand_score(groundtruth,m)
            
            proj0_labels=[m[i] for i in proj0] # Community memberships for the 1st two-mode projected graph
            proj1_labels=[m[i] for i in proj1] # Community memberships for the 2nd two-mode projected graph
            modularity_score_barber = sknetwork.clustering.bimodularity(badj,proj0_labels,proj1_labels)
            modularity_score_murata = modularity_murata(badj,proj0_labels+proj1_labels)
            modularity_score_1 = self.problem_.graph_proj1_.modularity(proj0_labels,weights=self.problem_.graph_proj1_.es['weight'])#
            modularity_score_2 = self.problem_.graph_proj2_.modularity(proj1_labels,weights=self.problem_.graph_proj2_.es['weight'])

            communities = [[] for i in range(max(proj0_labels+proj1_labels)+1)] ## List of list of node ids.
            for i,lab in enumerate(proj0_labels+proj1_labels):
                communities[lab].append(i)
            communities = [c for c in communities if c]
            clust = cdlib.NodeClustering(communities, graph=None, method_name=self.name_)
            conductance = cdlib.evaluation.conductance(self.graph_,clust).score
            coverage = cdlib.evaluation.edges_inside(self.graph_,clust).score
            performance = bi_performance(badj, proj0_labels+proj1_labels)
            gini = skbio.diversity.alpha.gini_index([len(c) for c in communities])
            
            # Returning a tuple instead in order to remove coordinates
            result = (
                self.name_,
                 len(c), modularity_score, modularity_score_1,
                modularity_score_2, adj_rand_index, modularity_score_barber,
                conductance, coverage, performance, gini, modularity_score_murata
            )

            temp_results.append(result)
        
        # Remove duplicates
        results_set = set(temp_results)
        # Building result dicts
        cols = ['name', 'num_clusters', 'modularity_score', 'modularity_score_1', 'modularity_score_2', 'adj_rand_index', 'modularity_score_barber',
                'conductance', 'coverage', 'performance', 'gini', 'modularity_score_murata']
        self.results_ = [{k:v for k,v in zip(cols, value)} for value in results_set]

    def compute_hypervolume(self):
        assert self.results_, "Results are not generated yet, please run the community detection first!"
        assert self.params_['save_history'], " History is not saved, not possible to calculate the hypervolume indicator!"

        X, F = self.res_.opt.get("X", "F")
        hist = self.res_.history
        self.n_evals_ = []             # corresponding number of function evaluations
        self.hist_F_ = []               # the objective space values in each generation

        for algo in hist:

            # store the number of function evaluations
            self.n_evals_.append(algo.evaluator.n_eval)

            # retrieve the optimum from the algorithm
            opt = algo.opt

            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            self.hist_F_.append(opt.get("F")[feas])
        
        if self.params_['mode'] == '3d':
            approx_ideal = np.array([-1.,-1., 1.])
            approx_nadir = np.array([1.,1., self.problem_.n_var_])
            ref_point = approx_nadir + 1e-03
        else: # 2d
            approx_ideal = np.array([-1., 1.])
            approx_nadir = np.array([1., self.problem_.n_var_])
            ref_point = approx_nadir + 1e-03
        
        metric = Hypervolume(ref_point=ref_point,
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal,
                     nadir=approx_nadir,
                     )
        self.hv_ = [metric.do(_F) for _F in self.hist_F_]

        return self.n_evals_, self.hv_

    def __str__(self) -> str:
        return f'<ComDetMultiCriteria: params: {self.params_}>'
        
    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_

def bi_performance(badj, communities):
    """
    Calculate the performance of a community assignment, i.e. the fraction of nodes pairs with edges and the same community or without edges and different communities.
    """

    poss_edges = badj.shape[0]*badj.shape[1]
    perf_pairs = 0
    edges = set(zip(badj.tocoo().row,badj.tocoo().col))
    for i in range(badj.shape[0]):
        for j in range(badj.shape[1]):
            if ((i,j) in edges and communities[i] == communities[badj.shape[0]+j]) or ((i,j) not in edges and communities[i] != communities[badj.shape[0]+j]):
                perf_pairs += 1
    return perf_pairs/poss_edges

def modularity_murata(badj,communities):
    """
    Calculate Murata modularity of a given community assignment.
    """
    
    ## Make the e array, fraction of edges between the two communities in each mode.
    e = np.zeros((max(communities)+1,max(communities)+1))
    ## Iterate over the edges.
    for s,t in zip(badj.tocoo().row,badj.tocoo().col):
        ## Increment e_lm where s in comm l and t in comm m.
        e[communities[s]][communities[t+badj.shape[0]]] += 1
    e /= 2*np.sum(e)

    ## Make the a array, the row sums of the e array.
    a = np.sum(e,axis=1)
    
    ## Now we calculate Q, the sum of max observed difference.
    q = 0
    for i in range(e.shape[0]):
        j = np.argmax(e[i])
        q += (e[i][j] - a[i]*a[j])
    return q

def make_badj(graph):
    """
    Turn an igraph object into a biadjency matrix from the edgelist.
    """
    vertex_map = {}  ## Map true id to bipartite id.
    lid,uid = 0,0
    for v in graph.vs():
        if v['name'] == 1:
            bid = uid
            uid += 1
        else:
            bid = lid
            lid += 1
        vertex_map[v.index] = bid
    edge_list = [(vertex_map[e.source],vertex_map[e.target]) for e in graph.es]
    badj = sknetwork.utils.edgelist2biadjacency(edge_list)
    return badj

########################### Some tests

def test_problem(mode="3d"):
    # Sample graph (using fig 06 parameters)
    from data_generation import ExpConfig, DataGenerator
    fig06_expconfig = ExpConfig(L=30, U=30, NumEdges=100, ML=0.5, MU=0.5, BC=0.1, NumGraphs=30,shuffle=False,seed=None,)
    datagen = DataGenerator(expconfig=fig06_expconfig) # or one can just call DataGenerator() --> Default config for data generation
    print(datagen)
    it = datagen.generate_data()
    graph = next(it)
    # igraph.summary(graph)
    print(f"A {mode} problem")
    pb  = MultiCriteriaProblem(mode=mode, graph=graph)
    print(pb)

def test_community_detection(mode="3d"):
    # Sample graph
    import pandas as pd
    from data_generation import ExpConfig, DataGenerator
    import pickle
    fig06_expconfig = ExpConfig(L=30, U=30, NumEdges=100, ML=0.5, MU=0.5, BC=0.1, NumGraphs=30,shuffle=False,seed=None,)
    datagen = DataGenerator(expconfig=fig06_expconfig) # or one can just call DataGenerator() --> Default config for data generation
    print(datagen)
    it = datagen.generate_data()
    graph = next(it)
    mc = ComDetMultiCriteria(
        params = {'mode': '3d', 'popsize': 50, 'termination': None, 'save_history': False, 'seed': None}
        )
    print(mc)
    results = mc.detect_communities(graph=graph).get_results()
    return results


if __name__ == "__main__":
    test_problem(mode="3d")
    # test_problem(mode="2d")
    # results = test_community_detection(mode="3d")