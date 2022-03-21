import numpy as np
import pandas as pd
import igraph
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
import condor
from condor import condor

class CommunityDetector():
    """
    Base class for community detection
    Inspired from the Estimator API of scikit-learn, cf. https://scikit-learn.org/stable/developers/develop.html
    Results are in self.params_ dictionary
    """

    def __init__(self, name="") -> None:
        # Any parameters not related to the data (Graph)
        # need to be defined here and have default values (in subclasses)
        self.name_ = name
        self.params_ = dict() # Parameters of the community detector
        self.results_ = [] # Results (list of dictionaries)
        
    def check_graph(self, graph):
        assert isinstance(graph, igraph.Graph), "graph must be of type igraph.Graph"
        assert graph.is_bipartite(return_types=False), "graph must be a bipartite graph"
        assert graph.is_connected(), "graph must be fully connected (one connected component)"
        assert len(graph.vs), "graph must not be empty"
        
    def compute_communities(self, graph, y=None): # graph is a bipartite graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        return self # Needs to return self

    def get_results(self):
        # Returns the community detection results
        return self.results_
    
    def get_params(self):
        # Returns the community detection parameters
        return self.params_


class ComDetFastGreedy(CommunityDetector):
    def __init__(self, name= "fastgreedy", params = {'weights': None}, min_num_clusters=1, max_num_clusters=15) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        # edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)
        res_dendo = self.graph_.community_fastgreedy(**self.params_)
       
        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs))
        min_num_clusters = self.min_num_clusters_
        max_num_clusters = min(self.max_num_clusters_, len(self.graph_.vs)) + 1
        
        for k in range(min_num_clusters, max_num_clusters):
            vx_clustering = res_dendo.as_clustering(k)
            proj0_labels=[vx_clustering.membership[i] for i in proj0]
            proj1_labels=[vx_clustering.membership[i] for i in proj1]
            modularity_score = self.graph_.modularity(vx_clustering)
            # modularity_score_1 = graph_proj1.modularity(vx_clustering.membership[0:lower], weights=graph_proj1.es['weight'])
            # modularity_score_2 = graph_proj2.modularity(vx_clustering.membership[lower:n_vertices], weights=graph_proj2.es['weight'])
            modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
            modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
            adj_rand_index = adjusted_rand_score(ground_truth, vx_clustering.membership)
            result = dict(
                name=self.name_,
                num_clusters = k,
                modularity_score = modularity_score,
                modularity_score_1 = modularity_score_1,
                modularity_score_2 = modularity_score_2,
                adj_rand_index = adj_rand_index,
            )
            self.results_.append(result)
        
    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDetEdgeBetweenness(CommunityDetector):
    def __init__(self, name= "edgebetweenness", params = {'directed': False, 'weights': None}, min_num_clusters=1, max_num_clusters=15) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        # edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)
        res_dendo = self.graph_.community_edge_betweenness(**self.params_)

        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs))
        min_num_clusters = self.min_num_clusters_
        max_num_clusters = min(self.max_num_clusters_, len(self.graph_.vs)) + 1
        
        for k in range(min_num_clusters, max_num_clusters):
            vx_clustering = res_dendo.as_clustering(k)
            proj0_labels=[vx_clustering.membership[i] for i in proj0]
            proj1_labels=[vx_clustering.membership[i] for i in proj1]
            modularity_score = self.graph_.modularity(vx_clustering)
            # modularity_score_1 = graph_proj1.modularity(vx_clustering.membership[0:lower], weights=graph_proj1.es['weight'])
            # modularity_score_2 = graph_proj2.modularity(vx_clustering.membership[lower:n_vertices], weights=graph_proj2.es['weight'])
            modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
            modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
            adj_rand_index = adjusted_rand_score(ground_truth, vx_clustering.membership)
            result = dict(
                name=self.name_,
                num_clusters = k,
                modularity_score = modularity_score,
                modularity_score_1 = modularity_score_1,
                modularity_score_2 = modularity_score_2,
                adj_rand_index = adj_rand_index,
            )
            self.results_.append(result)
        
    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDetWalkTrap(CommunityDetector):
    def __init__(self, name= "walktrap", params = {'weights': None, 'steps': 4}, min_num_clusters=1, max_num_clusters=15) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters


    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        # edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)
        res_dendo = self.graph_.community_walktrap(**self.params_) #? steps changed

        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs))
        min_num_clusters = self.min_num_clusters_
        max_num_clusters = min(self.max_num_clusters_, len(self.graph_.vs)) + 1
        
        for k in range(min_num_clusters, max_num_clusters):
            vx_clustering = res_dendo.as_clustering(k)
            proj0_labels=[vx_clustering.membership[i] for i in proj0]
            proj1_labels=[vx_clustering.membership[i] for i in proj1]
            modularity_score = self.graph_.modularity(vx_clustering)
            # modularity_score_1 = graph_proj1.modularity(vx_clustering.membership[0:lower], weights=graph_proj1.es['weight'])
            # modularity_score_2 = graph_proj2.modularity(vx_clustering.membership[lower:n_vertices], weights=graph_proj2.es['weight'])
            modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
            modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
            adj_rand_index = adjusted_rand_score(ground_truth, vx_clustering.membership)
            result = dict(
                name=self.name_,
                num_clusters = k,
                modularity_score = modularity_score,
                modularity_score_1 = modularity_score_1,
                modularity_score_2 = modularity_score_2,
                adj_rand_index = adj_rand_index,
            )
            self.results_.append(result)
        
    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDetMultiLevel(CommunityDetector):
    def __init__(self, name= "multilevel", params = {'weights': None, 'return_levels': False}, min_num_clusters=1, max_num_clusters=15) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters

    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        ground_truth = self.graph_.vs['GT']
        proj0 = [i for i, val in enumerate(vertices) if val == 0]
        proj1 = [i for i, val in enumerate(vertices) if val == 1]
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        # Run Multi-Level algorithm (not implemented in igraph package)
        res1 = graph_proj1.community_multilevel(**self.params_)
        res2 = graph_proj2.community_multilevel(**self.params_)

        # # Cluster assignment from each projection
        # assignment=res1.membership + res2.membership

        # Consider perturbation in creating new membership vector
        it1=0
        it2=0
        assignment = [0] * len(vertices)
        for vit in range(0, len(vertices)):
            if vertices[vit] == 0:
                assignment[vit] = res1.membership[it1]
                it1=it1+1
            else:
                assignment[vit] = res2.membership[it2]
                it2=it2+1

        k1 = max(res1.membership) + 1
        k2 = max(res2.membership) + 1
        d = np.zeros(shape=(k1+k2, k1+k2))

        # Calculate dissimilarity matrix between communities (rows/columns are community indices in the 2 projected graph, and values are the number of edges linking those communities (vertices))
        for ei in range(0, len(edges)): # For each edge in the bipartite graph
            #print(edges[ei],edges[ei][0],edges[ei][1])
            index1 = assignment[edges[ei][0]] # Get community index of the edge's source vertex in the first one-mode projection
            index2 = k1 + assignment[edges[ei][1]] # Get community index of the edge's target vertex in the second one-mode projection # Why assigning different communitites to one vertex? A guarantee that the 2nd vertex of each edge belongs to the second one-mode projection?
            if vertices[edges[ei][0]] == 0: # a (0,1) edge
                index1=assignment[edges[ei][0]]
                index2=k1+assignment[edges[ei][1]]
            else: # a (1,0) edge
                index1=k1+assignment[edges[ei][0]]
                index2=assignment[edges[ei][1]]

            # print(index1,index2)
            d[index1][index2] += 1 # Update matrix item
            d[index2][index1] += 1 # Update matrix item
        
        # Normalize (adding 1 to avoid division by zero) and setting the matrix main diagonal to zero
        for d1 in range(0, k1+k2):
            for d2 in range(0, k1+k2):
                d[d1][d2] = 1.0/(1.0+d[d1][d2])
            d[d1][d1] = 0
        
        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs)) # This is a different case (see below)
        for k in range(1, k1+k2):
            # Run hierarchical clustering on communities
            clustering = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='precomputed').fit(d)
            labels = clustering.labels_

            newlabels = np.zeros(n_vertices)

            # for v in range(0,lower):
            #     newlabels[v] = labels[assignment[v]]

            # for v in range(lower,n_vertices):
            #     newlabels[v] = labels[k1 + assignment[v]]

            for v in range(0,n_vertices):
                if vertices[v] == 0:
                    newlabels[v] = labels[assignment[v]]
                else:
                    newlabels[v] = labels[k1+assignment[v]]

            proj0_labels=[newlabels[i] for i in proj0]
            proj1_labels=[newlabels[i] for i in proj1]
            
            modularity_score = self.graph_.modularity(newlabels)
            # modularity_score_1 = graph_proj1.modularity(newlabels[0:lower], weights=graph_proj1.es['weight'])
            # modularity_score_2 = graph_proj2.modularity(newlabels[lower:n_vertices], weights=graph_proj2.es['weight'])
            modularity_score_1 = graph_proj1.modularity(proj0_labels, weights=graph_proj1.es['weight'])
            modularity_score_2 = graph_proj2.modularity(proj1_labels, weights=graph_proj2.es['weight'])
            adj_rand_index = adjusted_rand_score(ground_truth, newlabels)
            result = dict(
                    name=self.name_,
                    num_clusters = k,
                    modularity_score = modularity_score,
                    modularity_score_1 = modularity_score_1,
                    modularity_score_2 = modularity_score_2,
                    adj_rand_index = adj_rand_index,
                )
            self.results_.append(result)
        
    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDetBRIMNoPert(CommunityDetector):
    def __init__(self, name= "brim", params = {'method': 'LCS', 'project': False}, min_num_clusters=1, max_num_clusters=15) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters


    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs)) # This is a different case (see below)
        ground_truth = self.graph_.vs['GT']
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        net = pd.DataFrame(edges, dtype=str)

        co = condor.condor_object(dataframe=net, verbose=False)
        co.initial_community(**self.params_)
        co.brim()

        groundtruth1 = ground_truth[0:lower]
        groundtruth2 = ground_truth[lower:n_vertices]

        # output1 = co["reg_memb"]
        # output1 = output1["com"].tolist()
        # output2 = co["tar_memb"]
        # output2 = output2["com"].tolist()

        output1=co.reg_memb
        # print(output1.shape)
        output1=output1["com"].tolist()
        output2 = co.tar_memb
        # print(output2.shape)
        output2=output2["com"].tolist()
        
        # adj_rand_index_1 = adjusted_rand_score(groundtruth1, output2)
        # adj_rand_index_2 = adjusted_rand_score(groundtruth2, output1)
        output3 = output2 + output1
        adj_rand_index = adjusted_rand_score(ground_truth, output3)
        
        modularity_score = self.graph_.modularity(output3)
        # print(len(output2), len(graph_proj1.vs), len(graph_proj1.es['weight']))
        # print(len(output1), len(graph_proj2.vs), len(graph_proj2.es['weight']))
        modularity_score_1 = graph_proj1.modularity(output1, weights = graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(output2, weights = graph_proj2.es['weight'])

        k = (max(output3) + 1)
        result = dict(
                name=self.name_,
                num_clusters = k,
                modularity_score = modularity_score,
                modularity_score_1 = modularity_score_1,
                modularity_score_2 = modularity_score_2,
                adj_rand_index = adj_rand_index,
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_


class ComDetBRIM(CommunityDetector):
    def __init__(self, name= "brim", params = {'method': 'LCS', 'project': False}, min_num_clusters=1, max_num_clusters=15) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        
        assert min_num_clusters >= 1 and min_num_clusters <= max_num_clusters,\
        f"The minimum {min_num_clusters} and maximum {max_num_clusters} cluster numbers are not valid"
        self.min_num_clusters_ = min_num_clusters
        self.max_num_clusters_ = max_num_clusters


    def check_graph(self, graph):
        super().check_graph(graph)
        # Additional checks go here

    def detect_communities(self, graph, y=None):
        #TODO: fit instead of this and y as groundtruth or None to infer from the graph
        # Some checks
        self.check_graph(graph)
        self.graph_ = graph
        self.results_ = [] # Reset results at each call
        # Community detection done here (results stored in self.results_)
        self.__detect_communitites()
        return self # Needs to return self
       
    def __detect_communitites(self):
        # Actual community detection code
        vertices = list(map(int, self.graph_.vs['type']))
        edges = self.graph_.get_edgelist()
        lower = vertices.count(0)
        upper = vertices.count(1)
        n_vertices = len(self.graph_.vs)
        # num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs)) # This is a different case (see below)
        ground_truth = self.graph_.vs['GT']
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)

        # Fix edgelist representation for BRIM package (needs 0 vertices as start)
        for i in range (0, len(edges)):
            if vertices[edges[i][0]] == 0:
                temp = edges[i]
                edges[i] = (temp[0], temp[1])
            elif vertices[edges[i][1]] == 0:
                temp = edges[i]
                edges[i] = (temp[1], temp[0])
            else:
                print("Error")

        net = pd.DataFrame(edges, dtype=str)

        co = condor.condor_object(dataframe=net, verbose=False)
        co.initial_community(**self.params_)
        co.brim()

        # groundtruth1 = ground_truth[0:lower]
        # groundtruth2 = ground_truth[lower:n_vertices]

        # output1 = co["reg_memb"]
        # output1 = output1["com"].tolist()
        # output2 = co["tar_memb"]
        # output2 = output2["com"].tolist()

        output1=co.reg_memb
        output1=output1["com"].tolist()
        output2 = co.tar_memb
        output2=output2["com"].tolist()
        
        # adj_rand_index_1 = adjusted_rand_score(groundtruth1, output2)
        # adj_rand_index_2 = adjusted_rand_score(groundtruth2, output1)
        # output3 = output2 + output1

        output3 = np.full(n_vertices, -1, dtype=int)
        index1 = 0
        index2 = 0
        for v in range (0, n_vertices):
            if vertices[v] == 0:
                output3[v] = output2[index1]
                index1 += 1
            else:
                output3[v] = output1[index2]
                index2 += 1

        adj_rand_index = adjusted_rand_score(ground_truth, output3)
        
        modularity_score = self.graph_.modularity(output3)
        modularity_score_1 = graph_proj1.modularity(output1, weights = graph_proj1.es['weight'])
        modularity_score_2 = graph_proj2.modularity(output2, weights = graph_proj2.es['weight'])

        k = (max(output3) + 1)
        result = dict(
                name=self.name_,
                num_clusters = k,
                modularity_score = modularity_score,
                modularity_score_1 = modularity_score_1,
                modularity_score_2 = modularity_score_2,
                adj_rand_index = adj_rand_index,
            )
        self.results_.append(result)

    # Optional overriding
    # def get_results(self):
    #     # Returns the community detection results (dict free format)
    #     return self.results_

#TODO ComDetBRIM (perturbation version) based on ComDetBRIMNoPert
#TODO Compare condor results for the current version and the old version 1.1 that Julia used

########################################################
#### Utility
########################################################
def get_best_community_solutions(df_contestants):
    """
    Computes the best solution metrics among the hierarchical communities computed by community detection algorithms
    For a given algorith/graph pair, the best solution maximizes the adjusted rand index score
    """
    assert isinstance(df_contestants, pd.DataFrame), "df_contestants must be a data frame"
    # columns = ['name', 'num_clusters', 'modularity_score', 'modularity_score_1', 'modularity_score_2', 'adj_rand_index', 'graph_idx']
    columns = ['name', 'graph_idx', 'adj_rand_index']
    assert set(columns).issubset(set(df_contestants.columns)), f"Column names must include (in any order): {columns}"
    return df_contestants.groupby(['name', 'graph_idx'])['adj_rand_index'].max().reset_index()

import seaborn as sns
import matplotlib.pyplot as plt

def draw_best_community_solutions(df_best_community_solutions, ax=None):
    """
    Box plots the best solutions (adjusted rand index)
    """
    assert isinstance(df_best_community_solutions, pd.DataFrame), "df_best_community_solutions must be a data frame"
    columns = ['name', 'adj_rand_index']
    assert set(columns).issubset(set(df_best_community_solutions.columns)), f"Column names must include (in any order): {columns}"
    # stats = df_best_community_solutions.groupby(['name'])['adj_rand_index'].describe().reset_index(frop=False)
    # ax = df_best_community_solutions.boxplot(column='adj_rand_index', by='name')
        
    ax = sns.boxplot(y='adj_rand_index', x='name', data=df_best_community_solutions, ax=ax)
    return ax, df_best_community_solutions.groupby(['name'])['adj_rand_index'].describe().reset_index()
    # # , hue=None,
    # )
    # sns.boxplot(
    #     y='modularity_score_1', x='name', data=df, ax=axs[1]
    # # , hue=None,
    # )
    # sns.boxplot(
    #     y='modularity_score_2', x='name', data=df, ax=axs[2]
    # # , hue=None,
    # )
    # sns.boxplot(
    #     y='adj_rand_index', x='name', data=df, ax=axs[3]
    # # , hue=None,
    # )
    # # None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5, ax=None, **kwargs)


def test_community_detector():
    # Data generation
    from data_generation import ExpConfig, DataGenerator
    expconfig = ExpConfig()
    expgen = DataGenerator(expconfig=expconfig)
    print(expgen)
    datagen = expgen.generate_data()
    for it in datagen:
        # Save graph somewhere or viz.
        graph = it
        break
    igraph.summary(graph)
    print(graph.is_connected())
    # com_det = ComDetFastGreedy(num_clusters=15)
    # com_det = ComDetEdgeBetweenness(num_clusters=15)
    com_det = ComDetWalkTrap(num_clusters=15)
    print(com_det)
    com_det.detect_communities(graph=graph)
    result = com_det.get_results()
    params = com_det.get_params()
    print(result)
    print(params)
    import pandas as pd
    df = pd.DataFrame(result)
    print(df)


if __name__ == "__main__":
    
    # test_community_detector()
    # Data generation
    from data_generation import ExpConfig, DataGenerator
    expconfig = ExpConfig()
    expgen = DataGenerator(expconfig=expconfig)
    # print(expgen)
    datagen = expgen.generate_data()
    graphs = list(datagen)[:1]
    # igraph.summary(graphs[1])
    # for it in datagen:
    #     # Save graph somewhere or viz.
    #     graph = it
    #     break
    algos = [
        ComDetEdgeBetweenness(num_clusters=15),
        ComDetWalkTrap(num_clusters=15),
        ComDetFastGreedy(num_clusters=15),
    ][:1]
    results = []
    for g_idx, graph in enumerate(graphs):
        # print(g_idx)
        for algo in algos:
            # print(algo)
            result = algo.detect_communities(graph=graph).get_results()
            # print(result)
            for r in result:
                r['graph_idx'] = g_idx
                results.extend(result)

    import pandas as pd
    df = pd.DataFrame(results)
    print(df.shape)





    



