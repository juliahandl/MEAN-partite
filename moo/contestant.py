import igraph
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score

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
        pass

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
    def __init__(self, name= "fastgreedy", params = {'weights': None}, num_clusters=15) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        self.num_clusters_ = num_clusters

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
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)
        res_dendo = self.graph_.community_fastgreedy(**self.params_)
        num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs))
        ground_truth = self.graph_.vs['GT']
        for k in range(1, num_clusters):
            vx_clustering = res_dendo.as_clustering(k)
            modularity_score = self.graph_.modularity(vx_clustering)
            modularity_score_1 = graph_proj1.modularity(vx_clustering.membership[0:lower], weights=graph_proj1.es['weight'])
            modularity_score_2 = graph_proj2.modularity(vx_clustering.membership[lower:n_vertices], weights=graph_proj2.es['weight'])
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
    def __init__(self, name= "edgebetweenness", params = {'directed': False, 'weights': None}, num_clusters=15) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        self.num_clusters_ = num_clusters

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
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)
        res_dendo = self.graph_.community_edge_betweenness(**self.params_)
        num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs))
        ground_truth = self.graph_.vs['GT']
        for k in range(1, num_clusters):
            vx_clustering = res_dendo.as_clustering(k)
            modularity_score = self.graph_.modularity(vx_clustering)
            modularity_score_1 = graph_proj1.modularity(vx_clustering.membership[0:lower], weights=graph_proj1.es['weight'])
            modularity_score_2 = graph_proj2.modularity(vx_clustering.membership[lower:n_vertices], weights=graph_proj2.es['weight'])
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
    def __init__(self, name= "walktrap", params = {'weights': None, 'steps': 4}, num_clusters=15) -> None:
        #TODO: A range of cluster with a possibility to generate automatically (str argument)
        super().__init__(name)
        self.params_ = params
        self.num_clusters_ = num_clusters

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
        graph_proj1, graph_proj2 = self.graph_.bipartite_projection(multiplicity=True)
        res_dendo = self.graph_.community_walktrap(**self.params_) #? steps changed
        num_clusters = min(self.num_clusters_+ 1, len(self.graph_.vs))
        ground_truth = self.graph_.vs['GT']
        for k in range(1, num_clusters):
            vx_clustering = res_dendo.as_clustering(k)
            modularity_score = self.graph_.modularity(vx_clustering)
            modularity_score_1 = graph_proj1.modularity(vx_clustering.membership[0:lower], weights=graph_proj1.es['weight'])
            modularity_score_2 = graph_proj2.modularity(vx_clustering.membership[lower:n_vertices], weights=graph_proj2.es['weight'])
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





    



# class CommunityMultiLevelDetector(CommunityDetector):
#     def __init__(self) -> None:
#         super().__init__()

#     def check_graph(self):
#         pass

#     def compute_communities(self, graph, y=None):
#         super().compute_communities(graph, y)
#         self.__detect_communitites()
#         return self

#     def __detect_communitites(self):
#         # Actual community detection code
#         pass
    
#     # Optional overriding
#     def get_results(self):
#         # Returns the community detection results (dict free format)
#         return self.results_
    

# def dissimilarity_matrix():
#     pass