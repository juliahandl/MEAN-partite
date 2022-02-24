import igraph
import numpy as np

class ExpConfig():
    '''
    This class defines the configuration parameters required
    to generate the dataset (graphs) for community detection 
    '''
    def __init__(
        self, L=100, U=100, NumEdges=200, ML=0.4, MU=0.4, BC=0.2,
        NumGraphs=30, shuffle=True, seed=None):
        self.L = L # L vertices (lower graph part)
        self.U = U # U vertices (lower graph part)
        self.NumEdges=NumEdges
        
        self.ML = ML # Probability of lower nodes belonging to class 0
        self.MU = MU # Probability of upper nodes belonging to class 0
        self.NumGraphs = NumGraphs # Number of graphs to generate for this experiment
        self.BC = BC # Percentage of cross-community edges

        self.NumNodes = L+U # Number of graph nodes
        self.Vertices = [0] * int(self.L) + [1] * int(self.U)
        self.shuffle=shuffle
        self.seed = 42 if seed is None else seed
    
    def __repr__(self) -> str:
        return f'<ExpConfig: L={self.L}, U={self.U}, NumNodes={self.NumNodes}, NumEdges={self.NumEdges}, ML={self.ML}, MU={self.MU}, BC={self.BC}, NumGraphs={self.NumGraphs}, shuffle={self.shuffle}, seed={self.seed}>'


class DataGenerator():
    def __init__(self, expconfig=None):
        self.expconfig = ExpConfig() if expconfig is None else expconfig
    
    def __repr__(self) -> str:
        return f'<DataGenerator: {self.expconfig.__repr__()[1:-1]}>'

    def generate_data(self):
        rng = np.random.default_rng(seed=self.expconfig.seed)
        shuffle = self.expconfig.shuffle
        
        BC = int(100 * self.expconfig.BC)  # Specify percentage cross-community edges
        L = self.expconfig.L
        U = self.expconfig.U
        ML = self.expconfig.ML
        MU = self.expconfig.MU
        vertices = self.expconfig.Vertices
        for it in range(self.expconfig.NumGraphs):
            edges = []
            for i in range(0, self.expconfig.NumEdges):
                dice = rng.integers(0,100)
                added = False
            
                # Find an edge that does not yet exist and add
                # Stochastically generate within community or between community edges
                while added == False:
                    if dice <= 50-BC/2:
                        index1 = rng.integers(0,L*ML-1) # class 0 vertex (in Lower graph part)
                        index2 = rng.integers(L,L+U*MU-1) # class 0 vertex (in Upper graph part)
                    elif dice <= 50:
                        index1 = rng.integers(0,L*ML-1) # class 0 vertex (in Lower graph part)
                        index2 = rng.integers(L+U*MU,L+U-1) # class 1 vertex (in Upper graph part)
                    elif dice <= 50+BC/2:
                        index1 = rng.integers(L*ML,L-1)  # class 1 vertex (in Lower graph part)
                        index2 = rng.integers(L,L+U*MU-1) # class 0 vertex (in Upper graph part)
                    elif dice >= 50+BC/2:
                        index1 = rng.integers(L*ML,L-1) # class 1 vertex (in Lower graph part)
                        index2 = rng.integers(L+U*MU,L+U-1) # class 1 vertex (in Upper graph part)

                    newedge = [index1,index2]

                    if not (newedge in edges):
                        edges.append(newedge)
                        added = True
        
            # Specify original ground truth
            groundtruth=[0]*int(L*ML)+[1]*int(L*(1-ML))+[0]*int(U*MU)+[1]*int(U*(1-MU))
            # shapes = ["rectangle"] * int(L*ML) + ["circle"] * int(L*(1-ML)) + ["rectangle"] * int(U*MU) + ["circle"] * int(U*(1-MU))

            # Reduce to giant component
            g_i = igraph.Graph.Bipartite(vertices, edges)
        
            index_max = np.argmax(g_i.components().sizes()) # Get the largest graph component
            # print(g_i.components().sizes())

            if not shuffle:
                T = [groundtruth[i] for i in g_i.clusters()[index_max]]
                vT = [vertices[i] for i in g_i.clusters()[index_max]]
                g_new=g_i.clusters().giant()
                groundtruth=T
            
                # Create actual bipartite instance
                g_i_new = igraph.Graph.Bipartite(vT,g_new.get_edgelist())

                # Setting attributes
                g_i_new.vs['VX'] = vT # Vertices
                g_i_new.vs['GT'] = groundtruth # Ground truth
                yield g_i_new #, vT, groundtruth,
            else:
                #T = [groundtruth[i] for i in g_i.clusters()[index_max]]
                #vT = [vertices[i] for i in g_i.clusters()[index_max]]
                #g=g_i.clusters().giant()
                #print(g)
            
                oldelist = g_i.get_edgelist()
                oldorder= g_i.clusters()[index_max]#list(range(0,len(T)))
                order=oldorder.copy()
                #print(order)
                rng.shuffle(order)
                # random.shuffle(order)
                #print(groundtruth)
                #print(vT)
            #    print(oldelist)
                #
                #print(oldorder)
                #print(order)
                selected = g_i.clusters()[index_max]
                
                elist=[]
                
                # Create reduced edge list
                for i in range(0,len(oldelist)):
                    try :
                        # Retrieve old idnex
                        ti1 = oldorder.index(oldelist[i][0])
                        ti2 = oldorder.index(oldelist[i][1])
                        # Map vertices to new ids
                        i1 = order.index(oldelist[i][0])
                        i2 = order.index(oldelist[i][1])
                        elist.append([i1,i2])
                        #print(ti1,ti2)
                        #print(i1,i2)
                
                    except ValueError :
                        res = "Element not in list !"
                        # print(res)
                        

                    # print(elist[i])
                        #print(i1,i2)
                #    print(elist) 
                    
                # print(order)
                # print(len(elist))
                # print(len(oldelist))
                # print(T)
                # print(vT)

                labels = [groundtruth[i] for i in order]
                topbottom = [vertices[i] for i in order]
            #    print(topbottom)
                #print(labels)
                #print(topbottom)
                #print(groundtruth)
                #print(vT)
                #print(len(oldelist))
                #print(len(elist))
                
                # Create actual bipartite instance
                g = igraph.Graph.Bipartite(topbottom,elist)
                #g_i = Graph.Bipartite(vT,g.get_edgelist())

                # # Store instance and associated ground truth
                # g.write_edgelist(path+"Graph"+str(it)+".dat")
                # p = pd.DataFrame(labels)
                # p.to_csv(path+"Graph"+str(it)+".truth.dat", sep=',',header=None)
                # p = pd.DataFrame(topbottom)
                # p.to_csv(path+"Graph"+str(it)+".vertices.dat", sep=',',header=None)

                # Setting attributes
                g.vs['VX'] = topbottom # Vertices
                g.vs['GT'] = labels # Ground truth
                yield g#, vT, groundtruth,


def graphs_equal(g1, g2, attribs):
    """
    Checks the equality of two graphs
    Equal means vertex indices are equal, edge vertex indices are equal,
    and the provided vertex attributes are equal (if provided)
    """
    # Check indices
    if (g1.vs.indices != g2.vs.indices) or\
        (g1.get_edgelist() != g2.get_edgelist()):
        return False
    
    # Check attributes
    for attr in attribs:
        if (g1.vs[attr] != g2.vs[attr]):
            return False

    return True


def test_create_datagen():
    rng = np.random.default_rng()
    seed = rng.integers(low=0, high=1000000, size=1)
    # expconfig= ExpConfig(L=20, U=100, NumEdges=200, ML=0.4, MU=0.4, BC=0.2, NumGraphs=30, shuffle=True, seed=seed)
    expconfig= ExpConfig(L=100, U=500, NumEdges=1000, ML=0.4, MU=0.4, BC=0.1, NumGraphs=30, shuffle=True, seed=seed)

    print(expconfig)
    expgen = DataGenerator(expconfig=expconfig)
    print(expgen)
    datagen = expgen.generate_data()
    for i, it in enumerate(datagen, 1):
        # graph, vertices, groundtruth = it
        graph = it
        print(f"generated graph {i} : V {len(graph.vs)}, E:{len(graph.es)}")
        # break


def test_datagen_reproducibility(num_tests=1, attribs=['VX', 'GT']):
    print(f"Testing {num_tests} data generation configuration(s)")
    print(f"Using attribs {attribs}")
    for i in range(num_tests):
        different=False
        rng = np.random.default_rng()
        seed = rng.integers(low=0, high=1000000, size=1)
        print(f"\tConfig {i} using seed {seed}:", sep= ' ')
        # expconfig1= ExpConfig(L=20, U=100, NumEdges=200, ML=0.4, MU=0.4, BC=0.2, NumGraphs=30, shuffle=True, seed=seed)
        # expconfig2= ExpConfig(L=20, U=100, NumEdges=200, ML=0.4, MU=0.4, BC=0.2, NumGraphs=30, shuffle=True, seed=seed)
        expconfig1= ExpConfig(L=100, U=500, NumEdges=1000, ML=0.4, MU=0.4, BC=0.1, NumGraphs=30, shuffle=True, seed=seed)
        expconfig2= ExpConfig(L=100, U=500, NumEdges=1000, ML=0.4, MU=0.4, BC=0.1, NumGraphs=30, shuffle=True, seed=seed)
        expgen1 = DataGenerator(expconfig=expconfig1)
        expgen2 = DataGenerator(expconfig=expconfig2)
        datagen1 = expgen1.generate_data()
        datagen2 = expgen2.generate_data()
        for idx, (g1, g2) in enumerate(zip(datagen1, datagen2)):
            if not graphs_equal(g1, g2, attribs):
                different = True
                break
        if different:
            print(f"\t\tGraphs at index {idx} are not equal")
            break
        else:
            print(f"\t\tAll Graphs are equal")

if __name__ == "__main__":
    # test_create_datagen()
    test_datagen_reproducibility(num_tests=30, attribs=['VX', 'GT'])
