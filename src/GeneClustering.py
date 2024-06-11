import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

def CreateGeneNetwork(CNA_data, genes):
    # Initialise GenesXGenes matrix

    CNA_data_sparse = csr_matrix(CNA_data)
    # Compute the co-occurrence matrix using sparse matrix multiplication
    CNA_data_sparse = CNA_data_sparse.T.dot(CNA_data_sparse)
    # Convert back to dense matrix if necessary
    # CNA_data_sparse = CNA_data_sparse.toarray()    # Create a graph from the adjacency matrix
    
    G = nx.from_scipy_sparse_array(CNA_data_sparse)

    # Label the nodes using the gene list
    labels = dict(enumerate(genes))
    G = nx.relabel_nodes(G, labels)
    return G

# def CreateGeneNetwork(CNA_data, genes):
#     graph = nx.Graph()
#     graph.add_nodes_from(genes)

#     co_occurrence_matrix = np.dot(CNA_data.T, CNA_data)
#     G = nx.from_numpy_array(co_occurrence_matrix)
#     # Label the nodes using the gene list
#     labels = {i: genes[i] for i in range(len(genes))}
#     G = nx.relabel_nodes(G, labels)
#     return graph


if __name__ == '__main__':
    
    data_cna_path = "/Users/antanas/GitRepo/ChromeX/data/MesotheliomaData/meso_tcga_pan_can_atlas_2018/data_cna.txt"
    DataCNV = pd.read_csv(data_cna_path, sep='\t')
    Genes = DataCNV['Hugo_Symbol']
    
    DataCNV = DataCNV[DataCNV.columns[2:]]
    DataCNV = DataCNV.T
    DataCNV = DataCNV.where(DataCNV >= 0, 1)
    DataCNV = DataCNV.where(DataCNV < 2, 1)
    print(DataCNV.sum(axis=1))
    print(DataCNV.head())
    # Drop columns and genes where all values are 0
    # non_zero_columns = np.any(DataCNV != 0, axis=0)
    # Drop columns which have low occurance:
    column_sum = DataCNV.sum(axis=0)
    non_zero_columns = column_sum > 30
    DataCNV = DataCNV.loc[:, non_zero_columns]
    Genes = Genes[non_zero_columns]
    CNVGraph = CreateGeneNetwork(DataCNV.values, Genes.values)
    # Max of np matrix
    print("Min/Max: ",np.max(nx.to_numpy_array(CNVGraph)), np.min(nx.to_numpy_array(CNVGraph)))
    Louvain_communities = nx.algorithms.community.louvain_communities(CNVGraph, resolution = 1.40)
    print(len(Louvain_communities))
    print(list(Louvain_communities))
    # Filter the communities above a certain size
    Louvain_communities = [community for community in Louvain_communities if len(community) > 1]
    print(Louvain_communities)
    # Louvain_communities
    # print(len(Louvain_communities))
    print(Genes)
    # print(list(CNVGraph.nodes))
    # print(list(CNVGraph.edges))
    # print(CNVGraph.degree[0], CNVGraph.degree[1], CNVGraph.degree[2], CNVGraph.degree[780])
    subax1 = plt.subplot(111)
    nx.draw(CNVGraph)
    
    
    plt.show()
    
    print("Done")
    


    