from MultiClassifier import MultiClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from matplotlib import pyplot as plt    
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    
    data_cna_path = "/Users/antanas/GitRepo/ChromeX/data/MesotheliomaData/meso_tcga_pan_can_atlas_2018/data_cna.txt"
    data_rna_path = "/Users/antanas/GitRepo/ChromeX/data/MesotheliomaData/meso_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt"
    
    DataCNA = pd.read_csv(data_cna_path, sep='\t')
    DataRNA = pd.read_csv(data_rna_path, sep='\t')
    DataRNA = DataRNA.dropna()
    
    print(DataCNA.head(-10), "\n")
    print(DataRNA.head(-10), "\n")
    print("Intersection: ", DataCNA.columns.intersection(DataRNA.columns),
          "Length: ", len(DataCNA.columns.intersection(DataRNA.columns)))
    
    IntersectionPatients = DataCNA.columns.intersection(DataRNA.columns[2:])
    ProteinGenes = DataRNA['Hugo_Symbol']
    Genes = DataCNA['Hugo_Symbol']
    
    DataCNA = DataCNA[IntersectionPatients].T
    DataCNA = DataCNA.where(DataCNA >= 0, 1)
    DataCNA = DataCNA.where(DataCNA < 2, 1)
    
    DataRNA = DataRNA[IntersectionPatients].T
    
    DataCNA.columns = Genes
    DataRNA.columns = ProteinGenes
    
    Train_data, Test_data, Train_labels, Test_labels = train_test_split(DataRNA, DataCNA, test_size=0.20, random_state=0)
    
    # CNVClassifier = MultiOutputClassifier(DecisionTreeClassifier())
    # CNVClassifier = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=100)
    # CNVClassifier = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=0)
    
    # CNVClassifier = RandomForestClassifier(criterion='gini', random_state=0, max_depth=100)

    CNVClassifier = LogisticRegression(random_state=0, max_iter=1000, penalty='l1', solver='liblinear')
    CNVClassifier = MultiOutputClassifier(CNVClassifier)
    
    # CNVClassifier = RandomForestClassifier(criterion='gini', random_state=0, max_depth=100)
    # CNVClassifier = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=300)

    CNVClassifier.fit(Train_data, Train_labels)

    predictions = CNVClassifier.predict(Test_data)
    predictions = pd.DataFrame(predictions, columns=Test_labels.columns, index=Test_labels.index)
    
    # ROC AUC
    ROC = []
    from sklearn.metrics import roc_auc_score
    for gene in Test_labels.columns:
        ROC.append(roc_auc_score(Test_labels[gene], predictions[gene]))
    
    plt.plot(np.sort(ROC)[::-1])
    print(Test_labels.columns[np.argsort(ROC)[::-1]])
    plt.show()
    
    importances = [estimator.feature_importances_ for estimator in CNVClassifier.estimators_]
    
    plt.pcolor(importances)
    plt.show()
    
    # Multiclass Precision, Recall, F1-Score
    # accuracy = np.sum(predictions == Test_labels.values, axis=0)/len(predictions)
    # print("Accuracy: ", accuracy)
    
    # # Precision, Recall, F1-Score
    
    # precision = np.logical_and(predictions == 1, Test_labels.values == 1).sum(axis=0)/np.sum(predictions == 1, axis=0)
    # recall = np.logical_and(predictions == 1, Test_labels.values == 1).sum(axis=0)/np.sum(Test_labels.values == 1, axis=0)
    # f1_score = 2*precision*recall/(precision + recall)
    # # print(np.sum(Test_labels.values == 1, axis=0))
    # # plt.plot(np.sum(Test_labels.values == 1, axis=0))
    # plt.plot(accuracy)
    # plt.plot(precision, 'r', label='Precision')
    # plt.plot(recall, 'g', label='Recall')
    # plt.plot(f1_score, 'y', label='F1-Score')
    # plt.show()
    
    # print(DataCNA.head(), "\n")
    # print(DataRNA.head(), "\n")
    # classifier = DecisionTreeClassifier()
    
    # mc = MultiClassifier()

