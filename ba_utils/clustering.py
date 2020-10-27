'''
Clustering utilities
'''

from sklearn.cluster import SpectralCoclustering
def min_offcluster(exam_matrix_binary):
    '''
    look for the optimal number of clusters when trying to minimize the sum of elements outside of 
    the clusters in range from 3 to 30 (values can ge changed in function)

    arguments:
        exam_matrix_binary: a binary matrix
    returns:
        optimal number of clusters
    '''
    sum_vec = []
    for i in range(3, 30):
        model = SpectralCoclustering(n_clusters=i, random_state=0)
        model.fit(exam_matrix_binary+1)
        fitsum = 0
        for j in range(i):
            fitsum = fitsum + sum(exam_matrix_binary.iloc[model.row_labels_ == j, model.column_labels_ == j].sum())
        neg_sum = sum(exam_matrix_binary.sum()) - fitsum
        sum_vec.append(neg_sum)
    for i, value in enumerate(sum_vec):
        if value == min(sum_vec):
            return i + 3

def density(exam_matrix_binary):
    '''
    look for the optimal number of clusters when trying to get as dense clusters as possible
    in range from 3 to 30 (values can ge changed in function)

    arguments:
        exam_matrix_binary: a binary matrix
    returns:
        optimal number of clusters
    '''
    density_vec = []
    for i in range(3 ,30):
        model = SpectralCoclustering(n_clusters = i, random_state=0)
        model.fit(exam_matrix_binary+1)
        fitsum=[]
        for j in range(i):
            fitsum.append(sum(exam_matrix_binary.iloc[model.row_labels_ == j, model.column_labels_ == j].sum())/(sum(model.row_labels_ == j)*sum(model.column_labels_ == j)))
        density_vec.append(sum(fitsum)/i)
    for i, value in enumerate(density_vec):
        if value == max(density_vec):
            return i + 3 