from functions import *

FILEPATH = 'Solution/iris.csv'

if __name__ == "__main__":
    attributes = [
        'Sepal length',
        'Sepal width',
        'Petal length',
        'Petal width'
    ]
    class_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    m = 2
    # D is the dataset
    # L is an array showing to which class the sample at index column D[i] correspond to
    D, L = load_iris_datasets_from_file(FILEPATH)
    # DP_PCA is the reduced dimension dataset (using the PCA approach)
    DP_PCA = compute_PCA(D, m)
    DP_PCA_setosa, DP_PCA_versicolor, DP_PCA_virginica = filter_dataset_by_labels(DP_PCA, L)
    plot_scatter_attributes_X_label(DP_PCA_setosa, DP_PCA_versicolor, DP_PCA_virginica, title="PCA")

    DP_LDA1 = compute_LDA_generalized_eigenvalue(D, class_mapping, L, directions=2)
    DP_LDA1_setosa, DP_LDA1_versicolor, DP_LDA1_virginica = filter_dataset_by_labels(DP_LDA1, L)
    plot_scatter_attributes_X_label(DP_LDA1_setosa, DP_LDA1_versicolor,
                                    DP_LDA1_virginica, title="LDA Generalize Eigenvalue Problem")

    DP_LDA2 = compute_LDA_generalized_eigenvalue_by_joint_diagonalization(D, class_mapping, L, directions=2)
    DP_LDA2_setosa, DP_LDA2_versicolor, DP_LDA2_virginica = filter_dataset_by_labels(DP_LDA2, L)
    plot_scatter_attributes_X_label(DP_LDA2_setosa, DP_LDA2_versicolor, DP_LDA2_virginica,
                                    title="LDA Joint Diagonalization")


