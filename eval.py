import main as fct
import tree as ds
import numpy as np


def create_trees( data_source ):
    # param data_source: filename of the data source
    # return array of the six trees, tree[i] for emotion i+1

    # init basic variables
    (examples, labels) = fct.load_data( data_source )
    attributes = list(range(45))

    # create trees on the basis of the whole data set
    trees = [None] * 6
    for emotion in range(1,7):
        targets = fct.get_binary_targets( emotion, labels )
        root = fct.decision_tree_learning( examples, attributes, targets )
        trees[emotion-1] = ds.DecisionTree( emotion, root )

    return trees


def cross_validation( data_source ):
    # param data_source: filename of the data source

    # init basic variables
    fold_factor = 10
    (examples, labels) = fct.load_data( data_source )
    attributes = list(range(45))
    N = len(labels)
    cs = int( N / fold_factor )

    # statistical data for each emotion
    # cma[i][j] counts for how many data had label i+1 and label j+1 was predicted
    # zc counts how many data could not be assigned to exactly one class
    cma = np.zeros( (6, 6) )
    zc = 0

    for c in range(1,fold_factor):

        # divide the data set into training and test set
        training_examples = np.concatenate( (np.copy( examples[:(c*cs)] ), np.copy( examples[((c+1)*cs):] )), axis=0 )
        training_labels = np.concatenate( (np.copy( labels[:(c*cs)] ), np.copy( labels[((c+1)*cs):] )), axis=0 )
        test_examples = np.copy( examples[(c*cs):(c+1)*cs] )
        test_labels = np.copy( labels[(c*cs):(c+1)*cs].copy() )

        # generate decision trees on the basis of the training data set
        trees = [None] * 6
        for emotion in range(1,7):
            targets = fct.get_binary_targets( emotion, training_labels )
            root = fct.decision_tree_learning( training_examples, attributes, targets )
            trees[emotion-1] = ds.DecisionTree( emotion, root )

        # compute predictions
        pred = fct.testTrees( trees, test_examples )

        # count labels and predictions
        for i in range( len( test_examples ) ):
            if pred[i] == 0:
                zc += 1
                continue
            cma[test_labels[i] - 1, pred[i] - 1] += 1.0

    # print(zc)

    return cma
