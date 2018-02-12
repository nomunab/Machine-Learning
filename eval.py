import functions as fct
import tree as ds
import numpy as np


# creates six decision trees based on the whole data set and saves them in a pkl file
def create_trees( data_source, data_target='', visu_target='' ):
    # param data_source: filename of the data source
    # param data_target: filename for the trees (use .pkl), for '' trees will not be saved
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

    # save trees if desired
    if data_target != '':
        fct.save_object( trees, data_target )

    # save visualisation if desired
    if visu_target != '':
        out = ''
        for t in range( len( trees ) ):
            out += 'Tree for emotion ' + str( trees[t].getLabel() )
            out += trees[t].visualise() + '\n\n'
        fct.write_file( visu_target, out )

    return trees


def cross_validation( data_source ):
    # param data_source: filename of the data source
    # return: the average confusion matrix

    # init basic variables
    fold_factor = 10
    (examples, labels) = fct.load_data( data_source )
    attributes = list(range(45))
    N = len(labels)
    cs = int( N / fold_factor )

    # statistical data for each emotion: confusion matrix
    # cma[i][j] counts for how many data had label i+1 and label j+1 was predicted
    cma = np.zeros( (6, 6) )

    for c in range( fold_factor ):

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
                continue
            cma[test_labels[i] - 1, pred[i] - 1] += 1.0

    # average confusion matrix
    cma = ( 1. / float( fold_factor ) ) * cma

    return cma
