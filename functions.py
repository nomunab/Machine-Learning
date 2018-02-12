import scipy.io as sio #to read .mat files. Archives must be uncompressed.
import pickle
import math
import random
import tree as ds


#get the data from .mat files and store in respective lists
def load_data(filename): #NOTE: Matlab arrays start with index 1 while Python starts with 0.
    # param filename
    # return: a tupel of the examples list and the label list
    mat_contents = sio.loadmat(filename) #open a .mat file 
    examples = mat_contents['x'] #load the example dataset. size = N*45 where N = number of examples
    labels = mat_contents['y'] #load the labels. size = N * 1
    return (examples, labels)


# saves an object in a pickle file
def save_object( obj, filename ):
    with open( filename, 'wb' ) as output:
        pickle.dump( obj, output, pickle.HIGHEST_PROTOCOL )
    pass


# loads an object from a pickle file
def load_object( filename ):
    return pickle.load( open( filename, "rb" ) )

    
#create the list of binary_target setting all the examples with given target_value to 1 and everything else with 0.
def get_binary_targets(target_value, labels):
    # param target_value: the number of the emotion
    # param labels: the list of labels
    # return: the binary list classifying the examples into positive and negative
    binary_target = [0 for i in range(len(labels))] #initialize the binary_target array with 0s
    for i in range(len(labels)):
        if labels[i] == target_value:
            binary_target[i] = 1
        else:
            continue
    return binary_target


# the learning function
def decision_tree_learning( examples, attributes, binary_targets ):
    # param data: matrix of data, Nx45 with one datum in each row
    # param attributes: list of yet unconsidered attributes
    # param binary_targets: target label vector, Nx1
    # return: the root node of the learned decision tree

    N = len( binary_targets )
    node = ds.Node()

    # check whether all examples have the same labels
    i = 1
    while i < N and binary_targets[0] == binary_targets[i]:
        i = i + 1
    if i == N:
        node.setLeafNode( binary_targets[0], float( binary_targets[0] ) )
        return node

    # check whether there are no attributes left
    if len(attributes) == 0:
        node.setLeafNode( majority_value( binary_targets ), positive_ratio( binary_targets ) )
        return node

    # choose best classification attribute and remove it from list of attributes
    attr = choose_best_decision_attribute( examples, attributes, binary_targets )
    node.setAttribute( attr )
    sub_attributes = list(attributes)
    sub_attributes.remove( attr )

    for attr_val in [0,1]:

        # generate list of examples that take the respective attribute value
        sub_examples = []
        sub_targets = []
        for i in range(N):
            if examples[i][attr] == attr_val:
                sub_examples.append( examples[i] )
                sub_targets.append( binary_targets[i] )

        # if the attribute can not seperate the examples, the current node becomes a leaf node
        if len( sub_targets ) == 0:
            node.setLeafNode( majority_value( binary_targets ), positive_ratio( binary_targets ) )
            return node

        # create child nodes with respective subset of examples
        else:
            sub_node = decision_tree_learning( sub_examples, sub_attributes, sub_targets )
            node.setChild( attr_val, sub_node )
    return node


# sub-function used in learning function
def choose_best_decision_attribute( examples, attributes, binary_targets ):
    # param data: matrix of data, Nx45 with one datum in each row
    # param attributes: list of yet unconsidered attributes
    # param binary_targets: target label vector, Nx1
    # return: the attribute with the highest information gain, i.e. the lowest remainder

    N = len(binary_targets)
    # set up variables for memorising which attribute achieves the best value
    min_remainder_attr = -1;
    min_remainder = 2.0;

    # check all given attributes
    for attr in attributes:

        # number of negative and positive examples for attribute value 0
        att0 = [0.0,0.0]
        # number of negative and positive examples for attribute value 1
        att1 = [0.0,0.0]
        for i in range(N):
            if examples[i][attr] == 0:
                att0[binary_targets[i]] += 1.0
            else:
                att1[binary_targets[i]] += 1.0

        # compute the remainder of the attribute
        if att0[0] == 0.0 or att0[1] == 0.0:
            I_0 = 0.0
        else:
            I_0 = - ( att0[1] / ( att0[0] + att0[1] ) ) * math.log2( att0[1] / ( att0[0] + att0[1] ) ) \
                  - ( att0[0] / ( att0[0] + att0[1] ) ) * math.log2( att0[0] / ( att0[0] + att0[1] ) )
        if att1[0] == 0.0 or att1[1] == 0.0:
            I_1 = 0.0
        else:
            I_1 = - ( att1[1] / ( att1[0] + att1[1] ) ) * math.log2( att1[1] / ( att1[0] + att1[1] ) ) \
                  - ( att1[0] / ( att1[0] + att1[1] ) ) * math.log2( att1[0] / ( att1[0] + att1[1] ) )
        remainder = ( ( att0[0] + att0[1] ) / float(N) ) * I_0 + ( ( att1[0] + att1[1] ) / float(N) ) * I_1

        # check whether the attribute is better than every attribute considered so far
        if remainder < min_remainder:
            min_remainder = remainder
            min_remainder_attr = attr

    return min_remainder_attr
    

# sub-function used in learning function
def majority_value( binary_targets ):
    # param binary_targets: target vector according to specification on page 13
    # return: the mode of the binary targets
    if positive_ratio( binary_targets ) > 0.5:
        return 1
    else:
        return 0


# sub-function used in learning function
def positive_ratio( binary_targets ):
    # param binary_targets: target vector according to specification on page 13
    # return: the ratio of positive examples in the binary_target vector
    p_positive = binary_targets.count(1)
    return float( p_positive ) / float( len( binary_targets ) )


def testTrees( trees, x2 ):
    # param tree: a trained decision tree
    # param x2: the test data set, Nx45
    # return the list of emotion predictions for the examples

    N = len( x2 )
    pred = [0] * N

    for i in range( N ):
        tree_pred = [0] * len( trees )
        for t in range( len( trees ) ):
            tree_pred[t] = trees[t].classify( x2[i] )

        if tree_pred.count(1) == 1:
            pred[i] = tree_pred.index(1) + 1
        else:
            # strategy if classification is not unique
            # -- strategy one:
            pred[i] = random_classifier( tree_pred ) + 1
            # -- strategy two:
            #tree_pred_ratios = [0.0] * len( trees )
            #for t in range( len( trees ) ):
            #    tree_pred_ratios[t] = trees[t].getPositiveRatio( x2[i] )
            #pred[i] = positive_ratio_classifier( tree_pred_ratios ) + 1
            # -- supervision:
            #print('-----')
            #print( tree_pred )
            #print( tree_pred_ratios )
            #print( pred[i] )

    return pred


# chooses a classifier out of a list of possible classifiers randomly
def random_classifier( pred_class ):
    # param pred_class: a binary array, with pred_class[i] = 1 indicating that classifier i is possible
    # return: a classifier

    n = pred_class.count( 1 )
    if n == 0:
        n = len( pred_class )
    r = random.random()

    for i in range( len( pred_class ) ):
        if pred_class[i] == 1 or n == len( pred_class ):
            r -= 1.0 / float( n )
            if r < 0:
                return i

    # just for safety reasons (I am scared of numerical issues)
    return 0


# chooses the classifier that is most convinced that the datum is positive
def positive_ratio_classifier( pred_ratios ):
    # param pred_class: an array with the positive ratios of each classfier
    # return: a classifier

    if max( pred_ratios ) > 0.0:
        return pred_ratios.index( max( pred_ratios ) )
    else:
        return random_classifier( pred_ratios )
