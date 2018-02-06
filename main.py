import scipy.io as sio #to read .mat files. Archives must be uncompressed.
import math
import tree

#get the data from .mat files and store in respective lists
def load_data(filename): #NOTE: Matlab arrays start with index 1 while Python starts with 0. 
    mat_contents = sio.loadmat(filename) #open a .mat file 
    examples = mat_contents['x'] #load the example dataset. size = N*45 where N = number of examples
    labels = mat_contents['y'] #load the labels. size = N * 1
    binary_targets = [[0 for i in range(6)] for j in range(len(labels))]
    for i in range(1,7): #emotion labels start with 1 until 6
        binary_targets[i-1] = get_binary_targets(i, labels) #binary_targets[0] correspond with emotion label '1' where all the labels with value '1'is set to 1 with everything else being 0 and binary_targets[5] correspond with '6'where all the labels with value '6'is set to 1 with everything else being 0. Size = 6*N
    choose_best_decision_attribute(binary_targets[5])
    
#create the list of binary_target setting all the examples with target_value '4' to 1 and everything else with 0.        
def get_binary_targets(target_value, labels): 
    binary_target = [0 for i in range(len(labels))] #initialize the binary_target array with 0s
    for i in range(len(labels)):
        if labels[i] == target_value:
            binary_target[i] = 1
        else:
            continue
    return binary_target
    
load_data("cleandata_students.mat") #DELETE WHEN DONE TESTING


# the learning function, see table 1 on page 13
def decision_tree_learning( data, attributes, targets ):
    # TODO
    # param data: matrix of data according to specification on page 13
    # param attributes: vector of attributes according to specification on page 13
    # param binary_targets: target vector according to specification on page 13
    # return: the learned decision tree, i.e. an object of class Tree
    pass


# sub-function used in learning function
def choose_best_decision_attribute(binary_targets):
    p_positive = binary_targets.count(1)
    print p_positive
    p_negative = binary_targets.count(0)
    entropy_whole = ((-1.0)*p_positive/len(binary_targets)*math.log((p_positive/len(binary_targets)),2))-(p_negative/len(binary_targets)*math.log((p_negative/len(binary_targets)),2))
    

# sub-function used in learning function
def majority_value( binary_targets ):
    # TODO
    # param binary_targets: target vector according to specification on page 13
    # return: the mode of the binary targets
    pass



# TODO: implement evaluation part
