class DecisionTree:

    # the initialisation function
    def __init__( self, label, root ):

        # an attribute to give the tree a label
        self.label = label

        # the root node
        self.root = root

    def getRoot( self ):
        return self.root

    def setRoot( self, root_node ):
        self.root = root_node

    # computes the class prediction a given single datum
    def classify( self, datum ):
        # param datum: the list of activation units of one datum
        # return: 0 for negative classification, 1 for positive classification
        if self.root is None:
            print('Decision tree is not trained.')
            return None

        curNode = self.root
        while not curNode.isLeafNode():
            attr = curNode.getAttribute()
            curNode = curNode.getChild( datum[attr] )
        return curNode.getClassvalue()

    # finds the positive ratio for a given single datum
    def getPositiveRatio( self, datum ):
        # param datum: the list of activation units of one datum
        # return: the positive ratio for this datum
        if self.root is None:
            print('Decision tree is not trained.')
            return None

        curNode = self.root
        while not curNode.isLeafNode():
            attr = curNode.getAttribute()
            curNode = curNode.getChild( datum[attr] )
        return curNode.getPositiveRatio()


    # a function to create a visualisation of the tree
    def visualise(self):
        # TODO
        pass


class Node:

    # the initialisation function
    def __init__( self ):

        # a reference to the parent node
        self.parent = None
        # an array of references to the child nodes
        # child[0] refers to the child node for action unit turned off
        # child[1] refers to the child node for action unit turned on
        self.child = [None,None]
        # the number of the action unit the node contains
        self.attribute = None
        # see tree.class on page 14 of the manual
        self.classvalue = None
        # ratio of positive examples, i.e. basis for the classvalue
        self.positive_ratio = None

    def getChild( self, child_index ):
        # param child_index: is attribute value positive (1) or negative (0)
        return self.child[child_index]

    def getAttribute( self ):
        return self.attribute

    def getClassvalue( self ):
        return self.classvalue

    def getPositiveRatio( self ):
        return self.positive_ratio

    def isLeafNode( self ):
        return len( self.child ) == 0

    def isRootNode( self ):
        return self.parent == None

    def setParent( self, parent_node ):
        self.parent = parent_node

    def setChild( self, child_index, child_node ):
        # param child_index: is attribute value positive (1) or negative (0)
        self.child[child_index] = child_node
        child_node.setParent( self )

    def setAttribute( self, attribute ):
        self.attribute = attribute

    # makes the node a leaf node, i.e. sets all vars to the required values
    def setLeafNode( self, classvalue, positive_ratio ):
        # param positive_ratio: a float in [0.0,1.0]
        self.child = []
        self.attribute = None
        self.classvalue = classvalue
        self.positive_ratio = positive_ratio
