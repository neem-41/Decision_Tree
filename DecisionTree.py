import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

class Line:
    def __init__(self, direction, position):
        self.direction = direction # 0 or 1, horizontal or vertical direction
        self.position = position # a float value between 0 and 1
        self.impurity = np.inf 
class Node:
    def __init__(self, X, Y, depth):
        self.X = X # coordinates of the points
        self.Y = Y # colors of the points 
        self.depth = depth # level of the tree 
        self.split_point = None # a Line object 
        self.left = None # a Node object (left child)
        self.right = None # a Node object (right child)

        ### The following two attributes are designed for printing and plotting of the tree.
        self.boundary = [] # left, right, bottom, top ## for using in plot_tree only 
        self.info = "" # splitting info showing how this node was created ## for using in print_tree only 

    def get_number_of_points(self):
        # Return the number of points in this node
        return len(self.Y)

    def prediction(self):
        # Return the majority color of this node. 
        # blue, C1, -1; red, C2, 1 
        group = np.sign(sum(self.Y)) 
        if group:
            return group
        else: # same 1 and -1
            return self.Y[0]

    def get_impurity(self):
        # Return the impurity of this node
        # Gini index 
        neg, pos, n = sum(self.Y == -1), sum(self.Y == 1), len(self.Y)
        return pos/n*(1-pos/n)+neg/n*(1-neg/n)
class DecisionTree:
    def __init__(self, data_index, min_samples_split, max_depth):
        self.index = data_index
        self.min_samples_split = min_samples_split # default value is 1
        self.max_depth = max_depth if max_depth else np.inf # default value is np.inf 
        self.X, self.Y = self.read_data() # call read_data() to load the data from CSV file
        self.N, self.M = self.X.shape # N - number of points, M - number of features 

    def read_data(self):
        df = pd.read_csv('data_'+self.index+'.csv', sep=',',header=0, skiprows=[1,2])
        self.feature_names = df.columns[:-1]
        self.codes =[-1, 1]
        self.color_dict = {-1: "blue", 1: "red"}
        self.group_dict = {-1: "C1", 1: "C2"}
        for code in self.codes :
            df.iloc[df.iloc[:,-1] == self.group_dict[code], -1] = code
        data = df.values 
        X, Y = data[:,:-1], data[:,-1]
        return X, Y
    
    def get_lines(self, X): 
        # copy your code from Bestline.py 
        X_data= X.copy() 
        lines = [] 
        # Write your code here:
        for direction in [0,1]:
            X_data = sorted(X_data, key = lambda i: i[direction])
            for i in range(len(X) + 1):
                if i == 0:
                    position = (0 + X_data[i][direction])/2
                elif i == len(X):
                    position = (1 + X_data[i-1][direction])/2
                else:
                    position = (X_data[i][direction] + X_data[i-1][direction])/2 

                lines.append(Line(direction, position))
        

        return lines

    def calculate_line(self, line, X, Y): 
        # copy your code from Bestline.py and modify by 
        #   (1) using X and Y instead of self.X and self.Y
        #   (2) remove self.rule and only use Gini index to calculate impurity 
        impurity_value = 0
        # Write your code here:
        d = line.direction
        mask = np.array([True if np.sign(i[d] - line.position) == -1 else False for i in X])
        leftY = list(Y[mask])
        rightY = list(Y[~mask])

        impurity_left = leftY.count(1)/len(leftY) * (1 - leftY.count(1)/len(leftY)) + leftY.count(-1)/len(leftY) * (1 - leftY.count(-1)/len(leftY)) if len(leftY) != 0 else 0
        impurity_right = rightY.count(1)/len(rightY) * (1 - rightY.count(1)/len(rightY)) + rightY.count(-1)/len(rightY) * (1 - rightY.count(-1)/len(rightY)) if len(rightY) != 0 else 0
        
        
        impurity_value = ((len(leftY)/self.N) * impurity_left) + ((len(rightY)/self.N) * impurity_right)
        line.impurity = impurity_value


    def split(self, node):
        # Input: node to be evaluated and splitted if necessary 
        # Output: no output
        # Purpose: recursively call itself to split the node to left and right node. 
        lines = self.get_lines(node.X) 
        best_line = None
        # Copy and modify your code from Bestline.py to find the bestline for current node.
        # Your code goes here:
        for line in lines:
            self.calculate_line(line, node.X, node.Y)

            if best_line == None:
                best_line = line
            elif best_line.impurity > line.impurity:
                best_line = line


        # Update the split_point of current node as the best line
        node.split_point = best_line 
        # Create a mask to split the data of the node. 
        mask =node.X[:, best_line.direction] < best_line.position
        # Use mask and ~mask to initialize two Nodes using X, Y, and depth+1 as left child and right child 
        node.left = Node(node.X[mask], node.Y[mask], node.depth + 1)
        node.right = Node(node.X[~mask], node.Y[~mask], node.depth+1)

        # Call self.split() on left or right child if:
        #    (1) get_number_of_points() of this child is larger than self.min_samples_split
        #    (2) depth of this child is less than self.max_depth
        #    (3) get_impurity() of this child is not 0 (not all points in this child belong to same color)
        # Your code goes here:
        if node.left.get_number_of_points() > self.min_samples_split and node.left.depth < self.max_depth and node.left.get_impurity() != 0:
            self.split(node.left)
        if node.right.get_number_of_points() > self.min_samples_split and node.right.depth < self.max_depth and node.right.get_impurity() != 0:
            self.split(node.right)



    def train(self):
        # call self.split to recursively build the tree from root. 
        root = Node(self.X, self.Y, 0) 
        self.split(root) 
        return root

    def print_tree(self, root):
        # print the structure of the tree in the terminal 
        # DFS 
        nodes = [root]
        while nodes:
            node = nodes.pop()
            if node.info:
                print(node.info)
            if node.split_point: # if it is not a leaf 
                if node.left.split_point: # if left is not a leaf 
                    node.left.info = "| "*node.depth+self.feature_names[node.split_point.direction]+" < "+"{:.3f}".format(node.split_point.position)
                elif node.left.split_point is None:
                    pred = "Class: " + self.color_dict[node.prediction()]
                    node.left.info = "| "*node.depth+self.feature_names[node.split_point.direction]+" < "+"{:.3f}".format(node.split_point.position)+" -> "+pred
                if node.right.split_point:
                    node.right.info = "| "*node.depth+self.feature_names[node.split_point.direction]+" > "+"{:.3f}".format(node.split_point.position)
                elif node.right.split_point is None:
                    pred = "Class: " + self.color_dict[node.prediction()]
                    node.right.info = "| "*node.depth+self.feature_names[node.split_point.direction]+" > "+"{:.3f}".format(node.split_point.position)+" -> "+pred
                nodes.extend([node.left, node.right])

    def plot_tree(self, root, points_only=False):
        # visualize the tree on a two dimensional plane
        for code in self.codes:
            mask = self.Y == code
            X = self.X[mask]
            plt.plot(X[:,0], X[:,1], '.', color=self.color_dict[code])

        if not points_only:    
            root.boundary = [0,1,0,1]     
            nodes = [root]
            while nodes:
                node = nodes.pop()
                if node and node.split_point:
                    line = node.split_point
                    left, right, bottom, top = node.boundary
                    if line.direction: # horizontal direction
                        node.left.boundary = [left, right, bottom, line.position] # bottom region
                        node.right.boundary = [left, right, line.position, top] # top region
                        plt.hlines(line.position, left, right, alpha=1,color='k')
                    else: # vertical direction
                        node.left.boundary = [left, line.position, bottom, top] # left region
                        node.right.boundary = [line.position, right, bottom, top] # right region
                        plt.vlines(line.position, bottom, top, alpha=1,color='k') 

                    nodes.extend([node.left, node.right])

        # plt.xticks([], [])
        # plt.yticks([], [])
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title("Decision Tree on" + \
                   " Data " + self.index + \
                   " with min_samples_split "+ str(self.min_samples_split) + \
                   " and max_depth " + str(self.max_depth))
        plt.savefig("Result_Decision_Tree_" + self.index+ "_"+str(self.min_samples_split) +"_"+str(self.max_depth)+".png")
        # plt.title("data: "+self.index)
        # plt.savefig("Point_"+self.index+".png")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bestline')
    parser.add_argument('-data', dest='data_index', required = True, type = str, help='Index of dataset')
    parser.add_argument('-sample', dest='min_samples_split', required = False, default=1, type = int)
    parser.add_argument('-depth', dest='max_depth', required = False, default=None, type = int)
    
    args = parser.parse_args()

    obj = DecisionTree(args.data_index, args.min_samples_split, args.max_depth)
    root = obj.train()
    # print the tree to terminal 
    obj.print_tree(root)
    # Show points and boundary 
    obj.plot_tree(root)   
    # obj.plot_tree(root, True)

    # python BestLine.py -data [-sample]  [-depth]