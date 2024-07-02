"""Implementation of base Tree and Node class."""

import warnings

import numpy as np

import hmt
from hmt.exceptions import HMTError, HMTWarning
from hmt.utils import insort


class Node:
    """Attributes:
    ----------
    _path: str
        Binary representation of where the node is on the tree
        0 => left and 1 => right. e.g.
               1      1 = 0     4 = 000
             /   \    2 = 00    5 = 001
            2     3   3 = 01    6 = 010
           / \   /
          4   5 6
    """

    def __init__(self, node_id, observed=None, tree=None):
        self.id = node_id
        self.x = observed
        self.tree = tree
        self._path = ""
        self.mother = None
        self.d0 = None
        self.d1 = None

    def __len__(self):
        n_nodes = 1
        if self.d0 is not None:
            n_nodes += len(self.d0)
        if self.d1 is not None:
            n_nodes += len(self.d1)
        return n_nodes

    def __lt__(self, other):
        return self._path < other._path

    def __repr__(self):
        if self.x is None:
            return f"Node({self.id})"
        if isinstance(self.x, float):
            return f"Node({self.id}, x={round(self.x, 1)})"
        return f"Node({self.id}, x={self.x})"

    def printTree(self, level=0, attr_name=None, arrow="\u2b62 ", max_level=None):
        if max_level is not None and level == max_level:
            return
        if self.d1 is not None:
            self.d1.printTree(
                level + 1, attr_name, arrow="\u2ba3 ", max_level=max_level
            )
        if attr_name is not None:
            attr = getattr(self, attr_name)

            if isinstance(attr, float):
                attr = round(attr, 2)
            strlen = len(str(attr))
            if isinstance(attr, np.ndarray):
                strlen = 2 + attr.shape[0] * 5

            with np.printoptions(precision=2, suppress=True):
                print(
                    " " * (11 + strlen + len(attr_name)) * level
                    + arrow
                    + f"{self.id} ({attr_name} = {attr})"
                )
        else:
            print(" " * 5 * level + arrow + str(self.id))
        if self.d0 is not None:
            self.d0.printTree(
                level + 1, attr_name, arrow="\u2ba1 ", max_level=max_level
            )

    def xlen(self, attr_name="x"):
        """Returns the number of nodes for which attr is not nan"""
        attr = getattr(self, attr_name)

        if isinstance(attr, np.ndarray):
            num = 1 * ~np.isnan(attr)
        else:
            num = 1 or 0
        if self.d0 is not None:
            num += self.d0.xlen(attr_name)
        if self.d1 is not None:
            num += self.d1.xlen(attr_name)
        return num

    def set_tree(self, tree):
        self.tree = tree
        if self.d0 is not None:
            self.d0.set_tree(tree)
        if self.d1 is not None:
            self.d1.set_tree(tree)

    def leaves(self):
        if self.d0 is not None and self.d1 is not None:
            # Sorts using __lt__ which is based on _path attribute
            # leaves are sorted left to right by default
            return sorted(self.d0.leaves() + self.d1.leaves())
        if self.d0 is not None:
            return self.d0.leaves()
        if self.d1 is not None:
            return self.d1.leaves()
        return [self]

    def find(self, node_id):
        # if node_id == 9:
        #     print(self.id)
        if self.id == node_id:
            return self
        if self.d0 is not None:
            left = self.d0.find(node_id)
            if left is not None:
                return left
        if self.d1 is not None:
            right = self.d1.find(node_id)
            if right is not None:
                return right

    def where(self, attr, cond):
        nodes = []
        if cond(getattr(self, attr)):
            nodes.append(self)
        if self.d0 is not None:
            nodes += self.d0.where(attr, cond)
        if self.d1 is not None:
            nodes += self.d1.where(attr, cond)
        return nodes

    def sum(self, attrs, func=None):
        if isinstance(attrs, str):
            attr = getattr(self, attrs)
            if attr is None:
                total = 0
            elif func is None:
                total = 1.0 * attr
            else:
                total = func(attr)
        elif isinstance(attrs, tuple) or isinstance(attrs, list):
            attr_list = [getattr(self, attr) for attr in attrs]

            if func is None:
                total = sum(attr_list)
            else:
                total = func(*attr_list)
        if isinstance(total, np.ndarray):
            total[np.isnan(total)] = 0
        else:
            total = total or 0
        if self.d0 is not None:
            total += self.d0.sum(attrs, func)
        if self.d1 is not None:
            total += self.d1.sum(attrs, func)

        # if total is None:
        #     if isinstance(attrs, tuple):
        #         print(f"{attr_list = }")
        #     return 0

        return total

    def max(self, attr):
        x = getattr(self, attr)
        if self.d0 is not None:
            x = max(x, self.d0.max(attr))
        if self.d1 is not None:
            x = max(x, self.d1.max(attr))
        return x

    def apply(self, func, attr, drec=False):
        # Update the attribute to be the attribute with the function applied
        setattr(self, attr, func(getattr(self, attr)))
        if drec:
            if self.d0 is not None:
                self.d0.apply(func, attr, drec)
            if self.d1 is not None:
                self.d1.apply(func, attr, drec)

    def astype(self, nodeclass):
        new_node = nodeclass()
        old_dict = self.__dict__  # noqa F841
        new_dict = new_node.__dict__  # noqa F841

        if self.mother is not None:
            if int(self.path[-1]):
                # if d1
                self.mother.d1 = new_node
            else:
                self.mother.d0 = new_node

        if self.d0 is not None:
            new_node.d0.astype(nodeclass)
        if self.d1 is not None:
            new_node.d1.astype(nodeclass)


class Tree:
    def __init__(self, root=None):
        if root is not None:
            self.root = root
            self.root.set_tree(self)
            self.leaves = self.root.leaves()
        else:
            self.root = None
            self.leaves = []

    def __str__(self):
        if self.root is not None:
            self.root.printTree()
            return ""
        return "Tree()"

    def __len__(self):
        return len(self.root)

    def xlen(self, attr="x"):
        return self.root.xlen(attr)

    def show(self, attr_name=None, max_level=None):
        self.root.printTree(attr_name=attr_name, max_level=max_level)

    def get_node(self, node_id):
        node = self.root.find(node_id)
        if node is None:
            raise HMTError(f"Node {node_id} not found.")
        return node

    def get_leaf(self, node_id):
        for node in self.leaves:
            if node.id == node_id:
                return node
        raise HMTError(f"Node {node_id} not found in self.leaves.")

    def add_node(self, node, mother_id):
        # If this is the first node
        if self.root is None:
            self.root = node
            self.root.set_tree(self)
            self.leaves = node.leaves()
            return

        # Check if we can add the node
        try:
            mother_node = self.get_node(mother_id)
        except HMTError:
            raise HMTError(
                f"Mother node (node {mother_id}) not found while adding node {node.id}."
            ) from None

        node.set_tree(self)

        if mother_node.d0 is None:
            mother_node.d0 = node
            node._path = mother_node._path + "0"
        elif mother_node.d1 is None:
            mother_node.d1 = node
            node._path = mother_node._path + "1"
        else:
            raise HMTError(f"Mother node (node {mother_id}) already has two children.")

        node.mother = mother_node

        if mother_node in self.leaves:
            self.leaves.remove(mother_node)
        for leaf in node.leaves():
            insort(self.leaves, leaf)

    def update_node(self, node_id, observed):
        node = self.get_node(node_id)
        node.x = observed

    def remove(self, node):
        if int(node._path[-1]):
            node.mother.d1 = None
            if node.mother.d0 is None:
                self.leaves.append(node.mother)
        else:
            node.mother.d0 = None
            if node.mother.d1 is None:
                self.leaves.append(node.mother)

        if node in self.leaves:
            self.leaves.remove(node)

        if node.d0 is not None or node.d1 is not None:
            warnings.warn(
                f"Node {node.id} removed with children: [{node.d0}, {node.d1}]",
                HMTWarning,
            )

    def remove_node(self, node_id):
        node = self.get_node(node_id)
        self.remove(node)

    def where(self, attr, cond):
        return self.root.where(attr, cond)

    def remove_where(self, attr, cond):
        for node in self.where(attr, cond):
            self.remove(node)

    def sum(self, attrs, func=None):
        return self.root.sum(attrs, func)

    def sum_where(self, attr_to_sum, cond_attr, cond):
        nodes = self.where(cond_attr, cond)
        return np.sum([getattr(node, attr_to_sum) for node in nodes], axis=0)

    def mean(self, attr="x"):
        return self.sum(attr) / self.xlen(attr)

    def var(self, attr="x"):
        mean = self.mean(attr)
        return self.sum(attr, func=lambda x: (x - mean) ** 2) / (self.xlen(attr) - 1)

    def cov(self, attr="x"):
        mean = self.mean(attr)
        cov = self.sum(attr, func=lambda x: np.outer(x - mean, x - mean))
        cov = np.squeeze(cov / (len(self) - 1))
        if cov.ndim == 0:
            return float(cov)
        return cov

    def apply(self, func, attr):
        """Recursively applies the function func to attr of all nodes."""
        self.root.apply(func, attr, drec=True)

    def permute_attr(self, attr_str, perm):
        attr = getattr(self, attr_str)
        setattr(self, attr_str, attr[perm])

    def normalise(self, attr="x"):
        mean = self.mean(attr)
        var = self.var(attr)
        sd = np.sqrt(var)
        self.apply(lambda x: (x - mean) / sd, attr)
        return mean, sd

    def to_numpy(self):
        """Currently only works for data that is numbered root = 1
        and for node n, d0 has id 2 * n and d1 has id 2 * n + 1
        """
        queue = [self.root]
        X = []
        while len(queue) > 0:
            node = queue.pop(0)
            # if node.mother is None:
            #     mother_id = node.id
            # else:
            #     mother_id = node.mother.id

            # if isinstance(node.x, np.ndarray):
            #     X.append((node.id, mother_id, *node.x))
            # else:
            #     X.append((node.id, mother_id, node.x))
            X.append(node.x)
            if node.d0 is not None:
                queue.append(node.d0)
            if node.d1 is not None:
                queue.append(node.d1)
        return np.array(X)


class Forest:
    def __init__(self, TreeClass, NodeClass, tree_kwargs=None, node_kwargs=None):
        self.TreeClass = TreeClass
        self.NodeClass = NodeClass
        self.tree_kwargs = tree_kwargs
        self.node_kwargs = node_kwargs
        self.trees = []

    def __len__(self):
        return len(self.trees)

    def xlen(self, attr="x"):
        return np.sum([tree.xlen(attr) for tree in self.trees], axis=0)

    def sum(self, attrs, func=None):
        """Returns the sum over all trees of attributes after appling func to them"""
        return np.sum([tree.sum(attrs, func) for tree in self.trees], axis=0)

    def sum_where(self, attr_to_sum, cond_attr, cond):
        return np.sum(
            [tree.sum_where(attr_to_sum, cond_attr, cond) for tree in self.trees],
            axis=0,
        )

    def remove_where(self, cond):
        to_remove = []
        for tree in self.trees:
            if cond(tree):
                to_remove.append(tree)
        for tree in to_remove:
            self.trees.remove(tree)

    def apply(self, func, attr):
        for tree in self.trees:
            tree.apply(func, attr)

    def mean(self, attr="x"):
        return self.sum(attr) / self.xlen(attr)

    def var(self, attr="x"):
        mean = self.mean(attr)
        return self.sum(attr, func=lambda x: (x - mean) ** 2) / (self.xlen(attr) - 1)

    def normalise(self, attr="x"):
        mean = self.mean(attr)
        var = self.var(attr)
        sd = np.sqrt(var)
        self.apply(lambda x: (x - mean) / sd, attr)
        return mean, sd

    def permute_attr(self, attr_str, perm):
        attr = getattr(self, attr_str)
        setattr(self, attr_str, attr[perm])

    def read_txt(self, filepath, sep="\t", agg_func=None):
        # Read file
        with open(filepath) as f:
            lines = f.readlines()

        # Ensure file ends in empty line
        if lines[-1] and lines[-1][-1] != "\n":
            lines[-1] += "\n"
        # Format data into array
        lines = [[x for x in line[:-1].split(sep)] for line in lines]

        # Read each line into tree structure
        missing = []
        curr_tree = None
        for line in lines:
            # print(curr_tree)
            # If there is a new tree in the data
            if line[0] == line[1]:
                # Store old tree
                if curr_tree is not None:
                    self.trees.append(curr_tree)

                # Create new tree
                curr_tree = self.TreeClass(**self.tree_kwargs)

            # if curr_tree is None:
            #     ## Data doesn't start with a root
            #     # Create empty root
            #     obs = np.array([np.nan for _ in line[2:]])
            #     root_node = self.NodeClass(str(line[1]), obs)
            #     curr_tree = self.TreeClass(root=root_node, **self.tree_kwargs)

            # Add new node
            obs = np.array([float(x) for x in line[2:]])
            if agg_func is not None:
                obs = agg_func(obs)
            new_node = self.NodeClass(str(line[0]), obs)
            try:
                # Try adding node to current tree
                curr_tree.add_node(new_node, line[1])
            except HMTError:
                pass
            else:
                # Node added so move to next line
                continue

            # Node not added so try adding to other trees
            added = False
            for tree in self.trees:
                try:
                    tree.add_node(new_node, int(line[1]))
                except HMTError:
                    # Node could not be added to tree, try next one
                    continue
                else:
                    # Node added
                    added = True
                    break
            if not added:
                missing.append(int(line[0]))
        if missing:
            warnings.warn(
                f"Nodes {missing} could not be added to any trees.", HMTWarning
            )

        # Finally add the last tree created
        self.trees.append(curr_tree)

    def to_numpy(self):
        if isinstance(self.trees[0].root.x, np.ndarray):
            X = np.full((1, self.trees[0].root.x.shape[0]), np.nan)
        else:
            X = np.full((1, 1), np.nan)
        for tree in self.trees:
            treeX = tree.to_numpy()

            X = np.row_stack((X, treeX))
        X = X[~np.isnan(X).all(axis=1)]
        return X


def read_txt(filepath, sep="\t", agg_func=None, model_type="HMT"):
    # Determine type of tree to read data into
    if model_type == "HMT":
        forest = hmt.HMForest()
    elif model_type == "Kalman":
        forest = hmt.KalmanForest()
    else:
        raise HMTError("Only supported model types are 'HMT' and 'Kalman'")
    forest.read_txt(filepath, sep, agg_func)
    return forest
