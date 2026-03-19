"""
Implementation of base Node class and Forest class.
"""
from hmt.exceptions import HMTError
from hmt.utils import is_iterable
import numpy as np


class Node():
    """
    Attributes
    ----------

    _path: str
        Binary representation of where the node is on the tree, 0 => left and 1 => right.
        e.g. 
               1      1 = 0     4 = 000
             /   \    2 = 00    5 = 001
            2     3   3 = 01    6 = 010
           / \   /
          4   5 6
    """
    def __init__(self, node_id, observed=None, d0=None, d1=None):
        """Note d0 and d1 are not used, only in test cases"""
        self.id = node_id
        self.x = observed #TODO remove
        self._path = ''
        self.mother = None
        self.d0, self.d1 = d0, d1
        if self.d0 is not None:
            self.d0.mother = self
        if self.d1 is not None:
            self.d1.mother = self
        self.next, self.prev = None, None
    
    def __repr__(self):
        return f'Node({self.id})'

    def __lt__(self, other):
        return self._path < other._path
    
    def preorder(self):
        """Iterates over the tree with depth-first pre-order traversal"""
        yield self
        if self.d0 is not None:
            yield from self.d0.preorder()
        if self.d1 is not None:
            yield from self.d1.preorder()

    def postorder(self):
        """Iterates over the tree with depth-first post-order traversal"""
        if self.d0 is not None:
            yield from self.d0.postorder()
        if self.d1 is not None:
            yield from self.d1.postorder()
        yield self
    
    def depth(self):
        """Returns the level of the node in the tree."""
        return len(self.path)
    
    def apply(self, func, attr):
        """Apply function func to attribute attr."""
        setattr(self, attr, func(getattr(self, attr)))
    
    def attr_sum(self, attrs, func=None):
        """Deals with differnt types of sums needed"""
        if isinstance(attrs, str):
            attr = getattr(self, attrs)
            if attr is None:
                summand = 0
            elif func is None:
                summand = 1.0 * attr
            else:
                summand = func(attr)
        elif isinstance(attrs, tuple) or isinstance(attrs, list):
            attr_list = [getattr(self, attr) for attr in attrs]
            
            if func is None:
                summand = sum(attr_list)
            else:
                summand = func(*attr_list)
        
        if isinstance(summand, np.ndarray):
            summand[np.isnan(summand)] = 0
        else:
            summand = summand or 0
        return summand

    def multiple_sum(self, attrs, func=None):
        total = self.attr_sum(attrs, func)
        if self.d0 is not None:
            total = [sum(x) for x in zip(total, self.d0.multiple_sum(attrs, func))]
        if self.d1 is not None:
            total = [sum(x) for x in zip(total, self.d1.multiple_sum(attrs, func))]
        return total
    
    def multiple_sum_where(self, attrs, cond_attr, cond, n_sums=2, func=None):
        total = self.attr_sum(attrs, func) if cond(getattr(self, cond_attr)) else [0] * n_sums
        if self.d0 is not None:
            d0_total = self.d0.multiple_sum_where(attrs, cond_attr, cond, n_sums, func)
            total = [sum(x) for x in zip(total, d0_total)]
        if self.d1 is not None:
            d1_total = self.d1.multiple_sum_where(attrs, cond_attr, cond, n_sums, func)
            total = [sum(x) for x in zip(total, d1_total)]
        return total

    def add_daughter(self, daughter_node):
        raise HMTError("Testing if actually used")
        if self.d0 is not None:
            # Make daughter node d1
            if self.d1 is not None:
                raise HMTError(
                    f"Cannot add more than two daughters to node {self.id}" +
                    f"with current daughters {self.d0.id}, {self.d1.id}"
                    )
            daughter_node.mother = self
            daughter_node._path = self._path + "1"
            self.d1 = daughter_node
            return
        daughter_node.mother = self
        daughter_node._path = self._path + "0"
        self.d0 = daughter_node

    def remove_where(self, cond):
        # NOTE this doesn't actually delete things, use del
        if cond(self):
            if int(self._path[-1]):
                self.mother.d1 = None
            else:
                self.mother.d0 = None
            return
        if self.d0 is not None:
            self.d0.remove_where(cond)
        if self.d1 is not None:
            self.d1.remove_where(cond)
        # del self

    def printTree(self, level=0, attr_name=None, arrow='\u2B62 ', max_level=None):
        if max_level is not None and level == max_level:
            return
        if self.d1 is not None or (self.next is not None and self.next.d1 is not None):
            d1 = self.next.d1 if self.next is not None else self.d1
            d1.printTree(level + 1, attr_name, arrow='\u2BA3 ', max_level=max_level)
        if attr_name is not None:
            attr = getattr(self, attr_name)
            
            if isinstance(attr, float):
                attr = round(attr, 2)
            strlen = len(str(attr))
            if isinstance(attr, np.ndarray):
                strlen = 2 + attr.shape[0] * 5
            if self.next is None:
                with np.printoptions(precision=2, suppress=True):
                    print(
                        ' ' * (11 + strlen + len(attr_name)) * level + arrow + 
                        f'{self.id} ({attr_name} = {attr})'
                        )
            else:
                next_attr = getattr(self.next, attr_name)
                if isinstance(next_attr, float):
                    next_attr = round(next_attr, 2)
                with np.printoptions(precision=2, suppress=True):
                    print(
                        ' ' * (11 + strlen + len(attr_name)) * level + arrow + 
                        f'{self.id} ({attr_name} = {attr} \u2B62 {next_attr})'
                        )
        else:
            if self.next is None:
                print(' ' * 5 * level + arrow + str(self.id))
            else:
                print(' ' * 5 * level + arrow + str(self.id) + "|" + str(self.id))
        if self.d0 is not None or (self.next is not None and self.next.d0 is not None):
            d0 = self.next.d0 if self.next is not None else self.d0
            d0.printTree(level + 1, attr_name, arrow='\u2BA1  ', max_level=max_level)

class Forest:
    def __init__(self):
        self.roots = []
        self.leaves = []

    def preorder(self):
        """Iterates over all trees with pre-order traversal."""
        for root in self.roots:
            yield from root.preorder()

    def postorder(self):
        """Iterates over all trees with post-order traversal."""
        for root in self.roots:
            yield from root.postorder()
    
    def to_list(self):
        """Returns a list of all nodes in order of pre-traversal."""
        return [node for node in self.preorder()]
    

    def __len__(self):
        return len(self.to_list())

    def get_leaves(self):
        """Updates leaves attribute to be list of nodes without any children."""
        self.leaves = [node for node in self.preorder() if node.d0 is None and node.d1 is None]
    
    def get_node(self, node_id):
        """Returns node object given node_id."""
        for node in self.preorder():
            if node.id == node_id:
                return node
        return None

    def max(self, attr):
        """Returns maximum value of attribute over whole forest."""
        return max(getattr(node, attr) for node in self.preorder())

    def apply(self, func, attr):
        """Applies function to attribute over the whole forest."""
        for node in self.preorder():
            node.apply(func, attr)
    
    def set_forest_attr(self, attr, val):
        """Sets attribute attr to val for every node in forest."""
        for node in self.preorder():
            setattr(node, attr, val)
        
    def where(self, attr, cond):
        """Returns a list of nodes where attr satisfies cond."""
        nodes = []
        for node in self.preorder():
            if is_iterable(attr):
                attrs = [getattr(node, attr_name) for attr_name in attr]
                if cond(*attrs):
                    nodes.append(node)
            else:
                if cond(getattr(node, attr)):
                    nodes.append(node)
        return nodes
    
    def xlen(self, attr_name='x'):
        """Returns the number of nodes for which attr is not nan."""
        total = 0
        for node in self.preorder():
            attr = getattr(node, attr_name)
            if isinstance(attr, np.ndarray):
                total += 1 * ~np.isnan(attr)
            elif attr is not None and ~np.isnan(attr):
                total += 1
        return total
    
    def has_null(self, attr='x'):
        """Detects whether null values are present in forest"""
        for node in self.preorder():
            if np.isnan(getattr(node, attr)).any():
                return True
        return False

    def sum(self, attrs, func=None):
        """Sums attributes over forest"""
        return sum(node.attr_sum(attrs, func) for node in self.preorder())

    def sum_where(self, attrs, cond_attr, cond, func=None):
        """Sums func(attrs) wherever cond(cond_attr) is True"""
        total = 0
        for node in self.preorder():
            if cond(getattr(node, cond_attr)):
                total += node.attr_sum(attrs, func)
        return total
    
    def leaf_sum(self, attr, func=None):
        """Sums func(attr) over leaf nodes"""
        if func is None:
            return np.sum([getattr(leaf, attr) for leaf in self.leaves], axis=0)
        return np.sum([func(getattr(leaf, attr)) for leaf in self.leaves], axis=0)
    
    def null_indices(self):
        """Finds unique values of indidces where x is nan."""
        null_idxs = set()
        for node in self.preorder():
            if np.isnan(node.x).any():
                null_idxs = null_idxs.union({tuple(np.argwhere(np.isnan(self.x)).flatten())})
        return null_idxs
    
    def permute_attr(self, attr_str, perm):
        """Permutes an attribute of the tree."""
        attr = getattr(self, attr_str)
        setattr(self, attr_str, attr[perm])
    
    ####################################
    ##### Not refactored from here #####
    ####################################

    def multiple_sum(self, attrs, func=None):
        return [sum(x) for x in zip(*[root.multiple_sum(attrs, func) for root in self.roots])]
    
    def multiple_sum_where(self, attrs, cond_attr, cond, n_sums=2, func=None):
        return [sum(x) for x in zip(*[root.multiple_sum_where(attrs, cond_attr, cond, n_sums, func) for root in self.roots])]

    def show(self, attr_name=None, max_level=None, max_trees=None):
        for root in self.roots[:max_trees]:
            root.printTree(attr_name=attr_name, max_level=max_level)    

    def remove_where(self, cond):
        to_remove = []
        for root in self.roots:
            if cond(root):
                to_remove.append(root)
        for root in to_remove:
            self.roots.remove(root)

    def mean(self, attr='x'):
        return self.sum(attr) / self.xlen(attr)

    def var(self, attr='x'):
        mean = self.mean(attr)
        return self.sum(attr, func=lambda x: (x - mean) ** 2) / (self.xlen(attr) - 1)

    def normalise(self, attr='x'):
        mean = self.mean(attr)
        var = self.var(attr)
        sd = np.sqrt(var)
        self.apply(lambda x: (x - mean) / sd, attr)
        return mean, sd

    def null_indices_where(self, attr, cond):
        raise HMTError("Checking if deprecated")
        nulls = [root.null_indices_where(attr, cond) for root in self.roots]
        return nulls[0].union(*nulls[1:])