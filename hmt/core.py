"""
Implementation of base Tree and Node class.
"""
import hmt
from hmt.exceptions import HMTError, HMTWarning
from hmt.utils import insort, is_iterable
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Callable
from numbers import Number
import warnings


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
    def __init__(self, node_id, observed=None):
        self.id = node_id
        self.x = observed
        self._path = ''
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
            return f'Node({self.id})'
        if isinstance(self.x, float):
            return f'Node({self.id}, x={round(self.x, 1)})'
        return f'Node({self.id}, x={self.x})'


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
    

    def plot(
            self,
            ax=None,
            length_attr='x',
            color_attr=None,
            x_start=0,
            y_start=0,
            dx=1,
            dy=1,
            length_func=None,
            plot_death=True,
            death_gap=1,
            plot_next=True,
            next_gap=0
            ):
        colors = ["C2", "C0", "C1", "y"]
        if ax is None:
            ax = plt.gca()
        if is_iterable(length_attr):
            assert length_func is not None
            _length_attr = length_func(
                *[getattr(self, attr) for attr in length_attr]
            )
        else:
            _length_attr = getattr(self, length_attr).copy() if length_attr is not None else 1
            # print(_length_attr)
            _length_attr = length_func(_length_attr) if length_func is not None else _length_attr
        

        # print(_length_attr, '\n')
        color = colors[getattr(self, color_attr)] if color_attr is not None else 'k'

        if isinstance(_length_attr, Number):
            if self.d1 is not None:
                ax.plot((x_start + dx * _length_attr, x_start + dx * _length_attr), (y_start, y_start + dy), c=color)
                self.d1.plot(
                    ax=ax, length_attr=length_attr, length_func=length_func, color_attr=color_attr, x_start=x_start + dx * _length_attr, y_start=y_start + dy, dx=dx, dy=dy / 2
                    )
            
            ax.plot((x_start, x_start + dx * _length_attr), (y_start, y_start), c=color)

            if self.d0 is not None:
                ax.plot((x_start + dx * _length_attr, x_start + dx * _length_attr), (y_start, y_start - dy), c=color)
                self.d0.plot(
                    ax=ax, length_attr=length_attr, length_func=length_func, color_attr=color_attr, x_start=x_start + dx * _length_attr, y_start=y_start - dy, dx=dx, dy=dy / 2
                    )
        
        if isinstance(_length_attr, np.ndarray):
            tot_length = np.sum(_length_attr[~np.isnan(_length_attr)])
            d_color = colors[0]
            if self.d1 is not None:
                ax.plot((x_start + dx * tot_length, x_start + dx * tot_length), (y_start, y_start + dy), c=d_color)
                self.d1.plot(
                    ax=ax,
                    length_attr=length_attr,
                    length_func=length_func,
                    color_attr=color_attr,
                    x_start=x_start + dx * tot_length,
                    y_start=y_start + dy,
                    dx=dx, dy=dy / 2,
                    plot_death=plot_death, death_gap=death_gap,
                    plot_next=plot_next, next_gap=next_gap
                    )
            curr_x = x_start
            for i, length in enumerate(_length_attr):
                if not np.isnan(length):
                    next_x = curr_x + dx * length
                    ax.plot((curr_x, next_x), (y_start, y_start), c=colors[i])
                    curr_x = next_x


            if self.d0 is not None:
                ax.plot((x_start + dx * tot_length, x_start + dx * tot_length), (y_start, y_start - dy), c=d_color)
                self.d0.plot(
                    ax=ax,
                    length_attr=length_attr,
                    length_func=length_func,
                    color_attr=color_attr,
                    x_start=x_start + dx * tot_length,
                    y_start=y_start - dy,
                    dx=dx, dy=dy / 2,
                    plot_death=plot_death, death_gap=death_gap,
                    plot_next=plot_next, next_gap=next_gap
                    )
            
            if self.next is not None and plot_next:
                self.next.plot(
                    ax=ax,
                    length_attr=length_attr,
                    length_func=length_func,
                    color_attr=color_attr,
                    x_start=x_start + dx * tot_length + next_gap,
                    y_start=y_start,
                    dx=dx, dy=dy,
                    plot_death=plot_death, death_gap=death_gap,
                    plot_next=plot_next, next_gap=next_gap
                    )

            if plot_death and self.d is not None and self.d:
                ax.plot(curr_x + death_gap, y_start, markersize=5, marker='x', color='red')
            
        return ax

    

    def xlen(self, attr_name='x'):
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


    # def set_tree(self, tree):
    #     self.tree = tree
    #     if self.d0 is not None:
    #         self.d0.set_tree(tree)
    #     if self.d1 is not None:
    #         self.d1.set_tree(tree)


    def get_leaves(self):
        if self.d0 is not None and self.d1 is not None:
            # Sorts using __lt__ which is based on _path attribute
            # leaves are sorted left to right by default
            return sorted(self.d0.get_leaves() + self.d1.get_leaves())
        if self.d0 is not None:
            return self.d0.get_leaves()
        if self.d1 is not None:
            return self.d1.get_leaves()
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
    
    def attr_sum(self, attrs, func=None):
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
        return total

    def sum(self, attrs, func=None):
        total = self.attr_sum(attrs, func)
        if self.d0 is not None:
            total += self.d0.sum(attrs, func)
        if self.d1 is not None:
            total += self.d1.sum(attrs, func)
        return total
    
    def sum_where(self, attrs, cond_attr, cond, func=None):
        total = self.attr_sum(attrs, func) if cond(getattr(self, cond_attr)) else 0
        if self.d0 is not None:
            total += self.d0.sum_where(attrs, cond_attr, cond, func)
        if self.d1 is not None:
            total += self.d1.sum_where(attrs, cond_attr, cond, func)
        return total

    def multiple_sum(self, attrs, func=None):
        total = self.attr_sum(attrs, func)
        if self.d0 is not None:
            total = [sum(x) for x in zip(total, self.d0.multiple_sum(attrs, func))]
        if self.d1 is not None:
            total = [sum(x) for x in zip(total, self.d1.multiple_sum(attrs, func))]

        # print(self.id, total[0])
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

    def add_daughter(self, daughter_node):
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
    
    def find_null(self, attr='x'):
        if np.isnan(getattr(self, attr)).any():
            return True
        if self.d0 is not None:
            if self.d0.find_null(attr):
                return True
        if self.d1 is not None:
            if self.d1.find_null(attr):
                return True
        return False        
    
    def null_indices(self):
        null_idxs = {tuple(np.argwhere(np.isnan(self.x)).flatten())} if np.isnan(self.x).any() else set()
        if self.d0 is not None:
            null_idxs = null_idxs.union(self.d0.null_indices())
        if self.d1 is not None:
            null_idxs = null_idxs.union(self.d1.null_indices())
        return null_idxs
    
    def to_list(self):
        curr_list = [self]
        if self.d0 is not None:
            curr_list += self.d0.to_list()
        if self.d1 is not None:
            curr_list += self.d1.to_list()
        return curr_list
    
    def set_tree_attr(self, attr, val):
        setattr(self, attr, val)
        if self.d0 is not None:
            self.d0.set_tree_attr(attr, val)
        if self.d1 is not None:
            self.d1.set_tree_attr(attr, val)
        

class Tree():
    def __init__(self):
        self.root = None
        self.leaves = []


    def __str__(self):
        if self.root is not None:
            self.root.printTree()
            return('')
        return("Tree()")


    def __len__(self):
        return len(self.root)


    def xlen(self, attr='x'):
        return self.root.xlen(attr)


    def show(self, attr_name=None, max_level=None):
        self.root.printTree(attr_name=attr_name, max_level=max_level)


    def get_node(self, node_id):
        return self.root.find(node_id)


    def get_leaf(self, node_id):
        for node in self.leaves:
            if node.id == node_id:
                return node
        raise HMTError(f'Node {node_id} not found in self.leaves.')


    def add_node(self, node, mother_id):
        # If this is the first node
        if self.root is None:
            self.root = node
            self.root.set_tree(self)
            self.leaves = node.get_leaves()
            return

        # Check if we can add the node
        try:
            mother_node = self.get_node(mother_id)
        except HMTError:
            raise HMTError(f'Mother node (node {mother_id}) not found while adding node {node.id}.')
        
        node.set_tree(self)

        if mother_node.d0 is None:
            mother_node.d0 = node
            node._path = mother_node._path + '0'
        elif mother_node.d1 is None:
            mother_node.d1 = node
            node._path = mother_node._path + '1'
        else:
            raise HMTError(f"Mother node (node {mother_id}) already has two children.")

        node.mother = mother_node

        if mother_node in self.leaves:
            self.leaves.remove(mother_node)
        for leaf in node.get_leaves():
            insort(self.leaves, leaf)
    

    def update_node(self, node_id, observed):
        node = self.get_node(node_id)
        node.x = observed

    def find_null(self, attr='x'):
        return self.root.find_null(attr)
    
    def find_unique(self, attr='x'):
        return self.root.find_unique(attr)
    

    def remove(self, node):
        if node.mother is None:
            return
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
            warnings.warn(f"Node {node.id} removed with children: [{node.d0}, {node.d1}]", HMTWarning)


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
    

    def multiple_sum(self, attrs, func=None):
        return self.root.multiple_sum(attrs, func)
    

    def sum_where(self, attrs, cond_attr, cond, func=None):
        return self.root.sum_where(attrs, cond_attr, cond, func)


    def leaf_sum(self, attr, func=None):
        if func is not None:
            return np.sum([func(getattr(leaf, attr)) for leaf in self.leaves], axis=0)
        return np.sum([getattr(leaf, attr) for leaf in self.leaves], axis=0)

    def mean(self, attr='x'):
        return self.sum(attr) / self.xlen(attr)


    def var(self, attr='x'):
        mean = self.mean(attr)
        return self.sum(attr, func=lambda x: (x - mean) ** 2) / (self.xlen(attr) - 1)
    

    def cov(self, attr='x'):
        mean = self.mean(attr)
        cov = self.sum(
            attr,
            func = lambda x: np.outer(x - mean, x - mean)
        )
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
    

    def normalise(self, attr='x'):
        mean = self.mean(attr)
        var = self.var(attr)
        sd = np.sqrt(var)
        self.apply(lambda x: (x - mean) / sd, attr)
        return mean, sd


    def to_numpy(self):
        """
        Currently only works for data that is numbered root = 1
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


    def plot(self, ax=None, length_attr=None, length_func=None, color_attr=None, x_start=0, y_start=0, dx=1, dy=1):
        return self.root.plot(ax=ax, length_attr=length_attr, length_func=length_func, color_attr=color_attr, x_start=x_start, y_start=y_start, dx=dx, dy=dy)

    def null_indices(self):
        return self.root.null_indices()
    
    def to_list(self):
        return self.root.to_list()


class Forest:
    def __init__(self):
        self._roots = []
        self.leaves = []
    

    @property
    def roots(self):
        return self._roots
    
    @roots.setter
    def roots(self, new_roots):
        # Update roots attribute
        self._roots = new_roots
        # Update leaves to match new roots
        self.get_leaves()
    

    def __len__(self):
        return sum(len(root) for root in self.roots)
    

    def xlen(self, attr='x'):
        return np.sum([root.xlen(attr) for root in self.roots], axis=0)
    

    def sum(self, attrs, func=None):
        """
        Returns the sum over all trees of attributes after appling func to them
        """
        total = 0
        for root in self.roots:
            total += root.sum(attrs, func)
        return total


    def multiple_sum(self, attrs, func=None):
        """
        Returns the sum over all trees of attributes after appling func to them
        """
        return [sum(x) for x in zip(*[root.multiple_sum(attrs, func) for root in self.roots])]
    
    def multiple_sum_where(self, attrs, cond_attr, cond, n_sums=2, func=None):
        return [sum(x) for x in zip(*[root.multiple_sum_where(attrs, cond_attr, cond, n_sums, func) for root in self.roots])]


    def show(self, attr_name=None, max_level=None, max_trees=None):
        for root in self.roots[:max_trees]:
            root.printTree(attr_name=attr_name, max_level=max_level)

    def get_leaves(self):
        self.leaves = [] # clear current leaves
        for root in self.roots:
            self.leaves += root.get_leaves()
    

    def sum_where(self, attrs, cond_attr, cond, func=None):
        total = 0
        for root in self.roots:
            total += root.sum_where(attrs, cond_attr, cond, func)
        return total
    

    def find_null(self, attr='x'):
        for root in self.roots:
            if root.find_null(attr):
                return True
        return False
    

    def leaf_sum(self, attr, func=None):
        if func is None:
            return np.sum([getattr(leaf, attr) for leaf in self.leaves], axis=0)
        return np.sum([func(getattr(leaf, attr)) for leaf in self.leaves], axis=0)
    

    def remove_where(self, cond):
        to_remove = []
        for root in self.roots:
            if cond(root):
                to_remove.append(root)
        for root in to_remove:
            self.roots.remove(root)
        self.get_leaves()
    

    def apply(self, func, attr):
        for root in self.roots:
            root.apply(func, attr, drec=True)
    

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

    def permute_attr(self, attr_str, perm):
        attr = getattr(self, attr_str)
        setattr(self, attr_str, attr[perm])

    def get_node(self, node_id):
        for root in self.roots:
            node = root.find(node_id)
            if node is not None:
                return node
    
    def remove_root(self, root):
        self.roots.remove(root)
        self.get_leaves()

    def where(self, attr, cond):
        nodes = []
        for root in self.roots:
            nodes += root.where(attr, cond)
        return nodes

    def plot(
            self,
            axes=None,
            length_attr=None,
            length_func=None,
            color_attr=None,
            x_start=0,
            y_start=0,
            dx=1, dy=1,
            n_trees=None
            ):
        if n_trees is None:
            n_trees = len(self.roots)
        if axes is None:
            _, axes = plt.subplots(n_trees, 1, sharex=True, sharey=True)
        for i, ax in enumerate(axes.flatten()):
            self.roots[i].plot(ax=ax, length_attr=length_attr, length_func=length_func, color_attr=color_attr, x_start=x_start, y_start=y_start, dx=dx, dy=dy)
        return axes
    
    def null_indices(self):
        nulls = [root.null_indices() for root in self.roots]
        return nulls[0].union(*nulls[1:])

    def null_indices_where(self, attr, cond):
        nulls = [root.null_indices_where(attr, cond) for root in self.roots]
        return nulls[0].union(*nulls[1:])
    
    def to_list(self):
        curr_list = []
        for root in self.roots:
            curr_list += root.to_list()
        return curr_list
    
    def set_forest_attr(self, attr, val):
        for root in self.roots:
            root.set_tree_attr(attr, val)