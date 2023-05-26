from scipy.stats import pointbiserialr
from math import sqrt
import pandas as pd
import numpy as np

def get_merit(df, subset, label):
    """Calculates feature merits"""
    k = len(subset)

    # average feature-class correlation
    rcf_all = []
    for feature in subset:
        coeff = pointbiserialr( df[label], df[feature] )
        rcf_all.append( abs( coeff.correlation ) )
    rcf = np.mean( rcf_all )

    # average feature-feature correlation
    corr = df[subset].corr()
    corr.values[np.tril_indices_from(corr.values)] = np.nan
    corr = abs(corr)
    rff = corr.unstack().mean()

    return (k * rcf) / sqrt(k + k * (k-1) * rff)

class PriorityQueue:
    """Class that represents a quee"""
    def  __init__(self):
        self.queue = []

    def is_empty(self):
        return len(self.queue) == 0
    
    def push(self, item, priority):
        """
        item already in priority queue with smaller priority:
        -> update its priority
        item already in priority queue with higher priority:
        -> do nothing
        if item not in priority queue:
        -> push it
        """
        for index, (i, p) in enumerate(self.queue):
            if (set(i) == set(item)):
                if (p >= priority):
                    break
                del self.queue[index]
                self.queue.append( (item, priority) )
                break
        else:
            self.queue.append( (item, priority) )
        
    def pop(self):
        """pop element on queu"""
        # return item with highest priority and remove it from queue
        max_idx = 0
        for index, (i, p) in enumerate(self.queue):
            if (self.queue[max_idx][1] < p):
                max_idx = index
        (item, priority) = self.queue[max_idx]
        del self.queue[max_idx]
        return (item, priority)
    
def main(df, features, label):
    """Calculates cfs"""
    best_value = -1
    best_feature = ''
    for feature in features:
        coeff = pointbiserialr( df[label], df[feature] )
        abs_coeff = abs( coeff.correlation )
        if abs_coeff > best_value:
            best_value = abs_coeff
            best_feature = feature

    # initialize queue
    queue = PriorityQueue()
    # push first tuple (subset, merit)
    queue.push([best_feature], best_value)
    # list for visited nodes
    visited = []
    # counter for backtracks
    n_backtrack = 0
    # limit of backtracks
    max_backtrack = 5
    # repeat until queue is empty
    #or the maximum number of backtracks is reached
    while not queue.is_empty():
        # get element of queue with highest merit
        subset, priority = queue.pop()
        
        # check whether the priority of this subset
        # is higher than the current best subset
        if (priority < best_value):
            n_backtrack += 1
        else:
            best_value = priority
            best_subset = subset

        # goal condition
        if (n_backtrack == max_backtrack):
            break
        
        # iterate through all features and look of one can
        # increase the merit
        for feature in features:
            temp_subset = subset + [feature]
            
            # check if this subset has already been evaluated
            for node in visited:
                if (set(node) == set(temp_subset)):
                    break
            # if not, ...
            else:
                # ... mark it as visited
                visited.append( temp_subset )
                # ... compute merit
                merit = get_merit(df, temp_subset, label)
                # and push it to the queue
                queue.push(temp_subset, merit)
    return best_subset