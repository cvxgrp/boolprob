"""
Copyright 2015 Enzo Busseti

This file is part of bool_joint_distr.

bool_joint_distr is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

bool_joint_distr is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with bool_joint_distr.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cvxpy

class BooleanVariables(object):
    def __init__(self, size = 1, name = None):
        self.bool_var_size = size
        self._cvxpy_var = cvxpy.Variable(rows=2**size, name=name)
        
    def __repr__(self):
        """String to recreate the object.
        """
        return "BooleanVariables(%d)" % self.bool_var_size
    
    def __getitem__(self, i):
        if not isinstance(i, (int, long)):
            raise TypeError("Index must be integer.")
        if i < 0 or i >= self.bool_var_size:
            raise IndexError("Variable index out of range.")
        return BooleanVariableIndexer(self, i)
        
class BooleanVariableIndexer(object):
    def __init__(self, bool_var, i):
        self.bool_var = bool_var
        self.i = i
        
    def __eq__(self, other):
        if not ((other is True) or (other is False) or 
            isinstance(other, BooleanVariableIndexer)):
            raise SyntaxError("Specify elementary event(s) like " +
                              "(x[0] == True) or (x[0] == x[1]).")
        if isinstance(other, BooleanVariableIndexer):
            if not self.bool_var is other.bool_var:
                raise SyntaxError("Only compare variable of the same object.")
            return (self.__eq__(True) & other.__eq__(True)) | \
                   (self.__eq__(False) & other.__eq__(False))
        result = (np.arange(2**self.bool_var.bool_var_size) >> 
                 (self.bool_var.bool_var_size - self.i - 1)) % 2
        result = ~np.array(result, dtype=bool)
        if other:
            return Event(self.bool_var, result)
        else:
            return Event(self.bool_var, ~result)
        
class Event(object):
    def __init__(self, bool_var, indicator):
        self.bool_var = bool_var
        self.indicator = indicator
        
    def _check_compatible(self, other):
        if not self.bool_var is other.bool_var:
            raise SyntaxError("Only compose events of the same variable.")
            
    def __and__(self, other):
        self._check_compatible(other)
        return Event(self.bool_var, 
                     self.indicator.__and__(other.indicator))
    
    def __or__(self, other):
        self._check_compatible(other)
        return Event(self.bool_var, 
                     self.indicator.__or__(other.indicator))    
        
    def __invert__(self):
        return Event(self.bool_var, ~self.indicator) 

def Probability(event):
    return event.bool_var._cvxpy_var.T * event.indicator

class ConditionalProbability(object):
    """Can be used in eq. or inequality constraints with a scalar."""

    def __init__(self, event, condition):
        self.event = event
        self.condition = condition
        
    def _check_compatible(self, other):
        if not np.isscalar(other):
            raise TypeError

    def __eq__(self, other):
        self._check_compatible(other)
        return Probability(self.event & self.condition) == \
            other * Probability(self.condition)

    def __le__(self, other):
        self._check_compatible(other)
        return Probability(self.event & self.condition) <= \
            other * Probability(self.condition)    
    
    def __lt__(self, other):
        return self <= other

    def __ge__(self, other):
        self._check_compatible(other)
        return Probability(self.event & self.condition) >= \
            other * Probability(self.condition) 
        
    def __gt__(self, other):
        return self >= other  
    
def maximum_entropy(bool_vars, assumptions, **kwargs):
    prob = cvxpy.Problem(
        cvxpy.Maximize(cvxpy.sum_entries(cvxpy.entr(bool_vars._cvxpy_var))),
        assumptions + 
        [cvxpy.sum_entries(bool_vars._cvxpy_var) == 1] +
        [bool_vars._cvxpy_var >= 0])
    prob.solve(**kwargs)

def minimal_distribution(bool_vars, expression, assumptions, **kwargs):
    prob = cvxpy.Problem(
        cvxpy.Minimize(expression),
        assumptions + 
        [cvxpy.sum_entries(bool_vars._cvxpy_var) == 1] +
        [bool_vars._cvxpy_var >= 0])
    return prob.solve(**kwargs)

def maximal_distribution(bool_vars, expression, assumptions, **kwargs):
    return -minimal_distribution(bool_vars, -expression, assumptions, **kwargs)

