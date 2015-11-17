"""
Copyright 2015 Enzo Busseti

This file is part of Boolprob.

Boolprob is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Boolprob is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Boolprob.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cvxpy


class JointDistr(object):
    def __init__(self, size):
        self._bool_var_size = size
        self._cvxpy_var = cvxpy.Variable(rows=2**size)
        
    def __repr__(self):
        """String to recreate the object."""
        return "JointDistr(%d)" % self._bool_var_size

    def get_variables(self):
        """Return an Event object for each random variable."""
        indicators = [(np.arange(2**self._bool_var_size) >> 
                       (self._bool_var_size - i - 1)) % 2
                      for i in range(self._bool_var_size)]
        return [Event(self, ~np.array(indicator, dtype=bool)) 
                for indicator in indicators]

    def maximum_entropy(self, assumptions,  **kwargs):
        prob = cvxpy.Problem(
            cvxpy.Maximize(cvxpy.sum_entries(cvxpy.entr(self._cvxpy_var))),
                assumptions + 
                [cvxpy.sum_entries(self._cvxpy_var) == 1] +
                [self._cvxpy_var >= 0])
        prob.solve(**kwargs)

    def minimal_distribution(self, expression, assumptions, **kwargs):
        prob = cvxpy.Problem(
            cvxpy.Minimize(expression),
            assumptions + 
            [cvxpy.sum_entries(self._cvxpy_var) == 1] +
            [self._cvxpy_var >= 0])
        return prob.solve(**kwargs)

    def maximal_distribution(self, expression, assumptions, **kwargs):
        return -self.minimal_distribution(-expression, assumptions, **kwargs)


class Event(object):
    def __init__(self, bool_var, indicator):
        self.bool_var = bool_var
        self._indicator = indicator
        
    def _check_compatible(self, other):
        """Helper function when doing logic b/w events."""
        if not self.bool_var is other.bool_var:
            raise SyntaxError("Events don't belong to the same distribution.")
            
    def __and__(self, other):
        self._check_compatible(other)
        return Event(self.bool_var, 
                     self._indicator.__and__(other._indicator))
    
    def __or__(self, other):
        self._check_compatible(other)
        return Event(self.bool_var, 
                     self._indicator.__or__(other._indicator))    
        
    def __invert__(self):
        return Event(self.bool_var, ~self._indicator) 

    def __eq__(self, other):
        self._check_compatible(other)
        return Event(self.bool_var, 
                     self._indicator.__eq__(other._indicator))    


def Probability(event):
    """Probability of event. """
    return event.bool_var._cvxpy_var.T * event._indicator


class CondProbability(object):
    """Conditional probability for eq. or ineq. constraints (with a scalar)."""

    def __init__(self, event, condition):
        self.event = event
        self.condition = condition
        
    def _check_compatible(self, other):
        if not np.isscalar(other):
            raise SyntaxError("Only CondProbability(event, cond) <=> scalar.")

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

    @property
    def value(self):
        numerator = Probability(self.event & self.condition).value
        denominator = Probability(self.condition).value
        if (numerator is None) and (denominator is None):
            return None
        else:
            return numerator / denominator
