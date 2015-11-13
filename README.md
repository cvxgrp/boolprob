# Boolopt
*A Python tool to analyze joint distribution of boolean random variables.*

### Description
Boolopt allows you to study properties of boolean joint distribution, 
by specifying *assumptions*, computing *worst case* probabilities, 
and much more.

### Basic usage

```python
from boolopt import JointDistr, Probability, CondProbability

# define a joint distribution of 4 boolean random variables
joint_distr = JointDistr(4)

# assume the 4 variables are the default of 4 companies
A_default, B_default, C_default, D_default = \
  joint_distr.get_variables()
  
# define assumptions
assumptions = [Probability(A_default) == .1,
               Probability(B_default) == .1,
               Probability(C_default) == .1,
               Probability(D_default) == .1,
               Probability(A_default & B_default) == 0,
               CondProbability(C_default, B_default | D_default) == .5]

# find the maximum entropy distribution
joint_distr.maximum_entropy(assumptions)

print "Under the maximum entropy distribution, ",
print "the probability that at least one company defaults is %.3f." % \
  Probability(A_default | B_default | C_default | D_default).value
```
