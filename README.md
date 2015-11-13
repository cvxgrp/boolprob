# Boolprob
Boolprob is a tool to analyze joint probability distributions of boolean random variables.
The user can specify *assumptions* about the distribution, compute *worst case* probabilities, 
and much more.

### Basic usage

```python
from boolprob import JointDistr, Probability, CondProbability

# define a joint distribution of 4 boolean random variables
joint_distr = JointDistr(4)

# the 4 random variables are the defaults of 4 companies
A_default, B_default, C_default, D_default = \
  joint_distr.get_variables()
  
# define assumptions, you can use logic operators &, |, ~
assumptions = [Probability(A_default) == .1,
               Probability(B_default) == .1,
               Probability(C_default) == .1,
               Probability(D_default) == .1,
               Probability(A_default | B_default) == .15,
               CondProbability(C_default, B_default & D_default) == .5]

# find the maximum entropy distribution
joint_distr.maximum_entropy(assumptions)

print "Under the maximum entropy distribution, ",
print "the probability that at least one company defaults is %.3f." % \
  Probability(A_default | B_default | C_default | D_default).value
```
