from random import randint
import math

def _num_possible_datasets(size):
    """
    Return the number of possible datasets of the given size.
    This is the sum from 0 to size of the number ways to choose k of the elements to be 'cherry'.
    """
    return reduce((lambda acc, el: acc + C(size, el)), range(0,size), 0)

def C(n, k):
    """Return 'n choose k'. TODO: Memoize it to make it faster."""
    return math.factorial(n) / math.factorial(k) / math.factorial(n - k)

def p_given_h3(d):
    """
    Return the conditional probability of d given h3.
    Returns None if d is not in ['cherry', 'lime'].
    """
    if d == 'cherry':
        return 0.5
    elif d == 'lime':
        return 0.5
    else:
        return None

def p_given_h4(d):
    """
    Return the conditional probability of d given h4.
    Returns None if d is not in ['cherry', 'lime'].
    """
    if d == 'cherry':
        return 0.25
    elif d == 'lime':
        return 0.75
    else:
        return None

def p_cond(D, h):
    """
    Return the conditional probability of a dataset D given a hypothesis h.

    For this problem, the elements of D are statistically independent, so we can
    just take the product of the conditional probabilities of each element in D.
    """
    return reduce((lambda acc, el: acc * h(el)), D, 1)

def p_hypothesis(h):
    """
    Return the probability of the given hypothesis.
    For the purposes of this problem, there are 5 possible hypotheses, so this is always 1/5
    """
    return 1.0 / 5

def p_dataset(D):
    """
    Return the probability of the given dataset.
    Computed as 1 over the number of unique datasets of the same size as D.
    """
    return 1.0 / _num_possible_datasets(len(D))

#Hypothesis 3: 50% Cherry, 50% Lime
#Hypothesis 4: 25% Cherry, 75% Lime

h3_data = []
h4_data = []

for i in range(1,100):
	candidate = randint(0,100)
	if candidate < 50:
		h3_data.append("cherry")
	else:
		h3_data.append("lime")
	if candidate < 25:
		h4_data.append("cherry")
	else:
		h4_data.append("lime")

G3, G4 = [], []
for n in range(1,100):
    G3.append(p_cond(h3_data[0:n], p_given_h3))
    G4.append(p_cond(h4_data[0:n], p_given_h4))
