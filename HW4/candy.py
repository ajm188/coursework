from random import randint
import math

# distributions of each of the hypotheses
h1 = { 'cherry': 1.0, 'lime': 0.0 }
h2 = { 'cherry': 0.75, 'lime': 0.25 }
h3 = { 'cherry': 0.5, 'lime': 0.5 }
h4 = { 'cherry': 0.25, 'lime': 0.75 }
h5 = { 'cherry': 0.0, 'lime': 1.0 }

# Original ditribution of the hypotheses from the textbook
p_hypotheses = { h1: 0.1, h2: 0.2, h3: 0.4, h4: 0.2, h5: 0.1 }

def _num_possible_datasets(size):
    """
    Return the number of possible datasets of the given size.
    This is the sum from 0 to size of the number ways to choose k of the elements to be 'cherry'.
    """
    return reduce((lambda acc, el: acc + C(size, el)), range(0,size), 0)

def C(n, k):
    """Return 'n choose k'. TODO: Memoize it to make it faster."""
    return math.factorial(n) / math.factorial(k) / math.factorial(n - k)

def p_cond(D, h):
    """
    Return the conditional probability of a dataset D given a hypothesis h.

    For this problem, the elements of D are statistically independent, so we can
    just take the product of the conditional probabilities of each element in D.
    """
    return reduce((lambda acc, el: acc * h[el]), D, 1)

def p_hypothesis(h):
    """
    Return the probability of the given hypothesis.
    Look up the probability of the hypothesis from the global hash.
    """
    return p_hypotheses[h]

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

h3_part_i, h3_part_ii, h3_part_iii, h3_part_iv = [], [], [], []
h4_part_i, h4_part_ii, h4_part_iii, h4_part_iv = [], [], [], []
for n in range(1,100):
    # Parts i through iv for hypothesis 3
    h3_part_i.append(p_cond(h3_data[0:n], p_given_h3)))


    G3.append(p_cond(h3_data[0:n], p_given_h3))
    G4.append(p_cond(h4_data[0:n], p_given_h4))
