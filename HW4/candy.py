from random import randint
import math
import collections

# distributions of each of the hypotheses
h1 = { 'cherry': 1.0, 'lime': 0.0 }
h2 = { 'cherry': 0.75, 'lime': 0.25 }
h3 = { 'cherry': 0.5, 'lime': 0.5 }
h4 = { 'cherry': 0.25, 'lime': 0.75 }
h5 = { 'cherry': 0.0, 'lime': 1.0 }

# Original ditribution of the hypotheses from the textbook
p_hypotheses = collections.OrderedDict({ h1: 0.1, h2: 0.2, h3: 0.4, h4: 0.2, h5: 0.1 })

def _num_possible_datasets(size):
    """
    Return the number of possible datasets of the given size.
    This is the sum from 0 to size of the number ways to choose k of the elements to be 'cherry'.
    """
    return reduce((lambda acc, el: acc + C(size, el)), range(0,size), 0)

def C(n, k):
    """Return 'n choose k'."""
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
    candidate = randint(0,100)
    if candidate < 25:
        h4_data.append("cherry")
    else:
        h4_data.append("lime")

# Need one matrix for the h3 dataset and another for the h4 dataset
p_h3_matrix = [p_hypotheses.values()]
p_h4_matrix = [p_hypotheses.values()]

hypotheses = p_hypotheses.keys() # for mapping and indexing

h3_part_i, h3_part_ii, h3_part_iii, h3_part_iv = [], [], [], []
h4_part_i, h4_part_ii, h4_part_iii, h4_part_iv = [], [], [], []
for n in range(1,101):
    # Compute the P(hi|D[0:n]) for h1..h5. This is part i
    next_h3_row = map((lambda el: p_cond(h3_data[0:n], el)), hypotheses)
    p_h3_matrix.append(next_h3_row)
    
    next_h4_row = map((lambda el: p_cond(h4_data[0:n], el)), hypotheses)
    p_h4_matrix.append(next_h4_row)

# more loops! Monkeys and Wabbits Loop de loop
for n in range(0,100):
    # Part ii
    # P(Dn+1 = lime | d1,...,dn) = sum i = 1 to 5 p(dn+1 = lime | hi) * p(hi|d)
    # That last part of the product is what's in our matrices.
    
    # h3 case:
    h3_sum = 0
    h3_row = p_h3_matrix[n]
    for i in range(0,5):
        p_hi = h3_row[i] # this should be a probability
        h3_sum += p_hi * hypotheses[i]['lime']
    h3_part_ii.append(h3_sum)
    
    # h4 case:
    h4_sum = 0
    h4_row = p_h4_matrix[n]
    for i in range(0,5):
        p_hi = h4_row[i]
        h4_sum += p_hi * hypotheses[i]['lime']
    h4_part_ii.append(h4_sum)

for n in range(0,100):
    # part iii
    
    # h3 case:
    d = h3_data[0:n]
    row = p_h3_matrix[n]
    current_h3_MAP, current_MAP_max = None, None
    for i in range(3,5):
        p_hi = row[i]
        # only look at hypotheses 3 and 4
        # math.log accepts a base as the second argument (thank god)
        MAP_val = -math.log(p_cond(d, hypotheses[i]), 2) - math.log(p_hi, 2)
        if current_MAP_max is None or current_MAP_max < MAP_val:
            current_h3_MAP = hypotheses[i]
    # Now that we have a MAP hypothesis, we can compute P(d_n+1 = 'lime' | hMAP)
    
    