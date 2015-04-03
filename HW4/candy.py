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

def p_cond(D, h):
    """
    Return the conditional probability of a dataset D given a hypothesis h.

    For this problem, the elements of D are statistically independent, so we can
    just take the product of the conditional probabilities of each element in D.
    """
    return reduce((lambda acc, el: acc * h[el]), D, 1)

def compute_MAP_hypothesis(D, hypotheses, p_vector):
    """
    Return the MAP hypothesis for the dataset D.
    p_vector is the vector of probabilities of each hypothesis.
    For the purposes of this assignment, only consider h3 and h4.
    """
    MAP, MAP_max = None, None
    for i in range(3,5):
        p_hi = p_vector[i]
        # math.log accepts a base as the second argument
        MAP_val = -math.log(p_cond(D, hypotheses[i]), 2) - math.log(p_hi, 2)
        if MAP_max is None or MAP_max < MAP_val:
            MAP = hypotheses[i] # this can be modified to return just the index, if need be.
    return MAP

def compute_ML_hypothesis(D, hypotheses):
    """
    Return the ML hypothesis for the dataset D.
    The ML hypothesis is the hypothesis which maximizes P(D|hi).
    For the puposes of this assignment, only consider h3 and h4.
    """
    ML, ML_max = None, None
    for i in range(3,5):
        ML_val = p_cond(D, hypotheses[i])
        if ML_max is None or ML_max < ML_val:
            ML = hypotheses[i]
    return ML

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
    h3_MAP = compute_MAP_hypothesis(h3_data[0:n], hypotheses, p_h3_matrix[n])
    # Now that we have a MAP hypothesis, we can compute P(d_n+1 = 'lime' | hMAP)
    h3_part_iii.append(h3_MAP['lime'])
    
    # h4 case:
    h4_MAP = compute_MAP_hypothesis(h4_data[0:n], hypotheses, p_h4_matrix[n]) 
    h4_part_iii.append(h4_MAP['lime'])

for n in range(0,100):
    # part iv

    # h3 case:
    h3_ML = compute_ML_hypothesis(h3_data[0:n], hypotheses)
    # Now that we have a ML hypothesis, we can compute P(d_n+1 = 'lime' | hML)
    h3_part_iv.append(h3_ML['lime'])

    # h4 case:
    h4_ML = compute_ML_hypothesis(h4_data[0:n], hypotheses)
    h4_part_iv.append(h4_ML['lime'])
