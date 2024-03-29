from random import randint
import math
import collections
import matplotlib.pyplot as plt
import numpy as np

# distributions of each of the hypotheses
h1 = { 'cherry': 1.0, 'lime': 0.0 }
h2 = { 'cherry': 0.75, 'lime': 0.25 }
h3 = { 'cherry': 0.5, 'lime': 0.5 }
h4 = { 'cherry': 0.25, 'lime': 0.75 }
h5 = { 'cherry': 0.0, 'lime': 1.0 }

# Original ditribution of the hypotheses from the textbook
p_hypotheses = [0.1, 0.2, 0.4, 0.2, 0.1]

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
        _p_cond = p_cond(D, hypotheses[i])
        if p_hi == 0 or _p_cond == 0:
            # Depending on the dataset, probabilities can sometimes go to 0,
            # which breaks the log method. Don't let the happen.
            continue
        # math.log accepts a base as the second argument
        print "made it here"
        MAP_val = -math.log(_p_cond, 2) - math.log(p_hi, 2)
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

for i in range(1,101):
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
p_h3_matrix = [p_hypotheses]
p_h4_matrix = [p_hypotheses]

hypotheses = [h1, h2, h3, h4, h5] # for mapping and indexing

h3_part_i, h3_part_ii, h3_part_iii, h3_part_iv = [], [], [], []
h4_part_i, h4_part_ii, h4_part_iii, h4_part_iv = [], [], [], []
for n in range(1,101):
    # Compute the P(hi|D[0:n]) for h1..h5. This is part i
    prev_h3_row = p_h3_matrix[n - 1]
    next_h3_row = []
    for i in range(0,5):
        p_hi = prev_h3_row[i]
        denominator = 0.0
        for k in range(0,5): # I hate everything
            denominator += p_cond(h3_data[0:n], hypotheses[k]) * prev_h3_row[k]
        p_hi_given_d = (p_cond(h3_data[0:n], hypotheses[i]) * p_hi) / denominator
        next_h3_row.append(p_hi_given_d)
    p_h3_matrix.append(next_h3_row)
    
    prev_h4_row = p_h4_matrix[n - 1]
    next_h4_row = []
    for i in range(0,5):
        denominator = 0.0
        for k in range(0,5):
            denominator += p_cond(h4_data[0:n], hypotheses[k]) * prev_h4_row[k]
        p_hi = prev_h4_row[i]
        p_hi_given_d = (p_cond(h4_data[0:n], hypotheses[i]) * p_hi) / denominator
        next_h4_row.append(p_hi_given_d)
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
    print p_h3_matrix[n]
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


# Graph for i for h3:
# Graph for ii, iii, and iv for h3:
plt.plot(range(0, 100), h4_part_ii, label = 'part ii')
plt.plot(range(0,100),h4_part_iii, label = 'part iii')
plt.plot(range(0,100),h4_part_iv, label = 'part iv')
plt.legend()
plt.show()

# Graph for i for h4:
#plt.plot(range(0,100), i)

# Graph for ii, iii, iv for h4:
#plt.plot(range(0, 100), h3_part_ii)
#plt.plot(range(0,100),h3_part_iii)
#plt.plot(range(0,100),h3_part_iv)
#plt.show()

