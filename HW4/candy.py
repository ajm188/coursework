from random import randint

#Hypothesis 3: 50% Cherry, 50% Lime
#Hypothesis 4: 25% Cherry, 75% Lime

h3_data = []
h4_data = []

for i in range(0,100):
	candidate = randint(0,100)
	if candidate < 50:
		h3_data.append("cherry")
	else:
		h3_data.append("lime")
	if candidate < 25:
		h4_data.append("cherry")
	else:
		h4_data.append("lime")
	