#!/usr/bin/env python

import numpy as np
import pandas as pd

# First example

# Win probability vector

p = np.array([0.25,0.50,0.25])

# Decimal odds

d = np.array([5.0,1.5,2.0])

# Implied probabilities

i = np.array([])
for j in range(len(d)):
    i = np.append(i,1/d[j])

race = pd.DataFrame()

race['p'] = p
race['d'] = d
race['i'] = i

r = np.array([])
for j in range(len(p)):
    r = np.append(r,p[j]*d[j])

race['r'] = r

result = race
result['bet'] = False

R = 1.0
pt = 0.0
it = 0.0
while True:
    found = False
    for j, row in result.iterrows():
# Equivalent
#        if (row['r']>(1-pt-row['p'])/(1-it-row['i'])) and not(row['bet']):
        if (row['r']>R) and not(row['bet']):
            result.at[j,'bet'] = True
            pt = pt+row['p']
            it = it+row['i']
            R = (1-pt)/(1-it)
            found = True
            break
    if not(found):
        break

#R = (1-pt)/(1-it)

result['f'] = 0.0
for j, row in result.iterrows():
    if (row['bet']):
        result.at[j,'f'] = row['p']-R*row['i']

print(result)
