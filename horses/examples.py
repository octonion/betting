#!/usr/bin/env python

import numpy as np
import pandas as pd

# First example

# Divident rate, 1-(track take)%

D = 0.80

# Win probability vector

p = np.array([0.003247,0.003247,0.003247,0.01623,0.2273,0.1623,0.5844])

# Belief probability vector
# (fraction of all track money placed on that horse)

b = np.array([0.025,0.0375,0.0625,0.125,0.25,0.3125,0.1875])

race = pd.DataFrame()

race['p'] = p
race['b'] = b

r = np.array([])
for i in range(len(p)):
    r = np.append(r,p[i]*(D/b[i]))

race['r'] = r

result = race.sort_values(by=['r'], ascending=[False])
result['bet'] = False

#for i, row in result.iterrows():
#    result.set_value(i, 'bet', True)
#print(result)
    
#for i in result.index:
#    result.set_value(i, 'bet', False)
#print(result)

R = 1.0
pt = 0.0
bt = 0.0
for i, row in result.iterrows():
    if (row['r']>R):
        result.set_value(i, 'bet', True)
        pt = pt+row['p']
        bt = bt+row['b']/D
        R = (1-pt)/(1-bt)
    else:
        break

print(result)

# Second example

D = 0.85

p = np.array([0.25,0.1,0.1,0.4,0.15])

b = np.array([0.17,0.05667,0.034,0.34,0.3993])

race = pd.DataFrame()

race['p'] = p
race['b'] = b

r = np.array([])
for i in range(len(p)):
    r = np.append(r,p[i]*(D/b[i]))

race['r'] = r

result = race.sort_values(by=['r'], ascending=[False])

result['bet'] = False

R = 1.0
pt = 0.0
bt = 0.0
for i, row in result.iterrows():
    if (row['r']>R):
        result.set_value(i, 'bet', True)
        pt = pt+row['p']
        bt = bt+row['b']/D
        R = (1-pt)/(1-bt)
    else:
        break

print(result)
