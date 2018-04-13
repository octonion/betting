#!/usr/bin/env python3

import numpy as np
import pandas as pd

# First example

# Dividend rate, 1-(track take)%
# Note D = 1/P*, where P* is the sum of implied probabilities

D = 0.80

# Win probability vector

p = np.array([0.003247,0.003247,0.003247,0.01623,0.2273,0.1623,0.5844])

# Belief probability vector
# (fraction of all track money placed on that horse)

# Note these are implied probabilities * D

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

R = 1.0
pt = 0.0
bt = 0.0
for i, row in result.iterrows():
    if (row['r']>R):
        result.at[i,'bet'] = True
        pt = pt+row['p']
        bt = bt+row['b']/D
        R = (1-pt)/(1-bt)
    else:
        break

result['f'] = 0.0
for i, row in result.iterrows():
    if (row['bet']):
        result.at[i,'f'] = row['p']-R/(D/row['b'])

# Total Kelly fraction

ip = 0
for i, row in result.iterrows():
    if (row['bet']):
        ip = ip + row['b']/D

tf = pt-(1-pt)*ip/(1-ip)
print("Total Kelly fraction = ",tf)

result['o'] = 0.0
for i, row in result.iterrows():
    if (row['bet']):
        result.at[i,'o'] = tf*(row['p']/pt)

# Easy
result['e'] = 0.0
for i, row in result.iterrows():
    if (row['bet']):
        #result.at[i,'e'] = row['p']-(row['b']/D)*(pt-tf)/ip
        result.at[i,'e'] = row['p']-(row['b']/D)*(1-pt)/(1-ip)

stg = 0.0
for i, row in result.iterrows():
    stg = stg + row['p']*np.log(1-tf+D*row['f']/row['b'])

og = 0.0
for i, row in result.iterrows():
    og = og + row['p']*np.log(1-tf+D*row['o']/row['b'])

klg = 0.0
for i, row in result.iterrows():
    if (row['bet']):
        klg = klg + row['p']*np.log(row['p']*D/row['b'])

klg = klg + (1-pt)*np.log((1-pt)/(1-ip))
print("K-L growth = ",klg)
print("S&T growth = ",stg)
print("O growth = ",og)

print()
print(result)
print()

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
        result.at[i,'bet'] = True
        pt = pt+row['p']
        bt = bt+row['b']/D
        R = (1-pt)/(1-bt)
    else:
        break

result['f'] = 0.0
for i, row in result.iterrows():
    if (row['bet']):
        result.at[i,'f'] = row['p']-R/(D/row['b'])

# Total Kelly fraction

ip = 0
for i, row in result.iterrows():
    if (row['bet']):
        ip = ip + row['b']/D

tf = pt-(1-pt)*ip/(1-ip)
print("Total Kelly fraction = ",tf)

result['o'] = 0.0
for i, row in result.iterrows():
    if (row['bet']):
        result.at[i,'o'] = tf*(row['p']/pt)

# Easy
result['e'] = 0.0
for i, row in result.iterrows():
    if (row['bet']):
        #result.at[i,'e'] = row['p']-(row['b']/D)*(pt-tf)/ip
        result.at[i,'e'] = row['p']-(row['b']/D)*(1-pt)/(1-ip)

stg = 0.0
for i, row in result.iterrows():
        stg = stg + row['p']*np.log(1-tf+D*row['f']/row['b'])

og = 0.0
for i, row in result.iterrows():
        og = og + row['p']*np.log(1-tf+D*row['o']/row['b'])

klg = 0.0
for i, row in result.iterrows():
    if (row['bet']):
        klg = klg + row['p']*np.log(row['p']*D/row['b'])

klg = klg + (1-pt)*np.log((1-pt)/(1-ip))
print("K-L growth = ",klg)
print("S&T growth = ",stg)
print("O growth = ",og)

print()
print(result)
