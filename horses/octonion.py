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

# Implied probabilities are most natural

ip = b/D

race = pd.DataFrame()

race['p'] = p
race['ip'] = ip
race['r'] = race['p']/race['ip']

race = race.sort_values(by=['r'], ascending=[False])

race['bet'] = False

p_total = 0.0
ip_total = 0.0
for i, row in race.iterrows():
    # Must be a positive hedge
    if (row['p'] > row['ip']*(1-p_total)/(1-ip_total)):
        race.at[i,'bet'] = True
        p_total = p_total + row['p']
        ip_total = ip_total + row['ip']
    else:
        break

# Fractions as per binary Kelly

race['f'] = 0.0
for i, row in race.iterrows():
    if (row['bet']):
        race.at[i,'f'] = row['p']-row['ip']*(1-p_total)/(1-ip_total)

# Total fraction bet is as per binary Kelly

total_f = p_total - (1-p_total)*ip_total/(1-ip_total)
print("Total Kelly fraction = ",total_f)

# Optimal expected log growth is Kullback-Leibler divergence

klg = 0.0
for i, row in race.iterrows():
    if (row['bet']):
        klg = klg + row['p']*np.log(row['p']/row['ip'])

klg = klg + (1-p_total)*np.log((1-p_total)/(1-ip_total))
print("K-L growth = ",klg)

print()
print(race)
print()

# Second example

race = pd.DataFrame()

D = 0.85

p = np.array([0.25,0.1,0.1,0.4,0.15])
b = np.array([0.17,0.05667,0.034,0.34,0.3993])

race['p'] = p
race['b'] = b

# Implied probabilities are most natural

ip = b/D

race['ip'] = ip
race['r'] = race['p']/race['ip']

race = race.sort_values(by=['r'], ascending=[False])

race['bet'] = False

p_total = 0.0
ip_total = 0.0
for i, row in race.iterrows():
    # Must be a positive hedge
    if (row['p'] > row['ip']*(1-p_total)/(1-ip_total)):
        race.at[i,'bet'] = True
        p_total = p_total + row['p']
        ip_total = ip_total + row['ip']
    else:
        break

# Fractions as per binary Kelly

race['f'] = 0.0
for i, row in race.iterrows():
    if (row['bet']):
        race.at[i,'f'] = row['p']-row['ip']*(1-p_total)/(1-ip_total)

# Total fraction bet is as per binary Kelly

total_f = p_total - (1-p_total)*ip_total/(1-ip_total)
print("Total Kelly fraction = ",total_f)

# Optimal expected log growth is Kullback-Leibler divergence

klg = 0.0
for i, row in race.iterrows():
    if (row['bet']):
        klg = klg + row['p']*np.log(row['p']/row['ip'])

klg = klg + (1-p_total)*np.log((1-p_total)/(1-ip_total))
print("K-L growth = ",klg)

print()
print(race)
print()
