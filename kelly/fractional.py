#!/usr/bin/env python3

import csv
import sys

import numpy as np
import pandas as pd

csv_name = sys.argv[1]
fraction = float(sys.argv[2])

with open(sys.argv[1], 'r') as f:
    csv = csv.reader(f)
    for p in csv:
        ip = next(csv)
        m = int(p[0])
        
        # Our probabilities
        
        p = np.array([float(i) for i in p[1:]])
        n = int(ip[0])

        # Implied probabilities
        
        ip = np.array([float(i) for i in ip[1:]])
        print("Race:",n)
        print()

        race = pd.DataFrame()

        race['p'] = p
        race['p*'] = ip
        race['r'] = race['p']/race['p*']
        race = race.sort_values(by=['r'], ascending=[False])
        race['bet'] = False

        p_total = 0.0
        ip_total = 0.0
        
        for i, row in race.iterrows():
            # Must be a positive hedge
            if (row['p'] > row['p*']*(1-p_total)/(1-ip_total)):
                race.at[i,'bet'] = True
                p_total = p_total + row['p']
                ip_total = ip_total + row['p*']
            else:
                break

        # Fractions as per binary Kelly

        race['f'] = 0.0
        for i, row in race.iterrows():
            if (row['bet']):
                race.at[i,'f'] = row['p']-row['p*']*(1-p_total)/(1-ip_total)

        f = p_total-(1-p_total)*ip_total/(1-ip_total)
        growth = 0.0
        for i, row in race.iterrows():
            if (row['bet']):
                growth += row['p']*np.log(1-f+row['f']/row['p*'])
                
        growth += (1-p_total)*np.log(1-f)
        #print("Expected log-growth =",growth)

        # Optimal bet fraction is as per binary Kelly

        optimal_f = p_total - (1-p_total)*ip_total/(1-ip_total)
        print("Optimal Kelly fraction =",optimal_f)

        # Optimal expected log growth is Kullback-Leibler divergence

        klg = 0.0
        for i, row in race.iterrows():
            if (row['bet']):
                klg = klg + row['p']*np.log(row['p']/row['p*'])

        klg = klg + (1-p_total)*np.log((1-p_total)/(1-ip_total))
        print("Kullback-Leibler growth =",klg)

        print()
        print("Full Kelly optimal bets:")
        print()
        print(race)
        print()

        # Fractional Kelly

        print("Fraction of optimal Kelly =",fraction)
        f = fraction*optimal_f
        print("Fraction of bankroll =",f)
        print()
        
        p_growth = 0.0
        for i, row in race.iterrows():
            if (row['bet']):
                p_growth += row['p']*np.log(1-f+fraction*row['f']/row['p*'])

        p_growth += (1-p_total)*np.log(1-f)

        print("Proportional fractional expected log-growth =",p_growth)

        for i in reversed(race.index):
            if (race.at[i,'bet']) and (f*(race.at[i,'p']*(1-ip_total)/p_total+race.at[i,'p*'])+(race.at[i,'p']*ip_total/p_total-race.at[i,'p*']) < 0):
                race.at[i,'bet'] = False
                p_total = p_total-race.at[i,'p']
                ip_total = ip_total-race.at[i,'p*']

        growth = 0.0
        race['f'] = 0.0
        for i, row in race.iterrows():
            if (row['bet']):
                race.at[i,'f'] = f*(row['p']*(1-ip_total)/p_total+row['p*'])+(row['p']*ip_total/p_total-row['p*'])

        for i, row in race.iterrows():
            if (row['bet']):
                growth += row['p']*np.log(1-f+row['f']/row['p*'])

        growth += (1-p_total)*np.log(1-f)
        print("Optimal fractional expected log-growth =",growth)

        print()
        print("Fractional Kelly optimal bets:")
        print()
        print(race)
        print()
