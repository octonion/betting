from math import log
from scipy.optimize import minimize

from sys import argv

win = 1/float(argv[1])
draw = 1/float(argv[2])
lose = 1/float(argv[3])

print("Odds:")
print()
print("  win: {}".format(argv[1]))
print("  draw: {}".format(argv[2]))
print("  lose: {}".format(argv[3]))
print()

print("Implied:")
print()
print("  Pr(win) = {}".format(win))
print("  Pr(draw) = {}".format(draw))
print("  Pr(lose) = {}".format(lose))
print()

s = win+draw+lose

m_win = win/s
m_draw = draw/s
m_lose = lose/s

def kl(p, q):
    kls = (p-q)*log((p/(1-p))/(q/(1-q)))
    return(kls)

f = lambda p: (kl(p[0], win) - kl(p[1], draw))**2 + (kl(p[0], win) - kl(1-p[0]-p[1], lose))**2

cons = ({'type': 'ineq', 'fun': lambda p: p[0]},
        {'type': 'ineq', 'fun': lambda p: 1-p[0]},
        {'type': 'ineq', 'fun': lambda p: p[1]},
        {'type': 'ineq', 'fun': lambda p: 1-p[1]},
        {'type': 'ineq', 'fun': lambda p: p[0]+p[1]},
        {'type': 'ineq', 'fun': lambda p: 1-p[0]-p[1]})

bnds = [(0.001,0.999),(0.001,0.999)]

p0 = (m_win,m_draw)

solution = minimize(f, p0, bounds=bnds, constraints=cons)
x = solution['x']

e_win = x[0]
e_draw = x[1]
e_lose = 1-x[0]-x[1]
#print(solution)

print("Estimated:")
print()
print("  Pr(win) = {}".format(e_win))
print("  Pr(draw) = {}".format(e_draw))
print("  Pr(lose) = {}".format(e_lose))
print()
