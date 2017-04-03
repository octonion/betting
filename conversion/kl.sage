#!/usr/bin/sage -python

# Spurs

l1 = 1.143

# Jazz

l2 = 5.120

# Implied probabilities

i1 = 1/l1
i2 = 1/l2

# Equal-information gain odds ratio estimate for Spurs

o = (i2^i2*(1-i2)^(1-i2)/(i1^i1*(1-i1)^(1-i1)))^(1/(i1+i2-1))

# Spurs KL estimate

p1 = 1/(o+1)

# Jazz KL estimate

p2 = o/(o+1)

print("Spurs implied Pr=%s" % i1)
print("Spurs estimated Pr=%s" % p1)

print("Jazz implied Pr=%s" % i2)
print("Jazz estimated Pr=%s" % p2)
