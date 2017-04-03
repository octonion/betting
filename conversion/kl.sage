#!/usr/bin/sage -python

# Spurs

l1 = 5.120

# Jazz

l2 = 1.143

# Implied probabilities

i1 = 1/l1
i2 = 1/l2

# Equal-information gain odds ratio estimate for Spurs

o = (i1^i1*(1-i1)^(1-i1)/(i2^i2*(1-i2)^(1-i2)))^(1/(i1+i2-1))

# Spurs KL estimate

p1 = 1/(o+1)

# Jazz KL estimate

p2 = o/(o+1)

print(p1)
print(p2)


