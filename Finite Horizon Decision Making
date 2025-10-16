import numpy as np

C = 3
N = 4
cu = 3
cx = 1
cd = 5
cn = 4
W_k = [
    [0.2, 0.2, 0.2, 0.4],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.1, 0.4, 0.1, 0.4]
]

V = np.zeros((C + 1, N + 1)) 
policy = np.zeros((C + 1, N), dtype=int)
for x in range(C + 1):
    V[x][N] = cn * x

def g_k(x, u, w):
    return cu * u + cx * x + cd * max(0, w - x - u)

def f(x, u, w):
    return min(C, max(0, x + u - w))

def expected_cost(x, u, k):
    t = 0
    for w in range(len(W_k[k])):
        next_inventory = f(x, u, w)
        t += (g_k(x, u, w) + V[next_inventory][k + 1]) * W_k[k][w]
    return t

def solve(N):
    for k in range(N - 1, -1, -1):
        for x in range(C + 1):
            minCost = float("inf")
            best = 0
            for u in range(C + 1):
                total = expected_cost(x, u, k)
                if total < minCost:
                    minCost = total
                    best = u
            V[x][k] = minCost
            policy[x][k] = best
    return V, policy

V, policy = solve(N)
print(V[:, :N])  
print(policy)
