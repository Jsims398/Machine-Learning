import numpy as np

g = 9.8
l = 1 
m = 0.3
A = np.array([[0, 1], [g/l, 0]]).astype(np.float32)
B = np.array([[0], [1/(m*l**2)]]).astype(np.float32)
Q = np.array([[1, 0], [0, 0]]).astype(np.float32)
R = np.array([[3]]).astype(np.float32)
x0 = np.array([[1],[2]]).astype(np.float32)
N = 8

# A = np.array([[1,2],[-4,1]])
# B = np.array([[1,2],[3,2]])
# R = np.array([[1,0],[0,2]])
# Q = np.array([[1,0],[0,0]])
# x0 = np.array([[5],[10]])
# N = 9

# A = np.array([
#     [1, 2, 0],
#     [-4, 1, 0],
#     [0, -1, 3]
# ], dtype=np.float32)

# B = np.array([
#     [1, 2],
#     [0, -1],
#     [-1, 0]
# ], dtype=np.float32)

# Q = np.array([
#     [5, -4, 0],
#     [-4, 16, 0],
#     [0, 0, 0]
# ], dtype=np.float32)

# R = np.array([
#     [1, 0],
#     [0, 3]
# ], dtype=np.float32)

# x0 = np.array([[-13], [25], [20]], dtype=np.float32)
# N= 4

def f(x, u,A,B):
    return A @ x + B @ u

def j(Q, Xk, Uk):
    total = 0
    print("g")
    for x,u in zip(Xk, Uk):
        gvalue = g(Q, x, u)
        print(gvalue.round(2))
        total += gvalue
    return total

def g(Q, x, u):
    return x.T @ Q @ x + u.T @ R @ u

def V(Xk, Pk):
    return [x.T @ p @ x for p,x in zip(Pk, Xk)]

def lqr(A, B, x0, Q, R, N):
    Kk = [None] * N
    Pk = [None] * (N + 1)
    Pk[N] = Q
    
    for k in range(N - 1, -1, -1):
        Kk[k] = np.linalg.inv(R + B.T @ Pk[k + 1] @ B) @ (B.T @ Pk[k + 1] @ A)
        Pk[k] = Q + A.T @ Pk[k + 1] @ A - A.T @ Pk[k + 1] @ B @ Kk[k]
        print(Pk[k].round(2))

    Xk = [x0]
    Uk = []
    for k in range(N):
        u = -Kk[k] @ Xk[k]
        x_next = f(Xk[k], u, A, B)
        Uk.append(u)
        Xk.append(x_next)

    print(j(Q, Xk[:-1], Uk)) 
    return Pk, Xk

Pk, Xk = lqr(A, B, x0, Q, R, N)
print(V(Xk, Pk))
