import cvxpy as cp
import numpy as np

x = cp.Variable((12), nonneg=True)
s = cp.Variable((12), nonneg=True)
y = cp.Variable((12), nonneg=True)
z = cp.Variable((12), nonneg=True)

d = np.array([350, 325, 450, 640, 640, 550, 700, 670, 350, 425, 400, 650])

cost1 = np.array([50,50,50,50,50,50,50,50,50,50,50,50])
cost2 = np.array([20,20,20,20,20,20,20,20,20,20,20,20])

objective = cp.Minimize((cost1.T @ y)+(cost1.T @ z)+(cost2.T @ s))

A = np.array([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]])

constraints = [A @ s == d.T - x,
               A @ x == y - z]

problem = cp.Problem(objective, constraints)


print(problem.solve())
print(x.value)