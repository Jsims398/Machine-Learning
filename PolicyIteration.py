import numpy as np
import math

gamma = 0.1
epsilon = 1e-6

grid = [
    [0, 0, 0, 0, -5],
    [25, np.inf, np.inf, np.inf, 100],
    [0, 50, 0, np.inf, 0],
    [0, -1, 0, 0, 0]
]


# grid = [[0,0,0,-1],
#         [0,np.inf,0,100],
#         [0,0,0,0]
#         ]

direction_probs = {
    'U': 0.5,
    'R': 0.25,
    'D': 0.0,
    'L': 0.25
}

actions = ['U', 'R', 'D', 'L']
action_deltas = {
    'U': (-1, 0),
    'R': (0, 1),
    'D': (1, 0),
    'L': (0, -1)
}

clockwise = {actions[i]: actions[(i + 1) % len(actions)] for i in range(len(actions))}
counterclockwise = {actions[i]: actions[(i - 1) % len(actions)] for i in range(len(actions))}

rows = len(grid)
cols = len(grid[0])

def is_valid(i, j):
    return 0 <= i < rows and 0 <= j < cols and not math.isinf(grid[i][j])

V_init = np.array([[grid[i][j] if not math.isinf(grid[i][j]) else 0
                    for j in range(cols)] for i in range(rows)], dtype=float)

policy = [[None if math.isinf(grid[i][j]) else 'R' for j in range(cols)] for i in range(rows)]

def resolve_relative_action(intended, relative):
    if relative == 'U':
        return intended
    elif relative == 'R':
        return clockwise[intended]
    elif relative == 'L':
        return counterclockwise[intended]
    elif relative == 'D':
        return clockwise[clockwise[intended]]

def getNext(i, j, intended_action):
    transitions = []
    for rel_action, prob in direction_probs.items():
        actual = resolve_relative_action(intended_action, rel_action)
        di, dj = action_deltas[actual]
        ni, nj = i + di, j + dj

        if is_valid(ni, nj):
            transitions.append((ni, nj, prob))
        else:
            transitions.append((i, j, prob))
    return transitions

def policy_evaluation(V, policy):
    iteration = 0
    while True:
        delta = 0.0
        new_V = np.copy(V)
        for i in range(rows):
            for j in range(cols):
                if not is_valid(i, j):
                    continue
                action = policy[i][j]
                immediate = grid[i][j]
                expected_future = 0.0
                next_steps = getNext(i, j, action)
                for ni, nj, prob in next_steps:
                    expected_future += prob * V[ni][nj]

                new_val = immediate + gamma * expected_future
                new_V[i][j] = new_val
                delta = max(delta, abs(new_val - V[i][j]))

        V = new_V
        iteration += 1
        if delta < epsilon:
            break

    return V

def policy_improvement(V):
    new_policy = [[None for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if not is_valid(i, j):
                continue
            best_action = None
            best_value = float('inf')
            for action in actions:
                value = 0
                for ni, nj, prob in getNext(i, j, action):
                    cost = grid[ni][nj]
                    value += prob * (cost + gamma * V[ni][nj])
                if value < best_value:
                    best_value = value
                    best_action = action
            new_policy[i][j] = best_action
    return new_policy

def policy_iteration():
    V = np.copy(V_init)
    stable = False
    iteration = 0
    while not stable:
        iteration += 1
        print(f"\nPolicy Iteration Round {iteration}")
        V = policy_evaluation(V, policy)
        new_policy = policy_improvement(V)
        stable = (new_policy == policy)
        policy[:] = new_policy
        print("Updated Policy:")
        for row in policy:
            print(row)
        print("Value Function:")
        for row in V:
            print(["{:.2f}".format(v) for v in row])
    return V, policy

V_final, optimal_policy = policy_iteration()
