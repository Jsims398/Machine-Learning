import numpy as np
import math

rightProb = 0.5
upProb = 0.25
leftProb = 0.0
downProb = 0.25 

grid = [
    [0, 0, 0, 0, -5],
    [25, np.inf, np.inf, np.inf, 100],
    [0, 50, 0, np.inf, 0],
    [0, -1, 0, 0, 0]
]

def value_iteration(gamma=0.1, epsilon=1e-6):
    rows, cols = 4, 5

    V = np.array([[grid[i][j] if grid[i][j] != math.inf else 0
                   for j in range(cols)] for i in range(rows)], dtype=float)

    policy = np.full((rows, cols), ' ', dtype=str)

    actions = ['U', 'D', 'L', 'R']
    directions = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }

    def is_valid(i, j):
        return 0 <= i < rows and 0 <= j < cols and grid[i][j] != math.inf

    def next_state(i, j, option):
        di, dj = directions[option]
        ni, nj = i + di, j + dj
        if is_valid(ni, nj):
            return ni, nj
        else:
            return i, j

    while True:
        change = 0
        V_new = V.copy()

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == math.inf:
                    continue
                
                values = []
                for a in actions:
                    main_i, main_j = next_state(i, j, a)
                    if a == 'U':
                        option1, option2, option3 = 'L', 'R', 'D'
                    elif a == 'D':
                        option1, option2, option3 = 'R', 'L', 'U'
                    elif a == 'L':
                        option1, option2, option3 = 'D', 'U', 'R'
                    else: 
                        option1, option2, option3 = 'U', 'D', 'L'

                    option1_i, option1_j = next_state(i, j, option1)
                    option2_i, option2_j = next_state(i, j, option2)
                    option3_i, option3_j = next_state(i, j, option3)

                    expected_value = (
                        upProb * V[main_i, main_j] +
                        rightProb * V[option1_i, option1_j] +
                        leftProb * V[option2_i, option2_j] +
                        downProb * V[option3_i, option3_j]
                    )

                    values.append(expected_value)

                V_new[i, j] = grid[i][j] + gamma * min(values)
                policy[i, j] = actions[np.argmin(values)]
                change = max(change, abs(V_new[i, j] - V[i, j]))

        V = V_new
        if change < epsilon:
            break

    return V, policy

V, policy = value_iteration()
print("Value Function:")
print(V)
print("Policy:")
print(policy)
