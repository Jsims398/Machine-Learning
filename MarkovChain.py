A = np.array([
    [0,   0.9, 0.0, 0  ],
    [0.9, 0,   0.1, 0  ],
    [0.1, 0,   0.9,   0.9],
    [0,   0.1, 0.0, 0.1  ]
])

# A = np.array([
#     [.5, .3 , .2, .1],
#     [.2, .4 , .3, .2],
#     [.2, .2 , .4, .3],
#     [.1, .1 , .1, .4]
# ])
times = 10000

v0 = np.array([0.0, 1.0, 0.0, 0.0])
v = v0.copy()
for t in range(times):
    v = np.dot(A,v.transpose())   
print("\nMarkov Chain State after {} transitions:".format(times))
print(v)
