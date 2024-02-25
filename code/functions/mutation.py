import numpy as np

def reverse_sequence_mutation(route):
    # obtain indicies, where i < j
    i = -1
    j = -1
    while i >= j:
        i = np.random.randint(0, route.size)
        j = np.random.randint(0, route.size)

    # flips the sequence
    route[i:j] = np.flip(route[i:j])

    return route

def bitflip(packing):
    # flip a random amount of times
    pos=np.random.choice(range(len(packing)), np.random.randint(1, 11), replace=False)
    for i in pos:
        if packing[i]==0:
            packing[i]=1
        else:
            packing[i]=0
    return packing

