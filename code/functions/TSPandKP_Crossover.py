# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:44:31 2023

@author: dell
"""
import random
import copy
import numpy as np

def kp_single(bestpack1,bestpack2):
    a1 = list(bestpack1)
    a2 = list(bestpack2)
    packchild1 = copy.deepcopy(a1)
    packchild2 = copy.deepcopy(a2)
    # cross position
    y = random.randint(0,len(packchild1))
    # record cross entries
    packchild1[y:], packchild2[y:] = packchild2[y:], packchild1[y:]
    return packchild1,packchild2

def kp_two_point(parent1, parent2):
    length = len(parent1)
    # Randomly choose two crossover points
    cp1, cp2 = sorted(random.sample(range(1, length), 2))
    # Create offspring by combining the parts
    packchild1 = parent1[:cp1] + parent2[cp1:cp2] + parent1[cp2:]
    packchild2 = parent2[:cp1] + parent1[cp1:cp2] + parent2[cp2:]
    return packchild1, packchild2

def kp_uniform(parent1, parent2, crossover_rate=0.5):
    #param crossover_rate: Probability of each gene being selected from the first parent.
    packchild1, packchild2 = [], []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < crossover_rate:
            packchild1.append(gene1)
            packchild2.append(gene2)
        else:
            packchild1.append(gene2)
            packchild2.append(gene1)

    return packchild1, packchild2

# Performs partially mapped (PMX)
# Randomly copies a section of genes from parent 1 to the child
# the remaining positions are filled in with the genes from the second parent utilizing a mapping to avoid collisions
# Input: Parent 1 and Parent 2 as numpy arrays
# Output: Child 1 and child 2 as numpy arrays
def PMXCrossover(parent1, parent2):
    pos1 = pos2 = 0

    while pos1 >= pos2:
        pos1 = int(np.random.randint(0, len(parent1) + 1))
        pos2 = int(np.random.randint(0, len(parent1) + 1))

    child1 = np.zeros(len(parent1), dtype=int)
    child1[child1 == 0] = -1

    child1[pos1:pos2] = parent1[pos1:pos2]

    for i in np.concatenate([np.arange(0, pos1), np.arange(pos2, len(parent1))]):
        can = parent2[i]

        while can in child1[pos1:pos2]:
            can = parent2[np.where(parent1 == can)[0][0]]

        child1[i] = can

    child2 = np.zeros(len(parent2), dtype=int)
    child2[child2 == 0] = -1

    child2[pos1:pos2] = parent2[pos1:pos2]

    for i in np.concatenate([np.arange(0, pos1), np.arange(pos2, len(parent2))]):
        can = parent1[i]

        while can in child2[pos1:pos2]:
            can = parent1[np.where(parent2 == can)[0][0]]

        child2[i] = can

    return child1, child2


# ordered crossover
def ordered_crossover(perm1, perm2):
    k = random.randrange(len(perm1))  # determine cut point

    # splitting the permutations at the cut point
    original1 = np.array(perm1[:k])
    remaining1 = np.array(perm1[k:])
    remaining1_indidces = []
    original2 = np.array(perm2[k:])
    remaining2 = np.array(perm2[:k])
    remaining2_indidces = []

    # noting the indices of remaining1's items in perm2
    for x in remaining1:
        remaining1_indidces.append(perm1[np.where(perm2 == x)])


    # noting the indices of remaining2's items in perm1
    for x in remaining2:
        remaining2_indidces.append(perm2[np.where(perm1 == x)])

    # reorder remaining1 and remaining2 according to the previously noted indices
    remaining1_ordered = np.array([x for _, x in sorted(zip(remaining1_indidces, remaining1))])
    remaining2_ordered = np.array([x for _, x in sorted(zip(remaining2_indidces, remaining2))])

    # create the offspring by putting both parts of the permutation together again
    child1 = np.hstack((original1, remaining1_ordered))
    child2 = np.hstack((remaining2_ordered, original2))

    return child1, child2.astype(int)



def cyclic_crossover(perm1, perm2):
    perm1 = copy.deepcopy(perm1)
    perm2 = copy.deepcopy(perm2)
    cycle_found = False
    cycle = []
    i = 0
    cycle.append(perm1[i])  # initially start with the first element in perm1

    while not cycle_found: # unless we found a cycle, continue

        next = perm2[i]  # determine where to go next
        if not cycle.__contains__(next):  # if we havent visited this element already
            cycle.append(next)  # add to cycle
            i = np.where(perm1 == next)[0][0]  # find index of previously found element in perm1
        else:
            cycle_found = True  # if we have visited this element already, stop

    fixed_indices = []

    # find indices according to the elements we determined to be in a cycle
    for item in cycle:
        fixed_indices.append(np.where(perm1 == item)[0][0])

    # all the elements at indices determined previously stay fixed
    # for all other indices, swap elements from perm1 and perm2
    for i in range(len(perm1)):
        if i not in list(fixed_indices):
            perm1[i], perm2[i] = perm2[i], perm1[i]
    return perm1, perm2


def tsp_single(bestfather1,bestfather2):
    a1 = list(bestfather1)
    a2 = list(bestfather2)
    a1_1 = copy.deepcopy(a1)
    a2_1 = copy.deepcopy(a2)
    # cross position
    y = random.randint(0,len(a1_1))
    # record cross entries
    fragment1 = a1[y:]
    fragment2 = a2[y:]
    a1_1[y:], a2_1[y:] = a2_1[y:], a1_1[y:]
    a1_2 = []
    a2_2 = []
    #Repeat correction
    for i in a1_1[:y]:
        while i in fragment2:
            i = fragment1[fragment2.index(i)]
        a1_2.append(i)
    for i in a2_1[:y]:
        while i in fragment1:
            i = fragment2[fragment1.index(i)]
        a2_2.append(i)
    child1 = a1_2 + fragment2
    child2 = a2_2 + fragment1
    #print('--' * 25,'\nThe corrected descendants of the iteration are:\n{}\n{}'.format(child1, child2))
    return child1,child2


def tsp_two_point(bestfather1,bestfather2):
    #Partially Matched Crossover (PMX): Two-Point Crossover:
    length = len(bestfather1)
    crossover_point1 = random.randint(0, length - 3)
    crossover_point2 = random.randint(crossover_point1 + 1, length - 1)
    child1, child2 = list(bestfather1), list(bestfather2)
    # get crossover part
    middle1 = bestfather1[crossover_point1:crossover_point2]
    middle2 = bestfather2[crossover_point1:crossover_point2]
    # crossover
    child1[crossover_point1:crossover_point2] = middle2
    child2[crossover_point1:crossover_point2] = middle1
    # fix
    for i in range(length):
        if i < crossover_point1 or i >= crossover_point2:
            while child1[i] in middle2:
                child1[i] = bestfather1[bestfather2.index(child1[i])]
            while child2[i] in middle1:
                child2[i] = bestfather2[bestfather1.index(child2[i])]
    #print('--' * 25,'\nThe corrected descendants of the iteration are:\n{}\n{}'.format(child1, child2))
    return child1,child2



if __name__ == '__main__':
    p1 = np.array([0, 1, 2, 3, 4])
    p2 = np.array([4, 3, 2, 1, 0])
    p3 = np.array([1, 4, 2, 0, 3])
    p4 = np.array([3, 4, 0, 2, 1])

    print(tsp_two_point(list(p1), list(p3)))

        





