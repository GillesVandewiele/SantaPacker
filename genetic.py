import numpy as np
import pandas as pd
import random
from copy import deepcopy
import time


def ff_weight(gift, n=50, p=75):
    return np.percentile([sample_weight(gift) for i in range(n)], p)


def sample_weight(gift):
    _type = gift.split('_')[0]

    if _type == "horse":
        return max(0, np.random.normal(5, 2, 1)[0])
    if _type == "ball":
        return max(0, 1 + np.random.normal(1, 0.3, 1)[0])
    if _type == "bike":
        return max(0, np.random.normal(20, 10, 1)[0])
    if _type == "train":
        return max(0, np.random.normal(10, 5, 1)[0])
    if _type == "coal":
        return 47 * np.random.beta(0.5, 0.5, 1)[0]
    if _type == "book":
        return np.random.chisquare(2, 1)[0]
    if _type == "doll":
        return np.random.gamma(5, 1, 1)[0]
    if _type == "blocks":
        return np.random.triangular(5, 10, 20, 1)[0]
    if _type == "gloves":
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]

def get_bag_weight(bag):
    return sum([sample_weight(gift) for gift in bag])


def check_submission(submitted_bags, verbose=0):
    for bag in submitted_bags:
        if 0 < len(bag) < 3:
            if verbose: print('A non-empty bag contains less than 3 items')
            return False

    # No used more than once
    fl = [item for sublist in submitted_bags for item in sublist]
    if len(fl) != len(set(fl)) and verbose: print('A bag contains duplicate items')
    return len(fl) == len(set(fl))


def fitness(vector, gifts):
    bags = [[] for i in range(BAGS)]
    for i in range(len(gifts)):
        if vector[i] > 0:
            bags[int(vector[i])-1].append(gifts[i])
    return score_submission(bags) * check_submission(bags)


def tournament_selection(population, tournament_size=5, p=0.75):
    tournament_idx = list(np.array(population)[np.random.choice(len(population), size=tournament_size*2, replace=False)])
    tournament_1 = tournament_idx[:tournament_size]
    tournament_2 = tournament_idx[tournament_size:]

    # Our first tournament
    tournament_1 = sorted(tournament_1, key=lambda x: x[1], reverse=True)
    bags1 = None
    for i in range(tournament_size):
        if random.random() < (p * (1-p)**i) or i == tournament_size - 1:
            bags1 = tournament_1[i][0]
            break

    # Our second tournament
    tournament_2 = sorted(tournament_2, key=lambda x: x[1], reverse=True)
    bags2 = None
    for i in range(tournament_size):
        if random.random() < (p * (1-p)**i) or i == tournament_size - 1:
            bags2 = tournament_2[i][0]
            break

    return bags1, bags2


def best_fit(gifts, capacity=50, nr_bags=1000):
    _gifts = deepcopy(gifts)
    random.shuffle(_gifts)
    bags = [([], 0) for i in range(nr_bags)]
    for gift in _gifts:
        gift_weight = sample_weight(gift)
        differences = [capacity - (bag[1] + gift_weight) if capacity - (bag[1] + gift_weight) > 0 else float('inf') for bag in bags]
        i = np.argmin(differences)
        if differences[i] != float('inf'):
            bags[i] = (bags[i][0] + [gift], bags[i][1] + gift_weight)
            _gifts.remove(gift)
    return [bag[0] for bag in bags], _gifts


def first_fit(gifts, capacity=45, nr_bags=1000):
    _gifts = deepcopy(gifts)
    random.shuffle(_gifts)

    viable_bags = []
    for k in range(len(_gifts) - 3):
        gifts_triplet = _gifts[k:k+3]
        gifts_weight = sum([ff_weight(gift) for gift in gifts_triplet])
        if gifts_weight <= capacity:
            viable_bags.append([gifts_triplet, gifts_weight])
            for gift in gifts_triplet: _gifts.remove(gift)
        if len(viable_bags) == nr_bags:
            break

    bags = viable_bags
    for gift in _gifts:
        gift_weight = ff_weight(gift)
        for i in range(len(bags)):
            if (bags[i][1] + gift_weight) <= capacity:
                bags[i] = (bags[i][0] + [gift], bags[i][1] + gift_weight)
                _gifts.remove(gift)
                break
    return [bag[0] for bag in bags] + []*(1000-len(bags)), _gifts


def load_data():
    df = pd.read_csv('gifts.csv')
    return list(df['GiftId'].values)


def to_numeric_vector(bags, gifts):
    vector = np.zeros(len(gifts))

    items_in_bags = {}
    for i, bag in enumerate(bags):
        for gift in bag: items_in_bags[gift] = i + 1

    for i in range(len(gifts)):
        if gifts[i] in items_in_bags.keys():
            vector[i] = int(items_in_bags[gifts[i]])

    return list(vector)


def mutate1(individual):
    _individual = deepcopy(individual)
    _individual[np.random.randint(len(individual))] = np.random.randint(1,1000)
    return _individual


def mutate2(individual):
    _individual = deepcopy(individual)
    idx1, idx2 = np.random.choice(len(_individual), size=2)
    temp = _individual[idx1]
    _individual[idx1] = _individual[idx2]
    _individual[idx2] = temp
    return _individual


def crossover1(individual1, individual2):
    k1 = np.random.randint(1, len(individual1)-1)
    individual1_sub1, individual1_sub2 = individual1[:k1], individual1[k1:]
    individual2_sub1, individual2_sub2 = individual2[:k1], individual2[k1:]
    new_individual1 = list(individual1_sub1) + list(individual2_sub2)
    new_individual2 = list(individual2_sub1) + list(individual1_sub2)
    return new_individual1, new_individual2

POPULATION_SIZE = 1000
CAPACITY = 50
MUTATION_PROBABILITY = 0.05
TOURNAMENT_SIZE = 25
TOURNAMENTS = 10
SELECTION_PROB = 0.75
BAGS = 1000

submission_files = []
for i in range(10):
    submission_files.append('submission'+str(i)+'.csv')

tshirt_files = []
for i in range(10):
    tshirt_files.append('tshirt'+str(i)+'.csv')

greedy_files = []
for i in range(10):
    tshirt_files.append('greedy'+str(i)+'.csv')

files = ['submission0.csv', 'Santa_03.csv', 'Santa_BTB.csv', 'submission_gen7500.csv', 'Santa_combo_search.csv',
         'submission_naive2.csv', 'submission_gen100.csv', 'greedy0.csv', 'tshirt0.csv',
         'submission_gen200.csv', 'submission_gen700.csv']

def genetic(verbose=0):
    gifts = load_data()
    population = []

    start = time.time()
    # Initialization
    for i in range(int(POPULATION_SIZE/50)):
        ff_bags, ff_gifts = first_fit(gifts, nr_bags=BAGS)
        ff_individual = to_numeric_vector(ff_bags, gifts)
        print('FF:', fitness(ff_individual, gifts))
        population.append((ff_individual, fitness(ff_individual, gifts)))

    for file in files:
        file_bags = read_submission(file)
        file_individual = to_numeric_vector(file_bags, gifts)
        print('FILE:', file, fitness(file_individual, gifts))
        population.append((file_individual, fitness(file_individual, gifts)))
    if verbose >= 2: print('[TIME] Initialization phase took', str((time.time()-start)))

    generation = 0
    while 1:
        # Replacement
        start = time.time()
        population = sorted(population, key=lambda x: x[1], reverse=True)[:POPULATION_SIZE]
        if verbose >= 2 and not generation: print('[TIME] Replacement phase took', str((time.time()-start)))

        # Tournament selection and cross-over
        tournament_times = []
        crossover_times = []
        tournament_phase_start = time.time()
        for tournament in range(TOURNAMENTS):

            start = time.time()
            winner1, winner2 = tournament_selection(population)
            tournament_times.append(time.time() - start)

            start = time.time()
            new_individual1, new_individual2 = crossover1(winner1, winner2)
            _fitness1 = fitness(new_individual1, gifts)
            _fitness2 = fitness(new_individual2, gifts)
            if verbose >= 3 and not generation % 25: print('Crossover1:', _fitness1, _fitness2)
            if _fitness1 > 0: population.append((new_individual1, _fitness1))
            if _fitness2 > 0: population.append((new_individual2, _fitness2))

            ff_bags, ff_gifts = first_fit(gifts, nr_bags=BAGS)
            ff_individual = to_numeric_vector(ff_bags, gifts)

            new_individual1, new_individual2 = crossover1(winner1, ff_individual)
            _fitness1 = fitness(new_individual1, gifts)
            _fitness2 = fitness(new_individual2, gifts)
            if verbose >= 3 and not generation % 25: print('CrossoverFF:', _fitness1, _fitness2)
            if _fitness1 > 0: population.append((new_individual1, _fitness1))
            if _fitness2 > 0: population.append((new_individual2, _fitness2))

            new_individual1, new_individual2 = crossover1(winner2, ff_individual)
            _fitness1 = fitness(new_individual1, gifts)
            _fitness2 = fitness(new_individual2, gifts)
            if verbose >= 3 and not generation % 25: print('CrossoverFF:', _fitness1, _fitness2)
            if _fitness1 > 0: population.append((new_individual1, _fitness1))
            if _fitness2 > 0: population.append((new_individual2, _fitness2))

            crossover_times.append(time.time() - start)
        if verbose >= 2 and not generation:
            print('[TIME] Tournament phase took', str((time.time() - tournament_phase_start)))
            print('[TIME] Tournament selection:', str(np.mean(tournament_times)))
            print('[TIME] Crossover:', str(np.mean(crossover_times)))

        # Mutation
        start = time.time()
        for individual in population:
           if random.random() < MUTATION_PROBABILITY:
                mutation = mutate1(individual[0])
                _fitness = fitness(mutation, gifts)
                if verbose >= 3 and not generation % 25: print('New mutation:', _fitness)
                if _fitness > 0: population.append((mutation, _fitness))
           if random.random() < MUTATION_PROBABILITY:
                mutation = mutate2(individual[0])
                _fitness = fitness(mutation, gifts)
                if verbose >= 3 and not generation % 25: print('New mutation:', _fitness)
                if _fitness > 0: population.append((mutation, _fitness))
        if verbose >= 2 and not generation: print('[TIME] Mutation phase took', str((time.time() - start)))

        # Verbose and write submission file
        if verbose >= 1 and not generation % 5: print('Generation:', generation, '---', population[0][1], '(Population size:', str(len(population))+')')
        if not generation % 100: write_submission_file(population[0][0], gifts, 'submission_gen'+str(generation)+'.csv')

        generation += 1


def write_submission_file(vector, gifts, path):
    bags = [[] for i in range(BAGS)]
    for i in range(len(gifts)):
        if vector[i] > 0:
            bags[int(vector[i])-1].append(gifts[i])

    score_submission(bags, n=1000, verbose=1)
    with open(path, 'w') as f:
        f.write('Gifts\n')
        for bag in bags:
            f.write(' '.join(bag)+'\n')
    f.close()


def read_submission(path):
    bags = []
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            bags.append(line.split())
    return bags


def score_submission(submitted_bags, n=10, verbose=0):
    arr = np.zeros((n))
    rej = np.zeros((n))
    for i in range(n):
        tot = 0.0
        rejected = 0
        for bag in submitted_bags:
            w = sum([sample_weight(item) for item in bag])
            if w <= 50.0:
                tot += w
            else:
                rejected += 1
        arr[i] = tot
        rej[i] = rejected / len(submitted_bags)
    if verbose:
        print("mean of {} simulations: {}  std: {} ".format(n, arr.mean(), np.std(arr)))
        print("Reject rate: {}".format(rej.mean()))
    return arr.mean()

genetic(verbose=3)