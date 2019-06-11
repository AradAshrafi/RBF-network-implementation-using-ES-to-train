import random
from RBF import RBF
import numpy as np

mutation_prob = 0.4
crossover_prob = 0.8


# array of RBFs contain our desired chromosomes, V and ‫γ
# actually MU is equal to length of RBF array
def ES(train_data_input, train_data_output, MU, LAMBDA):
    array_of_RBFS = []

    # INITIAL POPULATION ------------------>
    for i in range(MU):
        rbf = RBF(RBF_input=train_data_input, RBF_output=train_data_output)
        array_of_RBFS.append(rbf)

    for rbf in array_of_RBFS:
        # first we must do some math :D
        # to calculate proper inner amounts (like Loss) in RBF based on determined procedure
        __do_required_calculation_to_determine_loss(rbf=rbf)

    # MAIN LOOP OF EVOLUTIONARY STRATEGY ------------------------>
    for i in range(200):
        print("iteration" + str(i))
        # now it's time to generate next generations and perform Evolutionary Strategy
        new_generation = __generate_new_generation(array_of_RBFS=array_of_RBFS, MU=MU, LAMBDA=LAMBDA,
                                                   RBF_input=train_data_input, RBF_output=train_data_output)
        for new_child in new_generation:
            # to calculate proper inner amounts (like Loss) in RBF based on determined procedure
            __do_required_calculation_to_determine_loss(rbf=new_child)

        # new Chromosomes will take place of current ones :(
        # Dream Theater - This Is The Life
        array_of_RBFS = __evolution_selection(new_generation=new_generation, MU=MU)

        for remained_child in array_of_RBFS:
            print(remained_child.Loss)

    array_of_RBFS.sort(key=lambda x: x.Loss, reverse=False)
    return array_of_RBFS[0]


def __do_required_calculation_to_determine_loss(rbf):
    # first we calculate G matrix based on centers(v) and gama [formula is written in README ]
    rbf.calculate_G_matrix()
    # then we calculate weights based on G matrix [formula is written in README ]
    rbf.calculate_weights()
    # then we calculate output with current G and W (y = GW) [formula is written in README :D ]
    rbf.calculate_output()
    # in the end we calculate error based on loss function defined in README
    rbf.calculate_error()


# my way of creating new generation :
# first iterate through all RBFs and make 2 children from each one (other parent will be chosen randomly)
# so up to now we have 2*MU children
# then I make more lambda - 2*MU children with completely random (I always choose lambda > 2m)
def __generate_new_generation(array_of_RBFS, MU, LAMBDA, RBF_input, RBF_output):
    new_generation = []
    for i in range(MU):
        first_parent = array_of_RBFS[i]
        second_parent = array_of_RBFS[random.randint(0, MU - 1)]
        third_parent = array_of_RBFS[random.randint(0, MU - 1)]

        first_child = __new_child(first_parent=first_parent, second_parent=second_parent, RBF_input=RBF_input,
                                  RBF_output=RBF_output) if (random.random() < crossover_prob) else first_parent
        second_child = __new_child(first_parent=first_parent, second_parent=third_parent, RBF_input=RBF_input,
                                   RBF_output=RBF_output) if (random.random() < crossover_prob) else first_parent
        new_generation.append(first_child)
        new_generation.append(second_child)

    for i in range(LAMBDA - 2 * MU):
        first_parent = array_of_RBFS[random.randint(0, MU - 1)]
        second_parent = array_of_RBFS[random.randint(0, MU - 1)]
        new_generation.append(__new_child(first_parent=first_parent, second_parent=second_parent, RBF_input=RBF_input,
                                          RBF_output=RBF_output))
    return new_generation


# create new child from it's parents
def __new_child(first_parent, second_parent, RBF_input, RBF_output):
    new_child_chromosome = np.true_divide(np.add(first_parent.get_chromosome(), second_parent.get_chromosome()), 2)
    new_child_chromosome = __mutation(chromosome=new_child_chromosome)
    new_child = RBF(RBF_input=RBF_input, RBF_output=RBF_output, GAMA=new_child_chromosome[0],
                    V=new_child_chromosome[1])
    return new_child


# mutate one amount in gama and V with some probability
def __mutation(chromosome):
    if (random.random() < (mutation_prob)):
        for i in range(2):
            # change one amount in gama vector
            index = random.randint(0, len(chromosome[0]) - 1)
            chromosome[0][index] = np.multiply(chromosome[0][index], random.uniform(0.5, 1.5))

            # change one amount in V vector
            index = random.randint(0, len(chromosome[1]) - 1)
            chromosome[1][index] = np.multiply(chromosome[1][index], random.uniform(0.5, 1.5))
    return chromosome


# evolution only selects in new generation because ES(MU, LAMBDA) and current generation all will die
def __evolution_selection(new_generation, MU):
    return __q_tournament(population_before_selection=new_generation, Q=3, ns=MU)


def __q_tournament(population_before_selection, Q, ns):
    selected_population = []
    for i in range(ns):
        opponents = []
        for j in range(Q):
            opponents.append(population_before_selection[random.randint(0, len(population_before_selection) - 1)])
        opponents.sort(key=lambda x: x.Loss, reverse=False)
        selected_population.append(opponents[0])
    return selected_population
# def __SUS_with_ranking():
#     print()
