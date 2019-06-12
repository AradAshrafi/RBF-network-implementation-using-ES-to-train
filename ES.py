import random
from RBF import RBF
import numpy as np

mutation_prob = 0.45
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
        __do_required_calculation_to_determine_loss(rbf=rbf, RBF_input=train_data_input, RBF_output=train_data_output)

    # MAIN LOOP OF EVOLUTIONARY STRATEGY ------------------------>
    for i in range(150):
        print("iteration" + str(i))
        # now it's time to generate next generations and perform Evolutionary Strategy
        # parents will be chosen randomly
        new_generation = __generate_new_generation(array_of_RBFS=array_of_RBFS, MU=MU, LAMBDA=LAMBDA)

        # new Chromosomes will take place of current ones :(
        # Dream Theater - This Is The Life
        array_of_RBFS = __evolution_selection_q_tournament(population_before_selection=new_generation,
                                                           RBF_input=train_data_input, RBF_ouptput=train_data_output,
                                                           ns=MU, Q=3)

        for remained_child in array_of_RBFS:
            print(remained_child.Loss)
        print()

    array_of_RBFS.sort(key=lambda x: x.Loss, reverse=False)
    return array_of_RBFS[0]


def __do_required_calculation_to_determine_loss(rbf, RBF_input=None, RBF_output=None):
    # this checking statement is useful in picking new generation
    # if rbf.Loss isn't 0 it means that we have calculated this chromosome loss before
    if rbf.Loss != 0:
        return
    rbf.set_input(RBF_input=RBF_input)
    rbf.set_output(RBF_output=RBF_output)
    # first we calculate G matrix based on centers(v) and gama [formula is written in README ]
    rbf.calculate_G_matrix()
    # then we calculate weights based on G matrix [formula is written in README ]
    rbf.calculate_weights()
    # then we calculate output with current G and W (y_prime = GW) [formula is written in README :D ]
    rbf.calculate_output()
    # in the end we calculate error based on loss function defined in README
    rbf.calculate_error()


# my way of creating new generation :
# first iterate through all RBFs and make 2 children from each one (other parent will be chosen randomly)
# so up to now we have 2*MU children
# then I make more lambda - 2*MU children with completely random (I always choose lambda > 2m)
def __generate_new_generation(array_of_RBFS, MU, LAMBDA):
    new_generation = []
    for i in range(MU):
        first_parent = array_of_RBFS[i]
        second_parent = array_of_RBFS[random.randint(0, MU - 1)]
        third_parent = array_of_RBFS[random.randint(0, MU - 1)]

        first_child = __new_child(first_parent=first_parent, second_parent=second_parent) if (
                random.random() < crossover_prob) else first_parent
        second_child = __new_child(first_parent=first_parent, second_parent=third_parent) if (
                random.random() < crossover_prob) else first_parent
        new_generation.append(first_child)
        new_generation.append(second_child)

    for i in range(LAMBDA - 2 * MU):
        first_parent = array_of_RBFS[random.randint(0, MU - 1)]
        second_parent = array_of_RBFS[random.randint(0, MU - 1)]
        new_generation.append(__new_child(first_parent=first_parent, second_parent=second_parent))
    return new_generation


# create new child from it's parents
def __new_child(first_parent, second_parent):
    new_child_chromosome = np.true_divide(np.add(first_parent.get_chromosome(), second_parent.get_chromosome()), 2)
    new_child_chromosome = __mutation(chromosome=new_child_chromosome)
    new_child = RBF(GAMA=new_child_chromosome[0], V=new_child_chromosome[1])
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
# perform q tournament
def __evolution_selection_q_tournament(population_before_selection, RBF_input, RBF_ouptput, ns, Q=2):
    selected_population = []
    for i in range(ns):
        opponents = []
        for j in range(Q):
            opponent = population_before_selection[random.randint(0, len(population_before_selection) - 1)]
            # calculate loss of selected child
            __do_required_calculation_to_determine_loss(opponent, RBF_input=RBF_input, RBF_output=RBF_ouptput)
            opponents.append(opponent)
        # best competitor will be chosen to stay alive
        opponents.sort(key=lambda x: x.Loss, reverse=False)
        selected_population.append(opponents[0])
    return selected_population

# def __SUS_with_ranking():
#     print()
