import random
from RBF import RBF
import numpy as np

mutation_prob = 0.4
crossover_prob = 0.7


# array of RBFs contain our desired chromosomes, V and ‫γ
# actually MU is equal to length of RBF array
def ES(train_data_input, train_data_output, MU, LAMBDA, number_of_centers, total_iterations, mode):
    array_of_RBFs = []

    # INITIAL POPULATION ------------------>
    for i in range(MU):
        rbf = RBF(RBF_input=train_data_input, RBF_output=train_data_output, centers_number=number_of_centers)
        array_of_RBFs.append(rbf)

    for rbf in array_of_RBFs:
        # first we must do some math :D
        # to calculate proper inner amounts (like Loss) in RBF based on determined procedure
        __do_required_calculation_to_determine_loss(rbf=rbf, RBF_input=train_data_input, RBF_output=train_data_output,
                                                    mode=mode)

    # MAIN LOOP OF EVOLUTIONARY STRATEGY ------------------------>
    for i in range(total_iterations):
        print("iteration" + str(i))
        # now it's time to generate next generations and perform Evolutionary Strategy
        # parents will be chosen randomly
        new_generation = __generate_new_generation(array_of_RBFS=array_of_RBFs, MU=MU, LAMBDA=LAMBDA)
        #---------------------------- Important------------------------------->
        # for ES (MU+ LAMBDA) uncomment next line ---------------->>>>>>>>>>>>>
        # new_generation.extend(array_of_RBFs)

        # new Chromosomes will take place of current ones :(
        # Dream Theater - This Is The Life
        array_of_RBFs = __evolution_selection_q_tournament(population_before_selection=new_generation,
                                                           RBF_input=train_data_input, RBF_ouptput=train_data_output,
                                                           ns=MU, Q=5, mode=mode)

        for remained_child in array_of_RBFs:
            print(remained_child.Loss)

    array_of_RBFs.sort(key=lambda x: x.Loss, reverse=False)
    return array_of_RBFs[0]


def __do_required_calculation_to_determine_loss(rbf, RBF_input=None, RBF_output=None, mode="regression"):
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
    rbf.calculate_error(mode)

    # make them release MEMORY :)) ---------------------------------------------
    rbf.set_input(None)
    rbf.set_output(None)


# my way of creating new generation :
# first iterate through all RBFs and make 2 children from each one (other parent will be chosen randomly)
# so up to now we have 2*MU children
# then I make more lambda - 2*MU children with completely random (I always choose lambda > 2m)
# ----------------------- important ------------------------------------->
# in ES we first do Mutation then we cross them over each other
# so after that I chose parents randomly, I Mutate them, Then I do cross over
def __generate_new_generation(array_of_RBFS, MU, LAMBDA):
    new_generation = []
    for j in range(2):
        for i in range(MU):
            parent = array_of_RBFS[i]
            # mutate parent chromosome
            parent_mutated_chromosome = __mutation(chromosome=parent.get_chromosome())
            parent.set_chromosome(parent_mutated_chromosome[0], parent_mutated_chromosome[1])
    for i in range(LAMBDA):
        first_parent = array_of_RBFS[random.randint(0, MU - 1)]
        second_parent = array_of_RBFS[random.randint(0, MU - 1)]
        new_generation.append(__new_child(first_parent=first_parent, second_parent=second_parent))
    return new_generation


# create new child from it's parents
# method we used in here was average crossing
def __new_child(first_parent, second_parent):
    new_child_chromosome = np.true_divide(np.add(first_parent.get_chromosome(), second_parent.get_chromosome()), 2)
    new_child = RBF(GAMA=new_child_chromosome[0], V=new_child_chromosome[1])
    return new_child


# mutate one amount in gama and V with some probability
def __mutation(chromosome):
    if (random.random() < (mutation_prob)):
        for i in range(2):
            # change one amount in gama vector
            index = random.randint(0, len(chromosome[0]) - 1)
            chromosome[0][index] = np.multiply(chromosome[0][index],
                                               random.uniform(random.uniform(0.45, 0.9), random.uniform(1.1, 1.55)))

            # change one amount in V vector
            index = random.randint(0, len(chromosome[1]) - 1)
            chromosome[1][index] = np.multiply(chromosome[1][index],
                                               random.uniform(random.uniform(0.45, 0.9), random.uniform(1.1, 1.55)))

    return chromosome


# evolution only selects in new generation because ES(MU, LAMBDA) and current generation all will die
# perform q tournament
def __evolution_selection_q_tournament(population_before_selection, RBF_input, RBF_ouptput, ns, Q=2, mode="regression"):
    selected_population = []
    for i in range(ns):
        opponents = []
        for j in range(Q):
            opponent = population_before_selection[random.randint(0, len(population_before_selection) - 1)]
            # calculate loss of selected child
            __do_required_calculation_to_determine_loss(opponent, RBF_input=RBF_input, RBF_output=RBF_ouptput,
                                                        mode=mode)
            opponents.append(opponent)
        # best competitor will be chosen to stay alive
        opponents.sort(key=lambda x: x.Loss, reverse=False)
        selected_population.append(opponents[0])
    return selected_population

# maybe for future I add SUS with rankings:D
# def __SUS_with_ranking():
