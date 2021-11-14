import time
import numpy as np
# Greedy algorithm with value calculated by Pi = 1 - (1-A)(1-A^2)...(1-A^10), independent events probability
import statistics as s
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx

def greedy_improved(g, cost, population, infected, infected_no, mc, config):

    start = time.time()

    I = np.ones((5000, 1))

    F = np.ones((5000, 5000))
    N = np.ones((5000, 5000))

    A = nx.to_numpy_matrix(g, nodelist=list(range(5000)))

    for i in range(0, 10):
        B = np.power(A, i + 1)
        C = F - B
        N = np.multiply(N, C)

    P = F - N

    pi = np.matmul(P, I)

    value = {}

    for i in range(5000):
        value[i] = pi[i, 0]

    # v/w in knapsack

    unit_v = {}

    for i in range(5000):
        unit_v[i] = value[i] / cost[i]

    sorted_4_greedy = []

    for node in sorted(unit_v, key=unit_v.get, reverse=True):
        sorted_4_greedy.append(node)

    time1 = time.time() - start

    for percent in range(5, 95, 5):

        k = int(population * percent / 100)

        start = time.time()

        current_greedy = 0
        greedy = []

        for node in sorted_4_greedy:
            C = cost[node]
            if (node not in infected) and (current_greedy + C <= k):
                greedy.append(node)
                current_greedy += C
            else:
                continue

        time2 = time.time() - start + time1

        # after immunizing the greedy algorithm

        g_greedy = g.__class__()
        g_greedy.add_nodes_from(g)
        g_greedy.add_edges_from(g.edges)

        for node in greedy:
            g_greedy.remove_node(node)

        #  model_greedy = ep.IndependentCascadesModel(g_greedy)

        config_greedy = mc.Configuration()
        config_greedy.add_model_initial_configuration('Infected', infected)

        for a, b in g_greedy.edges():
            weight = config.config["edges"]['threshold'][(a, b)]
            config_greedy.add_edge_configuration('threshold', (a, b), weight)
            g_greedy[a][b]['weight'] = weight

        #  model_greedy.set_initial_status(config_greedy)

        # Simulation 10 times
        result = []

        for i in range(10):

            model_greedy = ep.IndependentCascadesModel(g_greedy)
            model_greedy.set_initial_status(config_greedy)

            iterations_greedy = model_greedy.iteration_bunch(10)
            trends_greedy = model_greedy.build_trends(iterations_greedy)

            infected_greedy = 0

            for i in range(10):
                for j in iterations_greedy[i]['status']:
                    a = iterations_greedy[i]['status'][j]
                    if a == 1:
                        b = cost[j]
                    else:
                        b = 0
                    infected_greedy += b

            effect = (infected_no - infected_greedy - current_greedy) / population

            result.append(effect)

        # print("\nImmuned partition: ", percent)
        # print("\nProtect number of individuals: ", s.mean(result), " +- ", s.stdev(result))
        # print("\nTime: ", time2)

        print("Immuned partition: ", percent, end=', ')
        print("Protect number of individuals: ", s.mean(result), " +- ", s.stdev(result), end=',  ')
        print("Time: ", time2)

    return value, time2