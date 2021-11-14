import random

import numpy as np
import time

import statistics as s
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc


## Random Select Nodes Baseline

def rand(g, cost, population, infected, infected_no, config):

    for percent in range(5, 55, 5):

        k = int(population * percent / 100)

        result = []
        times = []

        # Repeat 10 times

        for t in range(10):

            start = time.time()

            randomlist = [None] * 5000
            for i in range(5000):
                randomlist[i] = i

            random.shuffle(randomlist)

            current_ran = 0
            ran = []

            for node in randomlist:
                C = cost[node]
                if (node not in infected) and (current_ran + C <= k):
                    ran.append(node)
                    current_ran += C
                else:
                    continue

            period = time.time() - start

            # Make a copy of graph g

            g_ran = g.__class__()
            g_ran.add_nodes_from(g)
            g_ran.add_edges_from(g.edges)

            for node in ran:
                g_ran.remove_node(node)

            model_ran = ep.IndependentCascadesModel(g_ran)

            config_ran = mc.Configuration()
            config_ran.add_model_initial_configuration('Infected', infected)

            for a, b in g_ran.edges():
                weight = config.config["edges"]['threshold'][(a, b)]
                config_ran.add_edge_configuration('threshold', (a, b), weight)
                g_ran[a][b]['weight'] = weight

            model_ran.set_initial_status(config_ran)

            # Simulation
            iterations_ran = model_ran.iteration_bunch(10)
            trends_ran = model_ran.build_trends(iterations_ran)

            infected_ran = 0

            for i in range(10):
                for j in iterations_ran[i]['status']:
                    a = iterations_ran[i]['status'][j]
                    if a == 1:
                        b = cost[j]
                    else:
                        b = 0
                    infected_ran += b

            effect = (infected_no - infected_ran - current_ran) / population
            result.append(effect)
            times.append(period)

        print("Immuned partition: ", percent, end=', ')
        print("Protect number of individuals: ", s.mean(result), " +- ", s.stdev(result), end=', ')
        print("Time for node selection: ", s.mean(times), " +- ", s.stdev(times))
