import time
import numpy as np
import networkx as nx
import ndlib.models.epidemics as ep
import statistics as s


# Greedy Algorithm. Value calculated by sigma (A + A^2 + ... + A^10): Yan, R., Li, D., Wu, W., Du, D. Z., & Wang, Y. (2019). Minimizing influence of rumors by blockers on social networks: algorithms and analysis.

def greedy_ori(g, cost, population, infected, infected_no, mc, config):

  start = time.time()

  I = np.ones((5000, 1))

  F = np.ones((5000, 5000))
  N = np.ones((5000, 5000))

  A = nx.to_numpy_matrix(g, nodelist=list(range(5000)))

  sigma = I
  for i in range(10):
    B = np.power(A, i+1)
    C = np.matmul(B, I)
    sigma += C

  value = {}

  for i in range(5000):
    value[i] = sigma[i, 0]

  # v/w in knapsack

  unit_v = {}

  for i in range(5000):
    unit_v[i] = value[i]/cost[i]

  sorted_4_greedy = []

  for node in sorted(unit_v, key=unit_v.get, reverse=True):
    sorted_4_greedy.append(node)

  time1 = time.time() - start

  for percent in range(5, 55, 5):

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

    config_greedy = mc.Configuration()
    config_greedy.add_model_initial_configuration('Infected', infected)

    for a, b in g_greedy.edges():
      weight = config.config["edges"]['threshold'][(a, b)]
      config_greedy.add_edge_configuration('threshold', (a, b), weight)
      g_greedy[a][b]['weight'] = weight

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

      effect = (infected_no - infected_greedy - current_greedy)/population

      result.append(effect)

    print("Immuned partition: ", percent, end=', ')
    print("Protect number of individuals: ", s.mean(result), " +- ", s.stdev(result), end=', ')
    print("Time: ", time2)



    # return percent, s.mean(result), s.stdev(result), time2