import networkx as nx
import ndlib
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import random
import statistics as s


# Network Gneration

def net_gen():
    # g = nx.erdos_renyi_graph(5000, 0.005)
    g = nx.barabasi_albert_graph(5000, 10)

    cost = {}
    for i in range(5000):
        C = random.randrange(1, 3)
        C = C * g.degree()[i]
        cost[i] = C

    infected = random.sample(range(0, 5000), 50)

    # Model Selection
    model = ep.IndependentCascadesModel(g)

    config = mc.Configuration()
    config.add_model_initial_configuration('Infected', infected)

    for a, b in g.edges():
        weight = random.randrange(1, 30)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    model.set_initial_status(config)

    # Simulation
    iterations = model.iteration_bunch(10)
    trends = model.build_trends(iterations)

    total_no = 0

    for i in range(10):
        a = iterations[i]['node_count'][1]
        total_no += a
        # print(a)

    print('total node #: ', total_no)

    infected_no = 0

    for i in range(10):
        for j in iterations[i]['status']:
            a = iterations[i]['status'][j]
            if a == 1:
                b = cost[j]
            else:
                b = 0
            infected_no += b

    print("infected total: ", infected_no)

    population = 0

    for i in cost:
        population += cost[i]

    print("population: ", population)

    degrees_no = [val for (node, val) in g.degree()]

    print("average degree: ", s.mean(degrees_no))

    return g, cost, population, infected, infected_no, mc, config
