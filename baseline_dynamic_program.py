import time
import statistics as s

# dynamic programming using dictionaries value and cost


def dp(g, cost, population, infected, infected_no, mc, config):
    for percent in range(5, 55, 5):

        start = time.time()

        W = int(population * percent / 100)
        n = 5000 - 50

        v = []
        c = []
        p = []

        j = 0
        i = 0

        for i in g.nodes:
            if i not in infected:
                v.append(value[i])
                c.append(cost[i])
                p.append(i)

        K = [[0 for w in range(W + 1)]
             for i in range(n + 1)]

        for i in range(n + 1):
            for w in range(W + 1):
                if i == 0 or w == 0:
                    K[i][w] = 0
                elif c[i - 1] <= w:
                    K[i][w] = max(v[i - 1] + K[i - 1][w - c[i - 1]], K[i - 1][w])
                else:
                    K[i][w] = K[i - 1][w]

        res = K[n][W]

        dp = []

        w = W
        i = 4950
        while i > 0:
            k = K[i][w]
            if (k != K[i - 1][w]):
                dp.append(p[i - 1])
                w -= cost[p[i - 1]]
            i -= 1

        current_dp = 0

        for node in dp:
            current_dp += cost[node]

        period = time.time() - start + time1

        # after immunizing the greedy algorithm

        g_dp = g.__class__()
        g_dp.add_nodes_from(g)
        g_dp.add_edges_from(g.edges)

        for node in dp:
            g_dp.remove_node(node)

        config_dp = mc.Configuration()
        config_dp.add_model_initial_configuration('Infected', infected)

        for a, b in g_dp.edges():
            weight = config.config["edges"]['threshold'][(a, b)]
            config_dp.add_edge_configuration('threshold', (a, b), weight)
            g_dp[a][b]['weight'] = weight

        # Simulation 10 times
        result = []

        for i in range(10):

            model_dp = ep.IndependentCascadesModel(g_dp)
            model_dp.set_initial_status(config_dp)

            iterations_dp = model_dp.iteration_bunch(10)
            trends_dp = model_dp.build_trends(iterations_dp)

            infected_dp = 0

            for i in range(10):
                for j in iterations_dp[i]['status']:
                    a = iterations_dp[i]['status'][j]
                    if a == 1:
                        b = cost[j]
                    else:
                        b = 0
                    infected_dp += b

            effect = (infected_no - infected_dp - current_dp) / population

            result.append(effect)

        # print("\nImmuned partition: ", percent)
        # print("\nProtect number of individuals: ", s.mean(result), " +- ", s.stdev(result))
        # print("\nTime: ", period)
        # print("\nTest: ", w)

        print("Immuned partition: ", percent, end=', ')
        print("Protect number of individuals: ", s.mean(result), " +- ", s.stdev(result), end=',  ')
        print("Time: ", period)