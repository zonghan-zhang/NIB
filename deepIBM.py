import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
import numpy as np
import time
import statistics as s
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc


# To-Do
# skip-connection -> not work
# initialization ->
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(2, 1)
        # self.linear2 = nn.Linear(3, 1)

    def forward(self, x):
        z = self.linear1(x)
        # y = torch.cat((z, x), 1)
        # y = self.linear2(y)
        return torch.sigmoid(z)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.constant_(m.weight, val=1)
        m.bias.data.fill_(0.01)


def deepIBM(g, cost, population, infected, infected_no, mc, config, value, time2):
    v = []
    c = []
    p = []

    for i in g.nodes:
        if i not in infected:
            v.append(float(value[i]))
            c.append(float(cost[i]))
            p.append(i)

    X = np.array(list(zip(v, c)))

    X = torch.from_numpy(X)
    v = torch.FloatTensor(v)
    c = torch.FloatTensor(c)

    for percent in range(5, 95, 5):

        start = time.time()
        k = int(population * percent / 100)

        model = Net()
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()

        for epoch in range(200):
            optimizer.zero_grad()
            y = model(X.float())
            loss = max(0, torch.matmul(c, y) - k) - 7 * torch.matmul(v, y)
            loss.backward()
            optimizer.step()

        z = y.detach().numpy()
        pr = {}
        for i in range(4950):
            pr[p[i]] = z[i][0]

        sorted_pr = []
        for node in sorted(pr, key=pr.get, reverse=False):
            sorted_pr.append(node)

        current_dl = 0
        dl = []

        for node in sorted_pr:
            C = cost[node]
            if (current_dl + C <= k):
                dl.append(node)
                current_dl += C
            else:
                continue

        period = time.time() - start + time2

        # after immunizing with the dl algorithm

        g_dl = g.__class__()
        g_dl.add_nodes_from(g)
        g_dl.add_edges_from(g.edges)

        for node in dl:
            g_dl.remove_node(node)

        config_dl = mc.Configuration()
        config_dl.add_model_initial_configuration('Infected', infected)

        for a, b in g_dl.edges():
            weight = config.config["edges"]['threshold'][(a, b)]
            config_dl.add_edge_configuration('threshold', (a, b), weight)
            g_dl[a][b]['weight'] = weight

        # Simulation 10 times
        result = []

        for ii in range(10):

            model_dl = ep.IndependentCascadesModel(g_dl)
            model_dl.set_initial_status(config_dl)

            iterations_dl = model_dl.iteration_bunch(10)
            trends_dl = model_dl.build_trends(iterations_dl)

            infected_dl = 0

            for i in range(10):
                for j in iterations_dl[i]['status']:
                    a = iterations_dl[i]['status'][j]
                    if a == 1:
                        b = cost[j]
                    else:
                        b = 0
                    infected_dl += b

            effect = (infected_no - infected_dl - current_dl) / population

            result.append(effect)

        print("Immuned partition: ", percent, end=', ')
        print("Protect number of individuals: ", s.mean(result), " +- ", s.stdev(result), end=',  ')
        print("Time: ", period)
