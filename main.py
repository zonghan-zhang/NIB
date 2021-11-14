
from baseline_greddy_improved import greedy_improved
from baseline_greedy_ori import greedy_ori
from baseline_random import rand
from deepIBM import deepIBM
from util import net_gen

import tqdm

# disable tqdm progress bar
def nop(it, *a, **k):
    return it
tqdm.tqdm = nop

# generate graph
g, cost, population, infected, infected_no, mc, config = net_gen()

print("Graph is alive\n==========")

# print("Random")
# rand(g, cost, population, infected, infected_no, config)
#
# print("\n======")
# print("greedy_ori")
# greedy_ori(g, cost, population, infected, infected_no, mc, config)

# print("\n======")
print("greedy_improved")
value, time2 = greedy_improved(g, cost, population, infected, infected_no, mc, config)

print("\n======")
print("deepIBM")
deepIBM(g, cost, population, infected, infected_no, mc, config, value, time2)

