import igraph
import numpy as np
from typing import Union
from collections import defaultdict

def __shift_interval_by(intervals, weight):
    """ Shift the intervals by a weight """
    return [[x + weight, y + weight] for [x, y] in intervals]


def __merge_overlapping_intervals(intervals):
    """ Get overlapping intervals and merge them, returnung less than len(intervals) many intervals. """
    intervals = np.array(intervals)
    starts = intervals[:, 0]
    ends = np.maximum.accumulate(intervals[:, 1])
    valid = np.zeros(len(intervals) + 1, dtype=bool)
    valid[0] = True
    valid[-1] = True
    valid[1:-1] = starts[1:] >= ends[:-1]
    return [list(x) for x in np.vstack((starts[:][valid[:-1]], ends[:][valid[1:]])).T]


def __merge_closest_intervals(intervals):
    """ Get the closest interval and merge those two, returnung exactly len(intervals)-1 many intervals."""
    diff = [y[0]-x[1] for x, y in zip(intervals, intervals[1:])]
    argmin = diff.index(min(diff))

    new_interval = [intervals[argmin][0], intervals[argmin+1][1]]
    return intervals[:argmin] + [new_interval] + intervals[argmin+2:]


def __create_pdb(graph, rev_top_sort, k=5):
    """
    Generates the pdbs (intervals). Each node will have up to k many intervals
    We build it up via the rev. top. sort. Intervals are merged when they overlap.
    If too many intervals are present on a node, then the closest ones will be merged.
    """

    # Initial attribute values:
    graph.vs[rev_top_sort[0]]["pdb"] = [[0, 0]]

    # iterate
    for node in rev_top_sort[1:]:
        intervals = []
        for out_edge in graph.incident(node, mode="OUT"):
            intervals.extend(
                __shift_interval_by(
                    graph.vs[graph.es[out_edge].target]["pdb"],
                    graph.es[out_edge]["weight"]
                )
            )

        sorted_intervals = __merge_overlapping_intervals(sorted(intervals, key=lambda x: x[0]))

        # Merge as long as we do not have less then k many intervals
        while True:
            if len(sorted_intervals) <= k:
                break
            else:
                sorted_intervals = __merge_closest_intervals(sorted_intervals)

        graph.vs[node]["pdb"] = sorted_intervals

    # retrieve entries and delete pdb attribute
    pdb_entries = [x + [[np.nan, np.nan]]*(k - len(x)) for x in graph.vs["pdb"]]
    del graph.vs["pdb"]

    # return pdbs
    return np.array(pdb_entries)


def create_fully_connected_graph(_set :Union[list, set, tuple]):
    """ 
    Generates a fully connected graph from the set.
    Also returns the topological order of the graph.
    """
    _set = list(_set)

    # Generate the initial graph
    graph = igraph.Graph(directed=True)
    top_order = list(range(len(_set)+2))
    graph.add_vertices(len(_set)+2)  # add start and stop node
    graph.vs[1:-1]["label"] = _set

    # Add the corresponding edges
    in_chain_edges = [(i, i+1)for i in range(len(_set)+1)]
    in_chain_weights = _set + [0]

    # Edges to each successive node
    new_set = _set + [0]
    in_edges = [(0, i+2) for i in range(0, len(_set)-1)] + [(k, i+2) for k in range(1, len(_set)) for i in range(k, len(_set))]
    in_weights = [new_set[i+1] for i in range(0, len(_set)-1)] + [new_set[i+1] for k in range(1, len(_set)) for i in range(k, len(_set))]

    # Add edges to the graph
    graph.add_edges(in_chain_edges + in_edges)
    graph.es["weight"] = in_chain_weights + in_weights

    return graph, top_order



def create_partially_fully_connected_graph(_set :Union[list, set, tuple]):
    """ 
    Generates a partially fully connected graph from the set (using sum(range(len(_set)))+2 many nodes)
    Also returns the topological order of the graph.
    """
    _set = list(_set)

    # Generate the initial graph
    graph = igraph.Graph(directed=True)
    top_order = list(range(sum(range(len(_set)+1)) + 2))
    graph.add_vertices(sum(range(len(_set)+1)) + 2)  # add start and stop node
    graph.vs[1:-1]["label"] = [i for b in [_set[idx:] for idx in range(len(_set))] for i in b]
    
    # Iteratre for each possibility
    in_chain_edges = []
    in_chain_weights = []
    offset = 1
    idx_offset = 1
    for idx_offset in range(0, len(_set)-1):  # TODO replace with for loop? on idx_offset? 
        in_chain_edges += [(0, offset)] + [(offset+i, offset+i+1)for i in range(len(_set)-idx_offset-1)] + [(len(_set)+offset-idx_offset-1, graph.vcount()-1)]
        in_chain_weights += _set[idx_offset:] + [0]

        # Do Interconnections
        in_chain_edges += [(offset+i, offset+k+1) for i in range(len(_set)-idx_offset) for k in range(i+1, len(_set)-idx_offset-1)]
        in_chain_weights += [_set[k+1+idx_offset] for i in range(len(_set)-idx_offset) for k in range(i+1, len(_set)-idx_offset-1)]

        # Do chain to end
        in_chain_edges += [(offset+i, graph.vcount()-1)for i in range(len(_set)-idx_offset-1)]
        in_chain_weights += [0]*(len(_set)-idx_offset-1)

        offset += len(_set) - idx_offset

    # Special case for last node remaining
    in_chain_edges += [(0, graph.vcount()-2), (graph.vcount()-2, graph.vcount()-1)]
    in_chain_weights += [_set[-1], 0]

    # Add edges to the graph
    graph.add_edges(in_chain_edges)
    graph.es["weight"] = in_chain_weights

    return graph, top_order


def __func_dist(pdb, s_interval):
    """ Function to decide wheather an interval is overlapping or not in pdb. """
    lower_index = np.searchsorted(pdb[:,1], s_interval[0])
    upper_index = np.searchsorted(pdb[:,0], s_interval[1])

    if upper_index > lower_index  or (len(pdb) != lower_index and lower_index == upper_index and pdb[:,0][lower_index] == s_interval[1]):
        # Also check for edge case on the right side of queried s_interval
        return True
    return False

def query_graph(tv_interval, _graph, _top_sort, _n_pdb):
    """ Retrieve paths using the top. sorted nodes and pdb entries """
    dd = defaultdict(lambda: [[], []])
    tv_interval = np.array(tv_interval)

    dd[_top_sort[0]][0] = [0]
    dd[_top_sort[0]][1] = [[_top_sort[0]]]
    for n in _top_sort[0:-1]:
        if n not in dd:
            continue
        edges = _graph.es.select(_source=n)
        eids = [e.index for e in edges]
        expand_tvs = [_graph.es[e]["weight"] for e in eids]
        achieved_tvs = [[p_tv + e_tv for e_tv in expand_tvs] for p_tv in dd[n][0]]

        # check if we expand
        val_fs = [
            [__func_dist(_n_pdb[out_edge.target, :], tv_interval - e_tv) for out_edge, e_tv in zip(edges, a_tvs)]
            for a_tvs in achieved_tvs
        ]

        # get all with val_f == true and compact them in our queue
        for p, v_fs, a_tvs in zip(dd[n][1], val_fs, achieved_tvs):
            for e, fs, cur_tv in zip(edges, v_fs, a_tvs):
                if fs:
                    dd[e.target][0].append(cur_tv)
                    dd[e.target][1].append([*p, e.target])

        # save memory
        del dd[n]

    return dd[_top_sort[-1]][1]




# test = [-817781123, -2074184841, 794733069, 327907086, -214831599, 1388864021, -162338151, 971804323, 31343141, -924873050, -1138943565, 1317377721, -535653691, 1681487557, 60933704, -1143347511, 608290391, -1920219048, 1692988261, 1796898044]

# test = [1,2,3,4,5]
test = [1299781766, 1549892490, 229664014, 155290526, 1215115934, 2018368555, 976069422, 947710648, 889496249, 1018376771, 1318631235, 198577357, 2090047824, 543544149, 1056557538, 796090084, 144334059, 28505075, 655235828, 1691614199]


g, g_top = create_fully_connected_graph(test)
pdbs = __create_pdb(g, g_top[::-1])
h_test = query_graph([test[12] + test[14]+ test[3]]*2, g, g_top, pdbs)



g_2, g_top_2 = create_partially_fully_connected_graph(test)
pdbs_2 = __create_pdb(g_2, g_top_2[::-1])
h_test_2 = query_graph([test[12] + test[14]+ test[3]]*2, g_2, g_top_2, pdbs_2)


print(h_test)


# h1 = query_graph([test[1]-1, test[1]+1], g, g_top, pdbs)
# h2 = query_graph([test[1], test[1]+1], g, g_top, pdbs)
# h3 = query_graph([test[1]-1, test[1]], g, g_top, pdbs)
# h4 = query_graph([test[1], test[1]], g, g_top, pdbs)




