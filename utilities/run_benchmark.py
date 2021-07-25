from matplotlib import pyplot
import timeit

from generate_sets import get_set

from collections import defaultdict
import sssgraph as sg
import random

import seaborn as sb

import pandas as pd

import tqdm

# TODO


def time_full_unsorted(_s, interval):
    sg.query_with_fully_connected_graph(_s, interval)


def time_full_sorted_asc(_s, interval):
    sg.query_with_fully_connected_graph(sorted(_s, reverse=False), interval)


def time_full_sorted_desc(_s, interval):
    sg.query_with_fully_connected_graph(sorted(_s, reverse=True), interval)



def time_partial_unsorted(_s, interval):
    sg.query_with_paritally_fully_connected_graph(_s, interval)


def time_partial_sorted_asc(_s, interval):
    sg.query_with_paritally_fully_connected_graph(sorted(_s, reverse=False), interval)


def time_partial_sorted_desc(_s, interval):
    sg.query_with_paritally_fully_connected_graph(sorted(_s, reverse=True), interval)






def benchmark(list_of_funcs, repititions = 30,  lower = 5, upper = 21):
    # TODO implement maximal time timeout!

    # Set benchmark result dicts
    dicts = [defaultdict(list) for _ in list_of_funcs]

    # Get the benchmarks
    for i in tqdm.tqdm(range(lower, upper)):
        _s = list(get_set(number_of_elements=i))
        for _ in tqdm.tqdm(range(repititions), leave=False):
            interval = [sum(random.sample(_s, random.randint(0, len(_s))))]*2
            for d, func in zip(dicts, list_of_funcs):
                d[i].append(timeit.timeit(lambda: func(_s, interval), number=1))


    # convert information into lists for the pandas frame
    t_iter = []
    t_func = []
    t_time = []
    t_set_size = []
    for d, f in zip(dicts, list_of_funcs):
        for key, val in d.items():
            t_iter.extend(list(range(len(val))))
            t_func.extend([f.__name__]* len(val) )
            t_set_size.extend([key]* len(val) )
            t_time.extend(val)

    # return pandas frame
    return pd.DataFrame({ "func": t_func, "iter": t_iter, "time": t_time, "setsize": t_set_size})

if __name__ == "__main__":
    # Parameter: number of repitions
    # Parameter: timeout limit in seconds (maximal execution time)

    # Benchmark all possibilities
    t = benchmark([
        time_full_unsorted,
        time_full_sorted_asc,
        time_full_sorted_desc,
        time_partial_unsorted,
        time_partial_sorted_asc,
        time_partial_sorted_desc,
    ], repititions=20)
    sb.lineplot(data=t, x="setsize", y="time", hue="func")
    pyplot.show()


    # Benchmark  only descending ones! (which is faster for sets with only positive numbers!)
    t = benchmark([
        time_full_sorted_desc,
        time_partial_sorted_desc,
    ], repititions=20, upper=30)
    sb.lineplot(data=t, x="setsize", y="time", hue="func")
    pyplot.show()