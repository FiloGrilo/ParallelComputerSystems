from curses.ascii import isdigit
from glob import glob
import re
from threading import local


results_path = "cachetest_results.txt"
global_result = []
with open(results_path) as results:
    for line in results:
        res = line.split(',')
        for s in res[0].split():
            if s.isdigit():
                size = int(s)
        for s in res[1].split():
            if s.isdigit():
                stride = int(s)
        for s in res[2].split():
            try:
                time = float(s)
            except:
                continue
        local_result = [size, stride, time]
        global_result.append(local_result)

print(global_result)