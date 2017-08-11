import csv
from utils.plotting import PlotHelper

f = open('LPHC_perf.csv', 'rt')
reader = csv.reader(f)
for row in reader:
    for design in row:
        cores = [int(n) for n in design[:design.rfind('(')].split(',')]
        left = 1. * int(design[design.rfind('(')+1:design.rfind(')')])
        assert len(cores) == 6
        xs = []
        xs.extend([1] * cores[0])
        xs.extend([2] * cores[1])
        xs.extend([3] * cores[2])
        xs.extend([4] * cores[3])
        xs.extend([5] * cores[4])
        if left <= 8:
            left = (left / 8)
        elif left <= 16:
            left = 1 +  (left / 16)
        elif left <= 32:
            left = 2 + (left / 32)
        elif left <= 64:
            left = 3 + (left / 64)
        elif left <= 128:
            left = 4 + (left / 128)
        elif left > 128:
            left = 4 + (left / 128)
        xs.append(left)
        ys = sorted(xs)
        print xs
        PlotHelper.plot_core_dist(xs, 'blue', design)
f.close()
