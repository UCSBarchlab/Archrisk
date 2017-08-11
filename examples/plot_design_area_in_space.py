import csv
from utils.plotting import PlotHelper

f = open('LPHC_perf.csv', 'rt')
reader = csv.reader(f)
for row in reader:
    for design in row:
        cores = [int(n) for n in design[:design.rfind('(')].split(',')]
        left = int(design[design.rfind('(')+1:design.rfind(')')])
        assert len(cores) == 6
        xs = []
        xs.append(8 * cores[0])
        xs.append(16 * cores[1])
        xs.append(32 * cores[2])
        xs.append(64 * cores[3])
        xs.append(128 * cores[4])
        if left < 8:
            xs.insert(0, left)
        elif left == 8:
            xs[0] += left
        elif left < 16:
            xs.insert(1, left)
        elif left == 16:
            xs[1] += left
        elif left < 32:
            xs.insert(2, left)
        elif left == 32:
            xs[2] += left
        elif left < 64:
            xs.insert(3, left)
        elif left == 64:
            xs[3] += left
        elif left < 128:
            xs.insert(4, left)
        elif left == 128:
            xs[4] += left
        elif left > 128:
            xs.append(left)
        ys = sorted(xs)
        print xs
        PlotHelper.plot_core_area_dist(xs, 'blue', design)
f.close()
