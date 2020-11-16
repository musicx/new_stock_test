import numpy as np


def higher(x, y):
    return x.high >= y.high and x.low >= y.low


def contain(x, y):
    return x.high >= y.high and x.low <= y.low


def lower(x, y):
    return x.high <= y.high and x.low <= y.low



def find_endpoints(klines):
    end = []  # item is (idx in merged list, is top end point, valid end point)
    last = 0
    for idx, kline in enumerate(klines[:-1]):
        insert = False
        if (idx == 0 or higher(kline, klines[idx - 1])) and higher(kline, klines[idx + 1]):
            end.append([idx, True, True])
            insert = True
        elif (idx == 0 or lower(kline, klines[idx - 1])) and lower(kline, klines[idx + 1]):
            end.append([idx, False, True])
            insert = True

        if insert and len(end) > 1:
            dist = ((idx - 1 == last and klines[last].right + kline.left > 2)
                    or (idx - 2 == last and klines[last].right + klines[last + 1].period() + kline.left > 2)
                    or (idx - 3 == last and klines[last].right + klines[last + 1].period() + klines[last+2].period() + kline.left > 2)
                    or idx - 3 > last)
            if end[-1][1] == end[-2][1]:
                if ((end[-1][1] and klines[end[-1][0]].high > klines[end[-2][0]].high) or
                        (not end[-1][1] and klines[end[-1][0]].low < klines[end[-2][0]].low)):
                    if dist:
                        end[-2][2] = False
                    else:
                        end.pop(-2)
                else:
                    end.pop()
                    insert = False
            elif not dist:
                end.pop()
                insert = False

        if insert:
            last = idx
    return end


def find_cycle(points, klines, mean_allowance=0.1, max_allowance=4, keep_invalid=False, use_merge=True):
    end_idx = dict([(p[0], (p[1], p[2])) for p in points if keep_invalid or p[2]])
    cumu = 0
    slices = []   # item is of (# or ticks, is Peak?)
    for idx, kline in enumerate(klines):
        if idx in end_idx:
            if cumu != 0:
                slices.append((cumu, False))
            if use_merge:
                slices.append((kline.period(), True))
            else:
                slices.append((kline.left, False))
                slices.append((1, True))
                slices.append((kline.right, False))
            cumu = 0
        else:
            cumu += kline.period()
    if len(slices) < 3:
        return []
    ticks = []
    for cnt, flag in slices:
        ticks.extend([flag] * cnt)
    start_point = slices[0][0] + slices[1][0]    # TODO maybe a different start point would result in a different cycle

    cycles = []   # modified items: length, avg, std
    for cycle in range(3, 120):
        length = []
        # print('cycle {}'.format(cycle))
        short = int(max(np.floor(cycle * (1 - mean_allowance)), cycle - max_allowance))
        long = int(min(np.ceil(cycle * (1 + mean_allowance)), cycle + max_allowance))
        found = False
        point = start_point
        before = point
        after = point + slices[2][0]
        while not found:
            if np.all(ticks[point:point+short]) and np.all(ticks[point:point+long]):
                break
                # always in peak phrase
            try:
                last_point = point
                if np.any(ticks[point+short: point+long]):
                    peaks = [short+i for i, p in enumerate(ticks[point+short: point+long]) if p]
                    min_distance = np.min([np.abs(cycle - d) for d in peaks])
                    point = point+cycle+min_distance if ticks[point+cycle+min_distance] else point+cycle-min_distance
                    for p in range(point-1, last_point+short-1, -1):
                        if ticks[p] and p > last_point:
                            before = p     # earliest peak point in full range
                    for p in range(point, last_point+long):
                        if ticks[p] and p > last_point:
                            after = p      # latest peak point in full range
                    length.append(point - last_point)
                elif np.any(ticks[before+short: point+short]):
                    delta = int((point - before) / 2.)
                    for p in range(last_point+short-1, before+short-1, -1):
                        if ticks[p] and p > last_point:
                            before = p
                    for p in range(before+short, last_point+short):
                        if ticks[p] and p > last_point:
                            point = p
                            after = p
                    if len(length) > 1:
                        length[-1] -= delta
                    length.append(point - last_point + delta)
                elif np.any(ticks[point+long: after+long]):
                    delta = int((after - point) / 2.)
                    for p in range(after+long-1, last_point+long-1, -1):
                        if ticks[p] and p > last_point:
                            before = p
                            point = p
                    for p in range(last_point+long, after+long):
                        if ticks[p] and p > last_point:
                            after = p
                    if len(length) > 1:
                        length[-1] += delta
                    length.append(point - last_point - delta)
                elif point < len(ticks):
                    break
                    # cannot find peak in given range
                if last_point == point:
                    break
                    # cannot move forward
            except IndexError as e:
                found = True
        if found and len(length) > 0:
            cycles.append((cycle, np.mean(length), np.std(length)))
    mean = []
    for cycle in cycles:
        mc = int(round(cycle[1]))
        if mc not in mean:
            mean.append(mc)
    return mean


def find_pair_cycle(cycles):
    # cycle_length = [x[0] for x in cycles]
    cycle_length = cycles
    pcycles = []
    passed = []
    cycle_set = set(cycle_length)
    for cycle in cycle_length:
        if cycle in passed:
            continue
        cand = [cycle]
        c = cycle
        while True:
            if c * 2 in cycle_set:
                c = c * 2
                cand.append(c)
            elif c * 2 + 1 in cycle_set:
                c = c * 2 + 1
                cand.append(c)
            elif c * 2 - 1 in cycle_set:
                c = c * 2 - 1
                cand.append(c)
            elif c * 2 + 2 in cycle_set:
                c = c * 2 + 2
                cand.append(c)
            elif c * 2 - 2 in cycle_set:
                c = c * 2 - 2
                cand.append(c)
            else:
                break
        if len(cand) > 1:
            pcycles.append(cand)
            passed.extend(cand)
    return pcycles

