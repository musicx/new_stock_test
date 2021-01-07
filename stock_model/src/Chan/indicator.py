import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from Chan.data_structure import *

class ChanIndicator(object):
    def __init__(self):
        pass

    def intersect(self, low1, high1, low2, high2):
        return high1 > low1 and high2 > low2 and min(high1, high2) > max(low1, low2)

    def near_pivot_lines(self, chart_frame, recent=3, limit=3):
        print('finding pivot lines for {}'.format(chart_frame.code))
        lines = chart_frame.detect_price_boundary()
        detected_idx = {}
        for candle in chart_frame.raw_candles[-recent:]:
            for idx, line in enumerate(lines):
                if self.intersect(candle.low, candle.high, line*0.99, line*1.01):
                    if (idx not in detected_idx):
                        detected_idx[idx] = 1
                    else:
                        detected_idx[idx] += 1
        within_idx = [idx for idx in detected_idx
                      if max(abs(chart_frame.raw_candles[-1].high - lines[idx]),
                             abs(chart_frame.raw_candles[-1].low - lines[idx])) / lines[idx] < limit / 100]
        line_distance = [(idx, abs(chart_frame.raw_candles[-1].close - lines[idx]) / lines[idx]) for idx in within_idx]
        line_distance.sort(key=lambda x: x[1])
        if len(line_distance) > 0:
            return lines[line_distance[0][0]]
        else:
            return -1