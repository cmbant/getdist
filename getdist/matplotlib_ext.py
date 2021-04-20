from matplotlib import ticker
from matplotlib.axis import YAxis
import math
import numpy as np
from bisect import bisect_left


class SciFuncFormatter(ticker.Formatter):
    # To put full sci notation into each axis label rather than split offsetText

    def __call__(self, x, pos=None):
        return "${}$".format(self._format_sci_notation('%.10e' % x))

    def format_data(self, value):
        # e.g. for the navigation toolbar, no latex
        return '%-8g' % value

    @staticmethod
    def _format_sci_notation(s):
        # adapted from old matplotlib
        # transform 1e+004 into 1e4, for example
        tup = s.split('e')
        try:
            significand = tup[0].rstrip('0').rstrip('.')
            sign = tup[1][0].replace('+', '')
            exponent = tup[1][1:].lstrip('0')
            if significand == '1' and exponent != '':
                # reformat 1x10^y as 10^y
                significand = ''
            if exponent:
                exponent = '10^{%s%s}' % (sign, exponent)
            if significand and exponent:
                return r'%s{\times}%s' % (significand, exponent)
            else:
                return r'%s%s' % (significand, exponent)
        except IndexError:
            return s


_min_label_len_chars = 1.35


class BoundedMaxNLocator(ticker.MaxNLocator):
    # Tick locator class that only returns ticks within bounds, and if pruned, pruned not to overlap ends of axes
    # Also tries to correct for long tick labels and avoid large tick-free gaps at ends of axes, and to get at
    # least two ticks where possible (even if it requires odd spacing or off-phase ticks)

    def __init__(self, nbins='auto', prune=True, step_groups=([1, 2, 5, 10], [2.5, 3, 4, 6, 8], [1.5, 7, 9])):
        self.bounded_prune = prune
        self._step_groups = [_staircase(np.array(steps), np.array(steps)) for steps in step_groups]
        self._offsets = []
        for g in step_groups:
            g2 = []
            for x in g:
                if x % 2 < 1e-6:
                    g2.append(x // 2)
                else:
                    g2.append(0)
            self._offsets.append(_staircase(np.array(g2), g))
        super().__init__(nbins=nbins, steps=step_groups[0])

    def _bounded_prune(self, locs, label_len):
        if len(locs) > 1 and self.bounded_prune:
            if locs[0] - self._range[0] < label_len * 0.5:
                locs = locs[1:]
            if self._range[1] - locs[-1] < label_len * 0.5 and len(locs) > 1:
                locs = locs[:-1]
        return locs

    def _get_label_len(self, locs):
        if not len(locs):
            return 0
        self._formatter.set_locs(locs)
        # get non-latex version of label
        form = self._formatter.format
        i = form.index('%')
        i2 = form.index('f', i)
        label = form[i:i2 + 1] % locs[0]
        char_len = len(label)
        if '.' in label:
            char_len -= 0.4
        if len(locs) > 1:
            label = form[i:i2 + 1] % locs[-1]
            char_len2 = len(label)
            if '.' in label:
                char_len2 -= 0.4
            char_len = max(char_len, char_len2)

        return max(_min_label_len_chars, char_len * self._font_aspect) * self._char_size_scale

    def tick_values(self, vmin, vmax):
        # Max N locator will produce locations outside vmin, vmax, so even if pruned
        # there can be points very close to the actual bounds. Let's cut them out.
        # Also account for tick labels with aspect ratio > 3 (default often-violated heuristic)
        # - use better heuristic based on number of characters in label and typical font aspect ratio

        axes = self.axis.axes
        tick = self.axis._get_tick(True)
        rotation = tick._labelrotation[1]

        if isinstance(self.axis, YAxis):
            rotation += 90
            ends = axes.transAxes.transform([[0, 0], [0, 1]])
            length = ((ends[1][1] - ends[0][1]) / axes.figure.dpi) * 72
        else:
            ends = axes.transAxes.transform([[0, 0], [1, 0]])
            length = ((ends[1][0] - ends[0][0]) / axes.figure.dpi) * 72
        size_ratio = tick.label1.get_size() / length
        cos_rotation = abs(math.cos(math.radians(rotation)))
        self._font_aspect = 0.65 * cos_rotation
        self._char_size_scale = size_ratio * (vmax - vmin)
        self._formatter = self.axis.major.formatter
        self._range = (vmin, vmax)

        # first guess
        if cos_rotation > 0.05:
            label_len = size_ratio * 1.5 * (vmax - vmin)
            label_space = label_len * 1.1
        else:
            # text orthogonal to axis
            label_len = size_ratio * _min_label_len_chars * (vmax - vmin)
            label_space = label_len * 1.25

        delta = label_len / 2 if self.bounded_prune else 0
        nbins = int((vmax - vmin - 2 * delta) / label_space) + 1
        if nbins > 4:
            # use more space for ticks
            nbins = int((vmax - vmin - 2 * delta) / ((1.5 if nbins > 6 else 1.3) * label_space)) + 1
        min_n_ticks = min(nbins, 2)
        nbins = min(self._nbins if self._nbins != 'auto' else 9, nbins)
        # First get typical ticks so we can calculate actual label length
        while True:
            locs, _ = self._spaced_ticks(vmin + delta, vmax - delta, label_len, min_n_ticks, nbins, False)
            if len(locs) or min_n_ticks == 1:
                break
            if nbins == 2:
                min_n_ticks -= 1
            nbins = max(min_n_ticks, 2)
        if cos_rotation > 0.05 and isinstance(self._formatter, ticker.ScalarFormatter) and len(locs) > 1:

            label_len = self._get_label_len(locs)
            locs = self._bounded_prune(locs, label_len)
            if len(locs) > 1:
                step = locs[1] - locs[0]
            # noinspection PyUnboundLocalVariable
            if len(locs) < max(3, nbins) or step < label_len * (1.1 if len(locs) < 4 else 1.5) \
                    or (locs[0] - vmin > min(step * 1.01, label_len * 1.5) or
                        vmax - locs[-1] > min(step * 1.01, label_len * 1.5)):
                # check for long labels, labels that are too tightly spaced, or large tick-free gaps at axes ends
                delta = label_len / 2 if self.bounded_prune else 0
                for fac in [1.5, 1.35, 1.1]:
                    nbins = int((vmax - vmin - 2 * delta) / (fac * max(2 * self._char_size_scale, label_len))) + 1
                    if nbins >= 4:
                        break
                if self._nbins != 'auto':
                    nbins = min(self._nbins, nbins)
                min_n_ticks = min(min_n_ticks, nbins)
                retry = True
                try_shorter = True
                locs = []
                while min_n_ticks > 1:
                    locs, good = self._spaced_ticks(vmin + delta, vmax - delta, label_len, min_n_ticks, nbins)
                    if len(locs):
                        if not good:
                            new_len = self._get_label_len(locs)
                            if not np.isclose(new_len, label_len):
                                label_len = new_len
                                delta = label_len / 2 if self.bounded_prune else 0
                                if retry:
                                    retry = False
                                    continue
                                locs = self._bounded_prune(locs, label_len)
                    elif min_n_ticks > 1 and try_shorter:
                        # Original label length may be too long for good ticks which exist
                        delta /= 2
                        label_len /= 2
                        try_shorter = False
                        locs, _ = self._spaced_ticks(vmin + delta, vmax - delta, label_len, min_n_ticks, nbins)
                        if len(locs):
                            label_len = self._get_label_len(locs)
                            delta = label_len / 2 if self.bounded_prune else 0
                            continue

                    if min_n_ticks == 1 and len(locs) == 1 or len(locs) >= min_n_ticks > 1 \
                            and locs[1] - locs[0] > self._get_label_len(locs) * 1.1:
                        break
                    min_n_ticks -= 1
                    locs = []
                if len(locs) <= 1 and size_ratio * self._font_aspect < 0.9:
                    scale, offset = ticker.scale_range(vmin, vmax, 1)
                    # Try to get any two points that will fit
                    for sc in [scale, scale / 10.]:
                        locs = [round((vmin * 3 + vmax) / (4 * sc)) * sc,
                                round((vmin + 3 * vmax) / (4 * sc)) * sc]
                        if locs[0] != locs[1] and locs[0] >= vmin and locs[1] <= vmax:
                            if self._valid(locs):
                                return locs
                    # if no ticks, check for short integer number location in the range that may have been missed
                    # because adding any other values would make label length much longer
                    loc = round((vmin + vmax) / (2 * scale)) * scale
                    if vmin < loc < vmax:
                        locs = [loc]
                        label_len = self._get_label_len(locs)
                        return self._bounded_prune(locs, label_len)
        else:
            return self._bounded_prune(locs, label_len)

        return locs

    def _valid(self, locs):
        label_len = self._get_label_len(locs)
        return (len(locs) < 2 or locs[1] - locs[0] > label_len * 1.1) and \
               (not self.bounded_prune or (locs[0] - self._range[0] > label_len / 2)
                and (self._range[1] - locs[-1] > label_len / 2))

    def _spaced_ticks(self, vmin, vmax, _label_len, min_ticks, nbins, changing_lengths=True):

        scale, offset = ticker.scale_range(vmin, vmax, nbins)
        _vmin = vmin - offset
        _vmax = vmax - offset
        _range = _vmax - _vmin
        eps = _range * 1e-6
        _full_range = self._range[1] - self._range[0]
        for sc in [100, 10, 1]:
            round_center = round((_vmin + _vmax) / (2 * sc * scale)) * sc * scale
            if _vmin - eps <= round_center <= _vmax + eps:
                break

        label_len = _label_len * 1.1
        raw_step = max(label_len, _range / ((nbins - 2) if nbins > 2 else 1))
        raw_step1 = _range / max(1, (nbins - (0 if self.bounded_prune else 1)))
        best = []
        best_score = -np.infty
        for step_ix, (_steps, _offsets) in enumerate(zip(self._step_groups, self._offsets)):

            steps = _steps * scale
            if step_ix and len(best) < 3:
                raw_step = max(raw_step, _range / 2)

            istep = min(len(steps) - 1, bisect_left(steps, raw_step))
            if not istep:
                continue
            # This is an upper limit; move to smaller or half-phase steps if necessary.
            for off in [False, True]:
                if off and (len(best) > 2 or len(best) == 2 and (not round_center or step_ix > 1)):
                    break
                for i in reversed(range(istep + 1)):
                    if off and not _offsets[i]:
                        continue
                    step = steps[i]
                    if step < label_len:
                        break

                    if step_ix and _vmin <= round_center <= _vmax:
                        # For less nice steps, try to make them hit any round numbers in range
                        best_vmin = round_center - ((round_center - _vmin) // step) * step
                    else:
                        best_vmin = (_vmin // step) * step

                    if off:
                        # try half-offset steps, e.g. to get -x/2, x/2 as well as -x,0,x
                        low = scale * _offsets[i]
                        if best_vmin - low >= _vmin:
                            best_vmin -= low
                        else:
                            best_vmin += low

                    sc = 10 ** (math.log10(step) // 1)
                    step_int = round(step / sc)

                    low = _ge(_vmin - best_vmin, offset, step)
                    high = _le(_vmax - best_vmin, offset, step)
                    if min_ticks <= high - low + 1 <= nbins:
                        ticks = np.arange(low, high + 1) * step + (best_vmin + offset)

                        if off and round_center and changing_lengths:
                            # If no nice number, see if we can shift points to get one
                            if step > 2 * sc:
                                for shift in [0, -1, 1, -2, 2]:
                                    if abs(shift * sc) >= step / 2:
                                        break
                                    shifted = ticks + shift * sc
                                    if any(np.round(shifted / sc / 10) * 10 == np.round(shifted / sc)) \
                                            and self._valid(shifted):
                                        ticks = shifted

                        big_step = step > raw_step1 and step > label_len * 1.5
                        no_more_ticks = min(3, len(ticks)) <= len(best)
                        odd_gaps = min_ticks > 1 and ((len(ticks) == 2 and step > _full_range * 0.7)
                                                      or self.bounded_prune and (
                                                          (ticks[0] - self._range[0] > max(min(_full_range / 3, step),
                                                                                           label_len * 1.1) or
                                                           self._range[1] - ticks[-1] > max(min(_full_range / 3, step),
                                                                                            label_len * 1.1)))
                                                      or not self.bounded_prune and len(ticks) == 3 and
                                                      step > max(2 * label_len, _full_range / 3)
                                                      and step_int > 1 and round(ticks[-1] / sc) % 10 > 0)

                        close_ticks = step < label_len * 1.3 and len(ticks) > 2
                        if (big_step and odd_gaps or close_ticks) and no_more_ticks:
                            continue
                        if len(best) and odd_gaps and step_ix or changing_lengths and not self._valid(ticks):
                            continue

                        too_few_points = (len(ticks) < 3 and (nbins > (3 if step_ix else 4)) or (
                                len(ticks) < max(2, (nbins + 1) // 2))) and step > label_len * 1.5
                        _score = -1 * too_few_points - step_ix * 2 - close_ticks * 2 - odd_gaps * 1
                        if len(ticks) < 3 and big_step:
                            _score -= 2
                        if off:
                            _score -= 3
                        if step_int == 1.0 and not off:
                            _score += 1
                        if 0. in steps:
                            _score += 1
                        if _score <= best_score:
                            continue
                        if off and not step_ix or big_step and (not len(best) or len(ticks) < len(best)) \
                                or close_ticks or too_few_points or odd_gaps:
                            # prefer spacing where some ticks nearish the ends and ticks not too close in centre
                            best = ticks
                            best_score = _score
                        else:
                            return ticks, True
        return best, False


def _closeto(ms, edge, offset, step):
    if offset > 0:
        digits = np.log10(offset / step)
        tol = max(1e-10, 10 ** (digits - 12))
        tol = min(0.4999, tol)
    else:
        tol = 1e-10
    return abs(ms - edge) < tol


def _le(x, offset, step):
    """Return the largest n: n*step <= x."""
    d, m = divmod(x, step)
    if _closeto(m / step, 1, abs(offset), step):
        return d + 1
    return d


def _ge(x, offset, step):
    """Return the smallest n: n*step >= x."""
    d, m = divmod(x, step)
    if _closeto(m / step, 0, abs(offset), step):
        return d
    return d + 1


def _staircase(steps, actual):
    if len(actual) > 1 and 10 * actual[0] == actual[-1]:
        flights = (0.1 * steps[:-1], steps, 10 * steps[1:])
    else:
        flights = (0.1 * steps, steps, 10 * steps)
    return np.hstack(flights)
