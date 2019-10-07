from matplotlib import ticker


class SciFuncFormatter(ticker.Formatter):
    # To put full sci notation into each axis label rather than split offsetText

    sFormatter = ticker.ScalarFormatter(useOffset=False, useMathText=True)

    def __call__(self, x, pos=None):
        return "${}$".format(SciFuncFormatter.sFormatter._formatSciNotation('%.10e' % x))

    def format_data(self, value):
        # e.g. for the navigation toolbar, no latex
        return '%-8g' % value


class BoundedMaxNLocator(ticker.MaxNLocator):
    # Tick locator class that only returns ticks within bounds, and if pruned, pruned not to overlap ends of axes
    # Also tries to correct default (simple x3 heuristic) for long x tick labels
    # check_long_tick_labels should be set if x axis and no tick rotation (or y and axis-aligned rotation)

    def __init__(self, nbins='auto', prune='both', check_long_tick_labels=True, **kwargs):
        self.bounded_prune = prune
        self.check_long_tick_labels = check_long_tick_labels
        super(BoundedMaxNLocator, self).__init__(nbins=nbins, **kwargs)

    def tick_values(self, vmin, vmax):
        # Max N locator will produce locations outside vmin, vmax, so even if pruned
        # there can be points very close to the actual bounds. Let's cut them out.

        locs = super(BoundedMaxNLocator, self).tick_values(vmin, vmax)
        locs = [x for x in locs if vmin <= x <= vmax]

        axes = self.axis.axes
        ends = axes.transAxes.transform([[0, 0], [1, 0]])
        length = ((ends[1][0] - ends[0][0]) / axes.figure.dpi) * 72
        tick = self.axis._get_tick(True)
        size_ratio = tick.label1.get_size() / length

        if self.check_long_tick_labels and \
                isinstance(self.axis.major.formatter, ticker.ScalarFormatter) and len(locs) > 1:

            font_aspect = 0.65
            char_frac = font_aspect * size_ratio

            formatter = self.axis.major.formatter

            def _get_Label_len():
                formatter.set_locs(locs)
                # get non-latex version of label
                form = formatter.format
                i = form.index('%')
                i2 = form.index('f', i)
                label = form[i:i2 + 1] % locs[0]
                char_len = len(label)
                if '.' in label:
                    char_len -= 0.4
                return char_frac * char_len * (vmax - vmin)

            label_len = _get_Label_len()
            if self.check_long_tick_labels and locs[1] - locs[0] < label_len * 1.1:
                # check for long labels not accounted for the the current "*3" aspect ratio heuristic for labels
                _nbins = self._nbins
                try:
                    self._nbins = max(1, int((len(locs) - 1) * (locs[1] - locs[0]) / label_len / 1.4)) + 1
                    locs = super(BoundedMaxNLocator, self).tick_values(vmin, vmax)
                    locs = [x for x in locs if vmin <= x <= vmax]
                    label_len = _get_Label_len()
                finally:
                    self._nbins = _nbins
        else:
            _get_Label_len = None
            if self.check_long_tick_labels:
                label_len = (locs[-1] - locs[0]) / (max(len(locs), 2) - 1)
            else:
                label_len = size_ratio * (vmax - vmin) * 1.5

        def prune(_locs):
            if len(_locs) > 1 and self.bounded_prune:
                if self.bounded_prune in ['both', 'lower'] and _locs[0] - vmin < label_len * 0.5:
                    _locs = _locs[1:]
                if self.bounded_prune in ['both', 'upper'] and vmax - _locs[-1] < label_len * 0.5 and len(_locs) > 1:
                    _locs = _locs[:-1]
            return _locs

        locs = prune(locs)
        if len(locs) == 1 and _get_Label_len and label_len < (vmax - vmin) / 3.0:
            _nbins = self._nbins
            try:
                self._nbins = 2
                locs = super(BoundedMaxNLocator, self).tick_values(vmin + label_len / 2, vmax - label_len / 2)
                locs = [x for x in locs if vmin <= x <= vmax]
                label_len = _get_Label_len()
                locs = prune(locs)
            finally:
                self._nbins = _nbins

        return locs
