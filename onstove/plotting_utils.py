import os
import matplotlib.offsetbox
from matplotlib.lines import Line2D
from matplotlib.text import Text
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import matplotlib
import matplotlib.pyplot as plt


class ScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """

    def __init__(self, style='single', size=1, extent=0.03, loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None,
                 frameon=True, linekw={}, textprops={}, **kwargs):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        size_units = matplotlib.offsetbox.AuxTransformBox(trans)
        if style == 'single':
            line = Line2D([0, size], [0, 0], **linekw)
            vline1 = Line2D([0, 0], [-extent / 2., extent / 2.], **linekw)
            vline2 = Line2D([size, size], [-extent / 2., extent / 2.], **linekw)
            text = Text(size / 2, extent + 0.01, str(int(size / 1000)) + ' km', horizontalalignment='center',
                        **textprops)
            size_bar.add_artist(line)
            size_bar.add_artist(vline1)
            size_bar.add_artist(vline2)
            size_bar.add_artist(text)
        elif style == 'double':
            line = Line2D([0, size], [0, 0], **linekw)
            vline1 = Line2D([0, 0], [0, extent], **linekw)
            vline2 = Line2D([size / 2, size / 2], [0, extent], **linekw)
            vline3 = Line2D([size, size], [0, extent], **linekw)
            text1 = Text(0, extent + 0.01, 0, horizontalalignment='center', **textprops)
            text2 = Text(size / 2, extent + 0.01, str(int(size / 2000)), horizontalalignment='center', **textprops)
            text3 = Text(size, extent + 0.01, str(int(size / 1000)), horizontalalignment='center', **textprops)
            text4 = Text(0, 0, 'km', horizontalalignment='center', **textprops)
            size_bar.add_artist(line)
            size_bar.add_artist(vline1)
            size_bar.add_artist(vline2)
            size_bar.add_artist(vline3)
            size_bar.add_artist(text1)
            size_bar.add_artist(text2)
            size_bar.add_artist(text3)
            size_units.add_artist(text4)

        self.vpac = matplotlib.offsetbox.HPacker(children=[size_bar, size_units],
                                                 align="center", pad=ppad, sep=sep)
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                                        child=self.vpac, prop=prop, frameon=frameon, **kwargs)


def scale_bar(style='single', size=100000, loc='lower right', frameon=False,
              pad=0, sep=4, linekw=None, ax=None, textprops=None, extent=0.02):
    if linekw is None:
        linekw = dict(color="black")
    if textprops is None:
        textprops = dict(color='black', weight='normal')

    if not ax:
        ax = plt.gca()
    scalebar = ScaleBar(style=style, size=size, loc=loc, frameon=frameon,
                        pad=pad, sep=sep, linekw=linekw, ax=ax,
                        textprops=textprops, extent=extent)
    ax.add_artist(scalebar)


def add_svg(path, ax=None, location=(0.95, 0.95), size=30, color='black', linewidth=1):
    if not ax:
        ax = plt.gca()

    svg_path, attributes = svg2paths(path)
    svg_marker = parse_path(attributes[0]['d'])

    svg_marker.vertices -= svg_marker.vertices.mean(axis=0)
    svg_marker = svg_marker.transformed(matplotlib.transforms.Affine2D().rotate_deg(180))
    svg_marker = svg_marker.transformed(matplotlib.transforms.Affine2D().scale(-1, 1))

    extente_x = max(ax.get_xlim()) - min(ax.get_xlim())
    x = min(ax.get_xlim()) + extente_x * location[0]

    extente_y = max(ax.get_ylim()) - min(ax.get_ylim())
    y = min(ax.get_ylim()) + extente_y * location[1]

    ax.plot(x, y, marker=svg_marker, markeredgewidth=linewidth,
            markersize=size, color=color)


def north_arrow(ax=None, location=(0.95, 0.95), size=30, color='black', linewidth=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'static/svg/north-arrow.svg')
    add_svg(file_path, ax=ax, location=location,
            size=size, color=color, linewidth=linewidth)
