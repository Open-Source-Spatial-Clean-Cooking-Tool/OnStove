import os
import matplotlib.offsetbox
from matplotlib.lines import Line2D
from matplotlib.text import Text
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import matplotlib
import matplotlib.pyplot as plt


class ScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """Object used to create a scale bar based on an axes transform.

    This makes use of the :doc:`matplotlib.offsetbox.AnchoredOffsetbox<matplotlib:api/offsetbox_api>` object to render
    the scale.

    Parameters
    ----------
    style: str, default 'single'
        Graphic style of the scale, available options 'single' or 'double'.
    size: int, default 1
        Length of bar in data units, this will match the units of the axes.
    extent: float, default 0.03
        Height of bar ends in axes fraction.
    loc: str or int, default 'upper left'
        A location code, same as matplotlib's legend, either: ``upper right``, ``upper left``, ``lower left``,
        ``lower right``, ``right``, ``center left``, ``center right``, ``lower center``, ``upper center`` or ``center``.
    ax: matplotlib.axes.Axes, optional
        Object of type :doc:`matplotlib.axes.Axes<matplotlib:api/axes_api>`.
    pad: float, default 0.4
        Padding around the child as fraction of the fontsize.
    borderpad: float, default 0.5
        Padding between the offsetbox frame and the `bbox_to_anchor`.
    sep: float, default 2
        Separation between the scale bar and the units.
    prop: matplotlib.font_manager.FontProperties, optional
        This is only used as a reference for paddings. If not given,
        :doc:`rcParams["legend.fontsize"]<matplotlib:tutorials/introductory/customizing>` (default: 'medium')
        is used.
    frameon: bool, default False
        Whether to draw a frame around the scale bar.
    linekw: dict, optional
        Style properties for the scale bar.
    textprops: dict, optional
        Font properties for the text.
    **kwargs
        All other parameters are passed on to :doc:`OffsetBox<matplotlib:api/offsetbox_api>`.
    """

    def __init__(self, style='single', size=1, extent=0.03, loc='upper left', ax=None,
                 pad=0.4, borderpad=0.5, sep=2, prop=None,
                 frameon=False, linekw={}, textprops={}, **kwargs):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        size_units = matplotlib.offsetbox.AuxTransformBox(trans)
        if style == 'single':
            line = Line2D([0, size], [0, 0], **linekw)
            vline1 = Line2D([0, 0], [-extent / 2., extent / 2.], **linekw)
            vline2 = Line2D([size, size], [-extent / 2., extent / 2.], **linekw)
            text = Text(size / 2, extent + 0.01, str(int(size / 1000)), horizontalalignment='center',
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
            size_bar.add_artist(line)
            size_bar.add_artist(vline1)
            size_bar.add_artist(vline2)
            size_bar.add_artist(vline3)
            size_bar.add_artist(text1)
            size_bar.add_artist(text2)
            size_bar.add_artist(text3)
        text4 = Text(0, 0, 'km', horizontalalignment='center', **textprops)
        size_units.add_artist(text4)

        self.vpac = matplotlib.offsetbox.HPacker(children=[size_bar, size_units],
                                                 align="center", pad=0, sep=sep)
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                                        child=self.vpac, prop=prop, frameon=frameon, **kwargs)


def scale_bar(style='single', size=100000, extent=0.02, loc='lower right', ax=None,
              borderpad=0.5, sep=4, frameon=False, linekw=None, textprops=None):
    """Function to create a :class:`ScaleBar` object and add it to a specified or current axes.

    This function takes as inputs the basic parameters needed to create a :class:`ScaleBar` and adds the object as a
    :doc:`matplotlib.artist.Artist<matplotli:api/artist_api>` to a specified
    :doc:`axes<matplotlib:api/axes_api>` or to the current :doc:`axes<matplotlib:api/axes_api>` in use if not
    specified.

    Parameters
    ----------
    style: str, default 'single'
        Graphic style of the scale, available options 'single' or 'double'.
    size: int, default 1
        Length of bar in data units, this will match the units of the axes.
    extent: float, default 0.03
        Height of bar ends in axes fraction.
    loc: str or int, default 'upper left'
        A location code, same as matplotlib's legend, either: ``upper right``, ``upper left``, ``lower left``,
        ``lower right``, ``right``, ``center left``, ``center right``, ``lower center``, ``upper center`` or ``center``.
    ax: matplotlib.axes.Axes, optional
        Object of type :doc:`matplotlib.axes.Axes<matplotlib:api/axes_api>`.
    borderpad: float, default 0.5
        Padding between the offsetbox frame and the `bbox_to_anchor`.
    sep: float, default 2
        Separation between the scale bar and the units.
    frameon: bool, default False
        Whether to draw a frame around the scale bar.
    linekw: dict, optional
        Style properties for the scale bar.
    textprops: dict, optional
        Font properties for the text.

    See also
    --------
    ScaleBar
    """
    if linekw is None:
        linekw = dict(color="black")
    if textprops is None:
        textprops = dict(color='black', weight='normal')

    if not ax:
        ax = plt.gca()
    scalebar = ScaleBar(style=style, size=size, loc=loc, frameon=frameon,
                        borderpad=borderpad, sep=sep, linekw=linekw, ax=ax,
                        textprops=textprops, extent=extent)
    ax.add_artist(scalebar)


def add_svg(path, ax=None, location=(0.95, 0.95), size=30, color='black', linewidth=1):
    """Function to add an ``svg`` image as an icon to a location on a specified axes.

    It reads in the path to a ``svg`` image, converts the svg to a marker and plots it to a location in a specified
    :doc:`axes<matplotlib:api/axes_api>` or the current axes in used if not defined.

    Parameters
    ----------
    path: str
        Path to a ``svg`` image.
    ax: matplotlib.axes.Axes, optional
        Object of type :doc:`matplotlib.axes.Axes<matplotlib:api/axes_api>`.
    location: tuple of floats, default (0.95, 0.95)
        Location to plot the image in fraction of the x and y axes.
    size: int, default 30
        Size of the image in pixels.
    color: str, default 'black'
        color of the image.
    linewidth: int, default 1
        Width of the borders of the image in pixels.
    """
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
    """Function to plot a north arrow in a map.

    It makes use of the :func:`add_svg` to add a north arrow icon in a specified location of an
    :doc:`axes<matplotlib:api/axes_api>`.

    Parameters
    ----------
    ax: matplotlib.axes.Axes, optional
        Object of type :doc:`matplotlib.axes.Axes<matplotlib:api/axes_api>`.
    location: tuple of floats, default (0.95, 0.95)
        Location to plot the image in fraction of the x and y axes.
    size: int, default 30
        Size of the image in pixels.
    color: str, default 'black'
        color of the image.
    linewidth: int, default 1
        Width of the borders of the image in pixels.

    See also
    --------
    add_svg
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'static/svg/north-arrow.svg')
    add_svg(file_path, ax=ax, location=location,
            size=size, color=color, linewidth=linewidth)
