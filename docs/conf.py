# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'onstove'
copyright = '2022, Camilo Ramirez and Babak Khavari'
author = 'Camilo Ramirez and Babak Khavari'

# The full version, including alpha/beta/rc tags
release = 'v0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    # 'sphinx.ext.napoleon',
    'numpydoc',
    # 'sphinx.ext.intersphinx',
    # 'sphinx.ext.coverage',
    # 'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    # 'sphinx_autodoc_typehints',
    # 'sphinx.ext.graphviz',
    # 'sphinx.ext.ifconfig',
    # 'matplotlib.sphinxext.plot_directive', maybe
    # 'IPython.sphinxext.ipython_console_highlighting',
    # 'IPython.sphinxext.ipython_directive',
    # 'sphinx.ext.mathjax', maybe
    # 'sphinx_panels', maybe
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_theme_options = {
   "logo": {
      "image_light": "OnStove_logo_color.svg",
      "image_dark": "OnStove_logo_dark.svg",
   },
   "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Open-Source-Spatial-Clean-Cooking-Tool/OnStove",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        }
   ],
   "show_nav_level": 2,
}
html_favicon = "_static/OnStove_favicon.svg"
html_css_files = ["onstove.css"]