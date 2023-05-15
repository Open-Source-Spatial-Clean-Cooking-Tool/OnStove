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
copyright = '2023, Camilo Ramirez and Babak Khavari'
author = 'Camilo Ramirez and Babak Khavari'

# The full version, including alpha/beta/rc tags
release = 'v0.1.3'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.linkcode' # to link functions to github source, requires a linkcode_resolve function
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'geopandas': ('https://geopandas.org/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'rasterio': ('https://rasterio.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'skimage': ('https://scikit-image.org/docs/stable/', None),
    'psycopg2': ('https://www.psycopg.org/docs/', None),
    'plotnine': ('https://plotnine.readthedocs.io/en/stable/', None)
}


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
      "image_light": "OnStove_logo_color_margins.svg",
      "image_dark": "OnStove_logo_dark_margins.svg",
   },
   "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Open-Source-Spatial-Clean-Cooking-Tool/OnStove",
            "icon": "fab fa-github",
        }
   ],
   "show_nav_level": 2,
}
html_favicon = "_static/OnStove_favicon.svg"
html_css_files = ["onstove.css"]
html_sidebars = {
   'index': ['search-field.html', 'globaltoc.html'],
   'onstove_model': ['search-field.html', 'globaltoc.html'],
   'quickstart': ['search-field.html', 'globaltoc.html'],
   'resources_and_license': ['search-field.html', 'globaltoc.html'],
   'contributions_and_partners': ['search-field.html', 'globaltoc.html'],
   '**': ['search-field.html', 'sidebar-nav-bs.html']
}
