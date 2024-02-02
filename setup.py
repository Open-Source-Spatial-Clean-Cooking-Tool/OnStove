from setuptools import setup
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='onstove',
    version=get_version('onstove/__init__.py'),
    packages=['onstove'],
    package_data={'onstove': ['static/svg/*.svg', 
                              'tests/test_data/*.*',
                              'tests/test_data/RWA/*.*',
                              'tests/test_data/RWA/*/*/*.*']},
    python_requires='>=3.10',
    install_requires=['dill',
                      'geopandas',
                      'jupyterlab',
                      'matplotlib',
                      'plotnine',
                      'psutil',
                      'psycopg2',
                      'python-decouple',
                      'rasterio',
                      'scikit-image',
                      'svgpathtools',
                      'svgpath2mpl']
)




