from setuptools import setup

setup(
    name='onstove',
    version='0.1.3',
    packages=['onstove'],
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




