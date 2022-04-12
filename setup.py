from setuptools import setup, find_packages

setup(name='moo',
    version='0.1.1',
    description='multi objective optimisation for bipartite graphs',
    url='https://github.com/leospinaf/BipartiteMOEA',
    author='Julia Handl/Hichem Barki',
    author_email='todo',
    license='todo',
    packages=['moo'],
    install_requires=['numpy',
    'python-igraph',
    'sklearn',
    'pymoo',
    'pandas',
    'condor @ git+https://git@github.com/genisott/pycondor.git@389932cfa4d1954aef7d1b725a33a6b2ef018de2#egg=condor',
    'seaborn',
    'tqdm',
    'psutil',
    'sknetwork',
    'cdlib',
    'skbio'
    ],
    zip_safe=False)
