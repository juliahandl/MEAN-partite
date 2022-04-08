from setuptools import setup, find_packages

setup(name='moo',
    version='0.1.1',
    description='multi objective optimisation for bipartite graphs',
    url='https://github.com/UoMResearchIT/mo-community-detection-bipartite',
    author='Julia Handl/Hichem Barki',
    author_email='todo',
    license='todo',
    packages=['moo'],
    install_requires=['numpy',
    'python-igraph',
    'sklearn',
    'pymoo',
    'pandas'],
    zip_safe=False)
