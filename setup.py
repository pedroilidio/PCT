from setuptools import setup

setup(name='PBCT',
      version='0.1',
      description='Predictive Bi-Clustering Trees',
      url='http://github.com/pedroilidio/PCT',
      author='Pedro Ilídio',
      author_email='ilidio@alumni.usp.br',
      license='GPLv3',
      packages=['PBCT'],
      scripts=['bin/PBCT']
      zip_safe=False,
      install_requires=[
          'pandas', 'numpy', 'numba', 'tqdm',
      ]
      extras_require={
          'Tree visualization': ['graphviz', 'matplotlib'],
      }
)
