from setuptools import setup

setup(name='pbct',
      version='0.1',
      description='Predictive Bi-Clustering Trees',
      url='http://github.com/pedroilidio/PCT',
      author='Pedro Ilidio',
      author_email='pedrilidio@gmail.com',
      license='GPLv3',
      packages=['PBCT'],
      scripts=['bin/PBCT'],
      zip_safe=False,
      install_requires=[
          'pandas', 'numpy', 'numba', 'tqdm',
      ],
      extras_require={
          'Tree visualization': ['graphviz', 'matplotlib'],
      },
)
