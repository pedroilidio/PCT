from setuptools import setup
from pathlib import Path

PATH_HERE = Path(__file__).parent
README = (PATH_HERE/"README.md").read_text()

setup(name='pbct',
      version='0.3.1',
      description='Predictive Bi-Clustering Trees in Python.',
      long_description=README,
      long_description_content_type="text/markdown",
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
