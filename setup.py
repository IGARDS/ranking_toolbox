from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = open(f"{this_directory}/README.md").read()

setup(name='pyrankability',
      version='0.1.4',
      description='Ranking Python Library',
      url='https://github.com/IGARDS/ranking_toolbox',
      author='Paul Anderson, Tim Chartier, Amy Langville, Kathryn Behling',
      author_email='pauleanderson@gmail.com',
      license='MIT',
      install_requires=[
          'gurobipy',
          'matplotlib',
          'pandas',
          'networkx',
          'altair',
          'pygraphviz',
          'scipy',
          'sklearn',
          'pytest',
          'nx_altair',
          'ipython',
          'tqdm'
      ],
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['pyrankability'],
      zip_safe=False)
