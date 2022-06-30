from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='pyrankability',
      version='0.1',
      description='Ranking Python Library',
      url='https://github.com/IGARDS/ranking_toolbox',
      author='Paul Anderson, Tim Chartier, Amy Langville, Kathryn Behling',
      author_email='pauleanderson',
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
      packages=['pyrankability'],
      zip_safe=False)
