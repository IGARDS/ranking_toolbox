from setuptools import setup

setup(name='pyrankability',
      version='0.1',
      description='Ranking Python Library',
      url='https://github.com/IGARDS/ranking_toolbox',
      author='Paul Anderson, Tim Chartier, Amy Langville, Kathryn Behling',
      author_email='pauleanderson',
      license='MIT',
      install_requires=[
          'gurobipy',
          'matplotlib'
      ],
      packages=['pyrankability'],
      zip_safe=False)
