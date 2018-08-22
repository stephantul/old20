# -*- coding: utf-8 -*-
"""Setup file."""
from setuptools import setup
from setuptools import find_packages


setup(name='old20',
      version='2.0.0',
      description='Calculate the old20 distance to a lexicon.',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/stephantul/old20',
      license='MIT',
      packages=find_packages(exclude=['examples']),
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3'],
      zip_safe=True)
