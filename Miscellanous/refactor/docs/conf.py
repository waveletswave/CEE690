import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Point to the parent folder

extensions = [
    'sphinx.ext.autodoc',  # <--- The magic extension
    'sphinx.ext.napoleon', # <--- Understands NumPy/Google style docstrings
    'sphinx.ext.viewcode', # <--- Adds links to source code
]

html_theme = 'sphinx_rtd_theme'
