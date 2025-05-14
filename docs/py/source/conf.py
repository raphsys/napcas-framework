import os, sys
# ajouter le package napcas installé en editable
sys.path.insert(0, os.path.abspath('../../python'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

extensions += ['breathe']
breathe_projects = {
    "napcas": os.path.abspath("../../docs/cpp")
}
breathe_default_project = "napcas"

