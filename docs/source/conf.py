project = 'napcas'
copyright = '2025, napcas Team'
author = 'napcas Team'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'breathe',
]

breathe_projects = {"napcas": "../build/xml"}
breathe_default_project = "napcas"

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']
