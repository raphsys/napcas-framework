# docs/conf.py

# -- Extensions ----------------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'nbsphinx',           # <-- pour importer les notebooks
    'myst_nb',            # <-- alternative MyST-NB (Markdown+notebook)
]

# -- nbsphinx options ----------------------------------------------------------
nbsphinx_execute = 'always'      # exécute les cellules à la génération
nbsphinx_kernel_name = 'napcas-kernel'

# -- MyST-NB options (si utilisé) ----------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

