import os
import sys

sys.path.insert(
    0, os.path.abspath("../../src")
)  # Asegúrate que Sphinx encuentre tu código fuente

project = "PySAG-UdeC"
copyright = "2025, Sebastian Galindo, Miguel Ángel Moreno"
author = "Sebastian Galindo, Miguel Ángel Moreno"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",  # Incluir documentación desde docstrings
    "sphinx.ext.napoleon",  # Para soportar docstrings estilo Google y NumPy
    "sphinx.ext.intersphinx",  # Para enlazar a documentación de Python, NumPy, etc.
    "sphinx.ext.viewcode",  # Añade enlaces al código fuente
    "myst_parser",  # Para parsear archivos Markdown
    "sphinx.ext.todo",  # Para gestionar TODOs si los usas
]

autodoc_typehints = "description"  #
autodoc_typehints_format = "short"  #
templates_path = ["_templates"]  #
exclude_patterns = []  #

language = "es"  #

html_theme = "sphinx_rtd_theme"  # O "alabaster" si lo prefieres
html_static_path = ["_static"]  #

# Configuración para intersphinx (opcional, pero recomendado)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Configuración para Napoleon (si usas docstrings estilo Google/NumPy)
napoleon_google_docstring = True
napoleon_numpy_docstring = True  # Parece que usas más este estilo
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Para que MyST-Parser pueda encontrar archivos .md fuera de la carpeta source
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Opcional: si quieres que los TODOs aparezcan en la documentación
todo_include_todos = True
