# This file is execfile()d with the current directory set to its containing dir.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import inspect
import os
import shutil
import sys

# from sphinx.ext.autosummary import Autosummary
# from sphinx.ext.autosummary import get_documenter
# from docutils.parsers.rst import directives
# from sphinx.util.inspect import safe_getattr

# -- Path setup --------------------------------------------------------------

__location__ = os.path.join(os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe())))

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.join(__location__, "../src/biopsykit"))

# -- Run sphinx-apidoc -------------------------------------------------------
# This hack is necessary since RTD does not issue `sphinx-apidoc` before running
# `sphinx-build -b html . _build/html`. See Issue:
# https://github.com/rtfd/readthedocs.org/issues/1139
# DON'T FORGET: Check the box "Install your project inside a virtualenv using
# setup.py install" in the RTD Advanced Settings.
# Additionally it helps us to avoid running apidoc manually

try:  # for Sphinx >= 1.7
    from sphinx.ext import apidoc
except ImportError:
    from sphinx import apidoc

output_dir = os.path.join(__location__, "api")
module_dir = os.path.join(__location__, "../src/biopsykit")
try:
    shutil.rmtree(output_dir)
except FileNotFoundError:
    pass

try:
    import sphinx

    cmd_line_template = "sphinx-apidoc --implicit-namespaces -e -f -M -o {outputdir} {moduledir}"
    # cmd_line_template = "sphinx-apidoc --implicit-namespaces -e -M -o {outputdir} {moduledir}"
    cmd_line = cmd_line_template.format(outputdir=output_dir, moduledir=module_dir)

    args = cmd_line.split(" ")
    if tuple(sphinx.__version__.split(".")) >= ("1", "7"):
        # This is a rudimentary parse_version to avoid external dependencies
        args = args[1:]

    apidoc.main(args)
except Exception as e:
    print("Running `sphinx-apidoc` failed!\n{}".format(e))

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "button",
    "nbsphinx",
    # "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = True

napoleon_numpy_docstring = True

# This value selects if automatically documented members are sorted alphabetical (value 'alphabetical'),
# by member type (value 'groupwise') or by source order (value 'bysource'). The default is alphabetical.
autodoc_member_order = "bysource"

# This value controls how to represent typehints
autodoc_typehints = "description"

# This value selects what content will be inserted into the main body of an autoclass directive.
autoclass_content = "init"


# Taken from sklearn config
# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/" "tex-chtml.js"


# Configure AutoStructify
# https://recommonmark.readthedocs.io/en/latest/auto_structify.html
def setup(app):
    from recommonmark.transform import AutoStructify

    params = {
        "enable_auto_toc_tree": True,
        "auto_toc_tree_section": "Contents",
        "auto_toc_maxdepth": 2,
        "enable_eval_rst": True,
        "enable_math": True,
        "enable_inline_math": True,
    }
    app.add_config_value("recommonmark_config", params, True)
    app.add_transform(AutoStructify)
    # app.add_directive("autoautosummary", AutoAutoSummary)


# class AutoAutoSummary(Autosummary):
#     option_spec = {"methods": directives.unchanged, "attributes": directives.unchanged}
#
#     required_arguments = 1
#
#     @staticmethod
#     def get_members(obj, typ, include_public=None):
#         if not include_public:
#             include_public = []
#         items = []
#         for name in dir(obj):
#             try:
#                 documenter = get_documenter(safe_getattr(obj, name), obj)
#             except AttributeError:
#                 continue
#             if documenter.objtype == typ:
#                 items.append(name)
#         public = [x for x in items if x in include_public or not x.startswith("_")]
#         return public, items
#
#     def run(self):
#         clazz = str(self.arguments[0])
#         try:
#             (module_name, class_name) = clazz.rsplit(".", 1)
#             m = __import__(module_name, globals(), locals(), [class_name])
#             c = getattr(m, class_name)
#             if "methods" in self.options:
#                 _, methods = self.get_members(c, "method", ["__init__"])
#
#                 self.content = ["~%s.%s" % (clazz, method) for method in methods if not method.startswith("_")]
#             if "attributes" in self.options:
#                 _, attribs = self.get_members(c, "attribute")
#                 self.content = ["~%s.%s" % (clazz, attrib) for attrib in attribs if not attrib.startswith("_")]
#         finally:
#             return super(AutoAutoSummary, self).run()


# Enable markdown
extensions.append("recommonmark")

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "BioPsyKit"
copyright = (
    "2020 - {}, Machine Learning and Data Analytics (MaD) Lab, Friedrich-Alexander-Universität "
    "Erlangen-Nürnberg (FAU)".format(datetime.now().year)
)

# -- Copy README, Changelog, etc. --------------------------------------
HERE = Path(__file__).parent
EXAMPLE_NOTEBOOKS_DIR = HERE.joinpath("examples/_notebooks")

shutil.copy(HERE.parent.joinpath("README.md"), HERE.joinpath("README.md"))
shutil.copy(HERE.parent.joinpath("CHANGELOG.md"), HERE.joinpath("CHANGELOG.md"))
shutil.copy(HERE.parent.joinpath("AUTHORS.md"), HERE.joinpath("source/about/AUTHORS.md"))
shutil.copy(HERE.parent.joinpath("CONTRIBUTING.md"), HERE.joinpath("source/contributing/CONTRIBUTING.md"))


def all_but_ipynb(dir, contents):
    result = []
    for c in contents:
        if os.path.isfile(os.path.join(dir, c)) and (not c.endswith(".ipynb")):
            result += [c]
    return result


shutil.rmtree(EXAMPLE_NOTEBOOKS_DIR, ignore_errors=True)
shutil.copytree(HERE.parent.joinpath("examples"), EXAMPLE_NOTEBOOKS_DIR, ignore=all_but_ipynb)
for file in EXAMPLE_NOTEBOOKS_DIR.glob("*.ipynb"):
    with file.open() as f:
        out = f.read()
    out = out.replace("%matplotlib widget", "%matplotlib inline")
    with file.open("w+") as f:
        f.write(out)


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]

# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {"sidebar_width": "300px", "page_width": "1200px", "show_toc_level": 3}
html_theme_options = {"show_toc_level": 3}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
from pkg_resources import get_distribution

release = get_distribution("biopsykit").version
# for example take major/minor
version = ".".join(release.split(".")[:3])

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = ""

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "buton.css",
]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "BioPsyKit-doc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ("letterpaper" or "a4paper").
    # "papersize": "letterpaper",
    # The font size ("10pt", "11pt" or "12pt").
    # "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    # "preamble": "",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [("index", "user_guide.tex", "BioPsyKit Documentation", "Robert Richer", "manual")]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = ""

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
nbsphinx_execute = "auto"

nbsphinx_kernel_name = "python3"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg'}",
]


nbsphinx_epilog = r"""

{% set docname = env.doc2path(env.docname) %}

.. button::
   :text: Download Notebook
   :link: {{ docname }}

"""

# -- External mapping --------------------------------------------------------
python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pyscaffold": ("https://pyscaffold.org/en/stable", None),
    "neurokit2": ("https://neurokit2.readthedocs.io/en/latest", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "nilspodlib": ("https://nilspodlib.readthedocs.io/en/latest/", None),
    "pingouin": ("https://pingouin-stats.org/", None),
}
