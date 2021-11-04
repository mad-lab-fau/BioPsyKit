# Contributing to BioPsyKit

This document contains information for developers that want to contribute to ``BioPsyKit``.

It includes further in-depth information on how to set up and use required tools
and learn about programing methods used in the development of this project.


## Contributing Guidelines

### General Information

`BioPsyKit` is under active development and welcomes everyone to contribute to code improvements, documentation, 
testing, and new features! [Why should I contribute?](https://github.com/jonschlinkert/idiomatic-contributing/)

The project is structured into different submodules that each are designed for specific purposes, such as 
``signals``, ``sleep``, ``saliva``, ``questionnaires``, as well as overarching submodules used throughout 
the package, such as ``plotting``, ``colors``, and ``utils``.  

Before adding new features, please think about on where your features might suit best into the project structure 
to that other users can find them easily. 

Additionally, please don't forget to comment and document your code, as well as adding examples and unit 
tests (more on that below) to your code.

### Git Workflow

As multiple people are expected to contribute to `BioPsyKit` at the same time, `BioPsyKit` implements a proper [Git 
workflow](https://guides.github.com/introduction/flow/) to prevent possible issues.

#### Branching structure

In `BioPsyKit` we use a *main + feature branches* structure.
This workflow is well explained [here](https://www.atlassian.com/blog/git/simple-git-workflow-is-simple).
  
All changes to the `main` branch should be performed using feature branches.
Before merging, the feature branches should be rebased onto the current `main` branch.


In summary, follow these steps:

##### Create a Pull Request
Create a new branch from the `main` branch. If you propose a new feature then name your branch something like 
`<feature_feature-name>` where `feature-name` roughly describes the purpose of the new feature. 
If you propose a bugfix, improvement, etc. then name your branch something like `<fix_fix-name>` where `fix-name` 
roughly describes the bugfix or improvement you propose.

Afterwards, create a [Pull Request](https://www.earthdatascience.org/courses/intro-to-earth-data-science/git-github/github-collaboration/how-to-submit-pull-requests-on-github/). 
Be sure to have a `WIP: ` at the very beginning of its name. 
The _source branch_ is the new branch you just created and the _target branch_ is `main`.

In the description, write shortly what you are going to fix, add, or improve, then start working!

##### During Development
During development, remember to adhere to the general recommendations for feature branches. 

As a reminder, feature branches...

- ...should be short-lived
- ...should be dedicated to a *single* feature
- ...should be worked on by a *single* person
- ...must be merged via a Pull Request and not manually
- ...must be reviewed before merging
- ...must pass the pipeline checks before merging
- ...should be rebased onto `main` if possible (remember to only rebase if you are the only person working on this branch!)
- ...should be pushed soon and often to allow everyone to see what you are working on
- ...should be associated with a Pull Request, which is used for discussions and code review.
- ...that are not ready to review, should have a Pull Request prefixed with `WIP: `
- ...should also close issues that they solve, once they are merged

##### Pushing Code

As soon as your new contribution to `BioPsyKit` is ready, it's time to ensure your code passes all necessary checks, 
such as formatting, linting, building documentation, and performing tests. To run these checks more easily this project 
uses [poethepoet](https://github.com/nat-n/poethepoet), a task runner that runs well with poetry and that provides 
a cross-platform CLI for common tasks. More details on that are explained in a 
[later Section](#Tools-used-in-BioPsyKit).


##### Ask to Merge Pull Request
When you think your implementation is done and ready to be merged into the `main` branch, remove the `WIP: ` prefix 
from the Pull Request’s name.  Then, assign a reviewer to the Pull Request. This person will review your code, and, 
if the reviewer agrees with your proposed solution, he or she will merge your changes into `main`. 
Attempting to merge a Pull Request into `main` will trigger [GitHub Actions](https://github.com/features/actions) that 
will again check if your code passes linting and tests and will afterwards build the documentation. 
Upon passing all checks, your changes will be automatically merged into `main`.

**That's it!**


In summary, your Git workflow should look similar to this:
```bash
# Create a new branch
git checkout main
git pull origin main
git checkout -b new-branch-name
git push origin new-branch-name
# Go to GitHub and create a new pull request with WIP prefix

# Do your work. During commit and push your
git push origin new-branch-name

# In case there are important changes in main, rebase
git fetch origin main
git rebase origin/main
# resolve potential conflicts
git push origin new-branch-name --force-with-lease

# Create a pull request and merge via web interface

# Once branch is merged, delete it locally, start a new branch
git checkout main
git branch -D new-branch-name

# Start at top!
```

#### Implementing Large Features

When implementing large features it sometimes makes sense to split it into individual merge requests/sub-features.
If each of these features are useful on their own, they should be merged directly into `main`.
If the large feature requires multiple pull requests to be usable, it might make sense to create a long-lived feature
branch, from which new branches for the sub-features can be created.
It will act as a `develop` branch for just this feature.

**Note**: Remember to rebase this temporary `develop` branch onto `main` from time to time!

#### General Git Tips

- Communicate with your Co-developers
- Commit often
- Commit in logical chunks
- Don't commit temp files
- Write at least somewhat [proper messages](https://chris.beams.io/posts/git-commit/)
   - Use the imperative mood in the subject line
   - Use the body to explain what and why vs. how
   - ...more see [link above](https://www.atlassian.com/blog/git/simple-git-workflow-is-simple)

## Development Tools

### Project Setup and Poetry

`BioPsyKit` only supports Python 3.7 and newer. First, install a compatible version of Python.
If you do not want to modify your system installation of Python you can use [conda](https://docs.conda.io/en/latest/)
or [pyenv](https://github.com/pyenv/pyenv).
However, there are some issues with using conda.
Please, check the [trouble shooting guide](#Troubleshooting) below.

`BioPsyKit` uses [poetry](https://python-poetry.org) to manage its dependencies.
Once you installed poetry, run the following commands to initialize a virtual env and install all development
dependencies:

```bash
poetry install
```
This will create a new folder called `.venv` inside your project dir.
It contains the python interpreter and all site packages.
You can point your IDE to this folder to use this version of Python.
For PyCharm you can find information about this 
[here](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html).

**In case you encounter any issues (with this command or any command below), please check the section on
 [trouble shooting](#trouble-shooting)**.
 
To add new dependencies:

```bash
poetry add <package name>

# Or in case of a dev dependency
poetry add --dev <package name>
```

For more commands see the [official documentation](https://python-poetry.org/docs/cli/).

To update dependencies after the `pyproject.toml` file was changed (It is a good idea to run this after a `git pull`):
```bash
poetry install --no-root

# or (see differences below)
poetry update
```

Running `poetry install` will only install packages that are not yet installed. `poetry update` will also check, if 
newer versions of already installed packages exist.

### Tools used in BioPsyKit

To make it easier to run command-line tasks this project uses [poethepoet](https://github.com/nat-n/poethepoet) to 
provide a cross-platform CLI for common tasks.

Install `poethepoet` into your project (so it works inside `poetry shell`):
```bash
poetry add --dev poethepoet
```
And into your default python environment (so it works outside of `poetry shell`)
```bash
pip install poethepoet
```

All commands need to be executed in the `venv` created by poetry. To list the available tasks, run:

```bash
$ poe
docs                 Build the html docs using Sphinx.
format               Reformat all files using black.
format_check         Check, but not change, formatting using black.
lint                 Lint all files with Prospector.
test                 Run Pytest with coverage.
update_version       Bump the version in pyproject.toml and biopsykit.__init__. Pass the options "major", "minor", or "patch" with the `-v` argument to bump the major, minor, or patch version, respectively.  
```

To run one of the commands execute (e.g. the `test` command):
```bash
poe test
```

To execute `format`, `lint`, and `test` all together, run:
```bash
poe default
```

You should run this as often as possible!
At least once before any `git push`.

To ensure that the whole library uses a consistent **format**, we use [black](https://github.com/psf/black) to
autoformat our code.
Black can also be integrated [into your editor](https://black.readthedocs.io/en/stable/integrations/editors.html), 
if you do not want to run it from the command line.
Because, it is so easy, we also use *black* to format the test-suite.

For everything *black* can not handle, we use [prospector](http://prospector.landscape.io/en/master/) to handle all 
other **linting** tasks. *Prospector* runs `pylint`, `pep257`, and `pyflakes` with custom rules to ensure consistent 
code and docstring style.

For **documentation** we follow the numpy doc-string guidelines and auto-build our API documentation using *Sphinx*.
To make your life easier, you should also set your IDE tools to support the numpy docstring conventions.

----
**Note**: In order to build the documentation, you need to additionally install [pandoc](https://pandoc.org/installing.html).

----


### Testing and Test Data

`BioPsyKit` uses `pytest` for **testing**. Besides using the `poe`-command, you can also use an IDE integration
available for most IDEs.

While all automated tests should go in the `tests` folder, it might be helpful to create some external test script 
from time to time.

For this you can simply install the package locally (using `poetry install`) and even get a Jupyter kernel with all
dependencies installed (see [IDE Config](#Configure-your-IDE)).
Test data is available under `example_data` and you can import it directly using the `get_...` helper functions:

```python
from biopsykit.example_data import get_sleep_imu_example

data, fs = get_sleep_imu_example()
```
 
### Configure your IDE

#### PyCharm

**Test runner**: Set the default testrunner to `pytest`. 

**Black**: Refer to this [guide](https://black.readthedocs.io/en/stable/editor_integration.html) 

**Autoreload for the Python console**:

You can instruct Pycharm to automatically reload modules upon changing by adding the following lines to
"Settings -> Build, Excecution, Deployment -> Console -> Python Console" in the Starting Script:

```python
%load_ext autoreload
%autoreload 2
```


#### Jupyter Lab/Notebooks

To set up a Jupyter environment that has ``BioPsyKit`` and all dependencies installed, run the following commands:

```bash
# poetry install including root!
poetry install
poetry run poe register_ipykernel
``` 

After this you can start Jupyter as always, but select "biopsykit" as a kernel when you want to run a notebook.

Remember to use the autoreload extension to make sure that Jupyter reloads BioPsyKit, when ever you change something in 
the library.
Put this in your first cell of every Jupyter Notebook to activate it:

```python
%load_ext autoreload  # Load the extension
%autoreload 2  # Autoreload all modules
```

### Release Model

BioPsyKit follows typically semantic visioning: A.B.C (e.g. 1.3.5)

- `A` is the major version, which will be updated once there were fundamental changes to the project
- `B` is the minor version, which will be updated whenever new features are added
- `C` is the patch version, which will be updated for bugfixes

As long as no new minor or major version is released, all changes should be interface compatible.
This means that the user can update to a new patch version without changing any user code!

This means at any given time we need to support and work with two versions:
The last minor release, which will get further patch releases until its end of life.
The upcoming minor release for which new features are developed at the moment.

Note that we will not support old minor releases after the release of the next minor release to keep things simple.
We expect users to update to the new minor release, if they want to get new features and bugfixes.

There is no fixed timeline for a release, but rather a list of features we will plan to include in every release.
Releases can often happen and even with small added features.

### Troubleshooting

#### `poetry not found` when using `zsh` as shell

If you have trouble installing `poetry` while using `zsh` as your shell, check this [issue](https://github.com/python-poetry/poetry/issues/507)

#### Installation issues while using `conda`

Setting up `poetry` with `conda` as the main Python version can be a little tricky.
First, make sure that you installed poetry in the [recommended way](https://python-poetry.org/docs/#installation) using 
the PowerShell command.

Then you have 2 options to start using poetry for this package:

1. Using a `conda env` instead of `venv`
    - Create a new conda env (using the required Python version for this project).
    - Activate the environment.
    - Run `poetry install --no-root`. Poetry will 
    [detect that you are already using a conda env](https://github.com/python-poetry/poetry/pull/1432) and will use it, 
    instead of creating a new one.
    - After running the poetry install command you should be able to use poetry without activating the conda env again.
    - Setup your IDE to use the conda env you created
2. Using `conda` python and a `venv`
    - This only works, if your conda **base** env has a Python version supported by the project (>= 3.7)
    - Activate the base env
    - Run `poetry install --no-root`. Poetry will create a new venv in the folder `.venv`, because it detects and
        handles the conda base env 
        [different than other envs](https://github.com/maksbotan/poetry/blob/b1058fc2304ea3e2377af357264abd0e1a791a6a/poetry/utils/env.py#L295).
    - Everything else should work like you are not using conda
    
#### Warning/Error about outdated/missing dependencies in the lock file when running `install` or `update`

This happens when the `pyproject.toml` file was changed either by a git update or by manual editing.
To resolve this issue, run the following and then rerun the command you wanted to run:

```bash
poetry update --lock
``` 

This will synchronise the lock file with the packages listed in `pyproject.toml` 
