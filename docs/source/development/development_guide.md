# Development Guide

This document contains information for developers that need further in-depth information on how to setup and use tools
and learn about programing methods used in development of this project.

[comment]: <> (If you are looking for a higher level overview over the guiding ideas and structure of this project, please visit the)
[comment]: <> ([Project Structure document]&#40;project_structure.md&#41;.)

## Project Setup and Poetry

*biopsykit* only supports Python 3.7 and newer.
First, install a compatible version of Python.
If you do not want to modify your system installation of Python you can use [conda](https://docs.conda.io/en/latest/)
or [pyenv](https://github.com/pyenv/pyenv).
However, there are some issues with using conda.
Please, check the [trouble shooting guide](#trouble-shooting) below.

*biopsykit* uses [poetry](https://python-poetry.org) to manage its dependencies.
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

## Tools we are using

To make it easier to run commandline tasks we use [doit](https://pydoit.org/contents.html) to provide a cross-platform 
cli for common tasks.
All commands need to be executed in the `venv` created by poetry.

To list the available tasks, run:

```bash
$ poetry run doit list
docs                 Build the html docs using Sphinx.
format               Reformat all files using black.
format_check         Check, but not change, formatting using black.
lint                 Lint all files with Prospector.
test                 Run Pytest with coverage.
update_version       Bump the version in pyproject.toml and biopsykit.__init__ .
```

To run one of the commands execute (e.g. the `test` command):
```bash
poetry run doit test
```

To execute `format`, `lint`, and `test` all together, run:
```bash
poetry run doit
# or if you want less output
petry run doit -v 0
```

You should run this as often as possible!
At least once before any `git push`.

**Protip**: If you do not want to type `poetry run` all the time, you can also activate the `venv` for your current
terminal session using `poetry shell`.
After this you can just type, for example, `doit test`.

To ensure that the whole library uses a consistent **format**, we use [black](https://github.com/psf/black) to
autoformat our code.
Black can also be integrated [into your editor](https://black.readthedocs.io/en/stable/integrations/editors.html), 
if you do not want to run it from the command line.
Because, it is so easy, we also use *black* to format the test-suite.

For everything *black* can not handle, we use [prospector](http://prospector.landscape.io/en/master/) to handle all 
other **linting** tasks. *Prospector* runs `pylint`, `pep257`, and `pyflakes` with custom rules to ensure consistent 
code and docstring style.

For **documentation** we follow the numpy doc-string guide lines and autobuild our API documentation using *Sphinx*.
To make your live easier, you should also set your IDE tools to support the numpy docstring conventions.


## Testing and Test data

This library uses `pytest` for **testing**. Besides using the doit-command, you can also use an IDE integration
available for most IDEs.

While all automated tests should go in the test folder, it might be helpful to create some external test script form 
time to time.
For this you can simply install the package locally (using `poetry install`) and even get a Jupyter kernel with all
dependencies installed (see [IDE Config](#Configure-your-IDE)).
Test data is available under `example_data` and you can import it directly using the `get_...` helper functions in 
conftest:

```python
from biopsykit.example_data import get_sleep_imu_example

data = get_sleep_imu_example()
```

[comment]: <> (### Regression Tests)

[comment]: <> (To prevent unintentional changes to the data, this project makes use of regression tests.)

[comment]: <> (These tests store the output of a function and compare the output of the same function at a later time to the stored)

[comment]: <> (information.)

[comment]: <> (This helps to ensure that a change did not modify a function unintentionally.)

[comment]: <> (To make this easy, this library contains a small PyTest helper to perform regression tests.)

[comment]: <> (A simple regression test looks like this:)

[comment]: <> (```python)

[comment]: <> (import pandas as pd)

[comment]: <> (def test_regression&#40;snapshot&#41;:)

[comment]: <> (    # Do my tests)

[comment]: <> (    result_dataframe = pd.DataFrame&#40;...&#41;)

[comment]: <> (    snapshot.assert_match&#40;result_dataframe&#41;)

[comment]: <> (```)

[comment]: <> (This test will store `result_dataframe` in a json file if the test is run for the first time.)

[comment]: <> (At a later time, the dataframe is loaded from this file to compare it.)

[comment]: <> (If the new `result_dataframe` is different from the file content the test fails.)

[comment]: <> (In case the test fails, the results need to be manually reviewed.)

[comment]: <> (If the changes were intentionally, the stored data can be updated by either deleting, the old file)

[comment]: <> (and rerunning the test, or by running ` pytest --snapshot-update`. Be careful, this will update all snapshots.)

[comment]: <> (The results of a snapshot test should be committed to the repo.)

[comment]: <> (Make reasonable decisions when it comes to the datasize of this data.)

[comment]: <> (For more information see `tests/_regression_utils.py` or)

[comment]: <> (`tests.test_stride_segmentation.test_barth_dtw.TestRegressionOnRealData.test_real_data_both_feed_regression` for an)

[comment]: <> ( example.)
 
## Configure your IDE


#### Pycharm

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

To set up a Jupyter environment that has biopsykit and all dependencies installed, run the following commands:

```
# poetry install including root!
poetry install
poetry run doit register_ipykernel
``` 

After this you can start Jupyter as always, but select "biopsykit" as a kernel when you want to run a notebook.

Remember to use the autoreload extension to make sure that Jupyter reloads biopsykit, when ever you change something in 
the library.
Put this in your first cell of every Jupyter Notebook to activate it:

```python
%load_ext autoreload  # Load the extension
%autoreload 2  # Autoreload all modules
```

## Release Model

BioPsyKit follows typically semantic visioning: A.B.C (e.g. 1.3.5)

- `A` is the major version, which will be updated once there were fundamental changes to the project
- `B` is the minor version, which will be updated whenever new features are added
- `C` is the patch version, which will be updated for bugfixes

As long as no new minor or major version is released, all changes should be interface compatible.
This means that the user can update to a new patch version without changing any user code!

This means at any given time we need to support and work with two versions:
The last minor release, which will get further patch releases until its end of life.
The upcoming minor release for which new features are developed at the moment.
However, in most cases we will also not create proper patch releases, but expect users to update to the newest git 
version, unless it was an important and major bug that got fixed.

Note that we will not support old minor releases after the release of the next minor release to keep things simple.
We expect users to update to the new minor release, if they want to get new features and bugfixes.

[comment]: <> (To make such an update model go smoothly for all users, we keep an active changelog, that should be modified a feature)

[comment]: <> (is merged, or a bug fixed.)

[comment]: <> (In particular changes that require updates to feature code should be prominently highlighted in the "Migration Guide")

[comment]: <> (section.)

There is no fixed timeline for a release, but rather a list of features we will plan to include in every release.
Releases can often happen and even with small added features.


## Git Workflow

As multiple people are expected to work on the project at the same time, we need a proper git workflow to prevent issues.

### Branching structure

This project uses a main + feature branches.
This workflow is well explained [here](https://www.atlassian.com/blog/git/simple-git-workflow-is-simple).
  
All changes to the `main` branch should be performed using feature branches.
Before merging, the feature branches should be rebased onto the current `main` branch.

Remember, feature branches...:

- should be short-lived
- should be dedicated to a single feature
- should be worked on by a single person
- must be merged via a Pull Request and not manually
- must be reviewed before merging
- must pass the pipeline checks before merging
- should be rebased onto `main` if possible (remember to only rebase if you are the only person working on this branch!)
- should be pushed soon and often to allow everyone to see what you are working on
- should be associated with a Pull Request, which is used for discussions and code review.
- that are not ready to review, should have a Pull Request prefixed with `WIP: `
- should also close issues that they solve, once they are merged

Workflow:
```bash
# Create a new branch
git checkout main
git pull origin main
git checkout -b new-branch-name
git push origin new-branch-name
# Go to GitHub and create a new pull request with WIP prefix

# Do your work
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

### For large features

When implementing large features it sometimes makes sense to split it into individual merge requests/sub-features.
If each of these features are useful on their own, they should be merged directly into `main`.
If the large feature requires multiple pull requests to be usable, it might make sense to create a long-lived feature
branch, from which new branches for the sub-features can be created.
It will act as a develop branch for just this feature.

**Note**: Remember to rebase this temporary dev branch onto main from time to time!

### General Git Tips

- Communicate with your Co-developers
- Commit often
- Commit in logical chunks
- Don't commit temp files
- Write at least somewhat [proper messages](https://chris.beams.io/posts/git-commit/)
   - Use the imperative mood in the subject line
   - Use the body to explain what and why vs. how
   - ...more see link above

## Trouble Shooting

##### `poetry not found` when using `zsh` as shell

If you have trouble installing `poetry` while using `zsh` as your shell, check this [issue](https://github.com/python-poetry/poetry/issues/507)

##### Installation issues while using `conda`

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
    
##### Warning/Error about outdated/missing dependencies in the lock file when running `install` or `update`

This happens when the `pyproject.toml` file was changed either by a git update or by manual editing.
To resolve this issue, run the following and then rerun the command you wanted to run:

```bash
poetry update --lock
``` 

This will synchronise the lock file with the packages listed in `pyproject.toml` 
