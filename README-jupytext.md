<!-- #region -->
# Python scripts for Quagmire

The scripts in this directory are in the "light" format with structured comments that allows them to be synchronised with notebooks. This markdown file also has a pairing with a notebook. The python scripts (and markdown files) are considered to be the master copy and the notebooks are autogenerated for the purposes of visualisation on github and backward compatibility for anyone not using [jupytext](https://github.com/mwouts/jupytext/).


## Jupytext pairings

This is how to link (or re-link) the python files in a given directory to their notebook representations
This should be harmless on existing files and so can be used to update when a new .py is added.

If you are running in a jupytext ennvironment, then this file can be opened as a notebook and run.

<!-- #endregion -->

```sh

jupytext --set-formats "Notebooks/Tutorial//ipynb,Examples/Tutorial//py:light" Examples/Tutorial/*.py
jupytext --set-formats "Notebooks/WorkedExamples//ipynb,Examples/WorkedExamples//py:light" Examples/WorkedExamples*.py
jupytext --set-formats "Notebooks/IdealisedExamples//ipynb,Examples/IdealisedExamples//py:light" Examples/IdealisedExamples*.py
jupytext --set-formats "Notebooks/LandscapeEvolution//ipynb,Examples/LandscapeEvolution//py:light" Examples/LandscapeEvolution*.py
```

```python

```
