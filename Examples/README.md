This is how to link the python files in a given directory to their notebook representations



```sh

cd Tutorial
jupytext --set-formats "../../Notebooks/Tutorial//ipynb,py:light" *.py
```

```sh

cd WorkedExamples
jupytext --set-formats "../../Notebooks/WorkedExamples//ipynb,py:light" *.py
```

```sh

cd IdealisedExamples
jupytext --set-formats "../../Notebooks/IdealisedExamples//ipynb,py:light" *.py
```

```sh

cd LandscapeEvolution
jupytext --set-formats "../../Notebooks/LandscapeEvolution//ipynb,py:light" *.py
```

```python
!ls ../Notebooks/
```

```python
pwd
```

```python

```
