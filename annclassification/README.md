## Installations

- This testing package requires a different conda venv environement, compared to the root venv/
- The root venv/ has `gensim` library, which require `numpy` version 1.26, which is not compatible with `pandas`
- Note: `conda` does not install `scikeras` at the moment.

```bash
  conda install -p venv/ ipykernel tensorflow pandas numpy scikit-learn tensorboard matplotlib streamlit
  pip install scikeras
```

- Or, but not preferable compared to using `conda install`:

```bash
  pip install -r requirements.txt
```
