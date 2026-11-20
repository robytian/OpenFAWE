# The OpenFAWE documentation site

Sphinx sources of https://openfawe.readthedocs.io (Read the Docs builds them with `.readthedocs.yaml` at the
repository's root). The user guide's pages are cut from `docs/user-guide.md` and `docs/user-guide.zh.md` at every
build (`conf.py`); edit those two files, not `guide/` or `zh/`.

Build locally:

```bash
pip install -r docs/site/requirements.txt
sphinx-build -b html docs/site docs/site/_build/html
```
