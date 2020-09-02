dfs_optimizer
==============================

Optimizers for Daily Fantasy

First Principles Project Organization
------------
```
.
├── README.md                           <-- This README
├── fps_dfs_optimizer  <-- package, named `fps_<repo-name>` to avoid PyPI naming conflicts
│   ├── __init__.py                     <-- Specify version number of package. <major>.<minor>.<build>
│   │                                       Increments correspond to <API change>.<Feature addition>.<Improvement>   
│   ├── analyses                        <-- analysis work, jupyter notebooks, etc
│   ├── data                            <-- All data files
│   │   ├── raw
│   │   └── transformed
│   ├── models                          <-- models and their subfolder in here
│   │   ├── deployed                    <-- models in this folder are installed with the package by default
│   └── preprocessing                   <-- any refactor preprocessing files in here
├── requirements.txt                    <-- keep this up to date so that a new user can pip install -r requirements.txt
├── setup.py                            <-- make package installable with `pip install` and define `install_requirements`
├── MANIFEST.in                         <-- grab files that are not `*.py` and install them with the package
└── unit_tests
    └── test_environment.py
```
--------
