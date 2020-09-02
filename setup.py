import setuptools
import os
import fps_dfs_optimizer

# Get the version of this package
version = fps_dfs_optimizer.version

# Get the long description of this package
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='fps_dfs_optimizer',
    version=version,
    author="First Principles",
    author_email="rcpattison@gmail.com",
    description="Optimizers for Daily Fantasy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/firstprinciples/projects/dfs_optimizer",
    packages=setuptools.find_packages(exclude=['unit_tests']),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'Pillow',
        'scikit-learn',
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
