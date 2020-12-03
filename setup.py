from setuptools import find_packages, setup
import versioneer

min_python_version = "3.6"
max_python_version = "3.9"

min_numba_version = "0.52.0"
install_requires = [
    "numba >={}".format(min_numba_version),
]

packages = find_packages(include=["numba_extras", "numba_extras.*"])
metadata = dict(
    name="numba-extras",
    description="Extra features for Numba",
    version=versioneer.get_version(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Compilers",
    ],
    author="Anaconda, Inc.",
    url="https://github.com/numba/numba-extras",
    packages=packages,
    install_requires=install_requires,
    python_requires=">={},<{}".format(min_python_version, max_python_version),
    license="BSD",
)

with open("README.md") as f:
    metadata["long_description"] = f.read()
    metadata["long_description_content_type"] = "text/markdown"

setup(**metadata)
