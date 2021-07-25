from setuptools import find_packages, setup

# read README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sssgraph',
    version='0.1.0',
    author="Dominik Lux",
    description="sssgraph, Tackling the SubSetSum-Problem with graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Luxxii/sssgraph",
    project_urls={
        "Bugs": "https://github.com/Luxxii/sssgraph/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: MIT License',
    ],
    license="MIT",
    python_requires=">=3.6",
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'python-igraph']
)
