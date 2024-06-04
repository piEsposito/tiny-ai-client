from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_desc = f.read()

with open(path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    install_requires = f.read()

setup(
    name="tiny-ai-client",
    packages=find_packages(),
    version="0.0.2",
    description="Tiny AI client for LLMs. As simple as it gets.",
    author="Pi Esposito",
    url="https://github.com/piEsposito/tiny_ai_client",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
    ],
)
