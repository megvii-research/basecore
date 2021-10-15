import re
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


with open("basecore/__init__.py", "r") as file:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        file.read(), re.MULTILINE
    ).group(1)


setuptools.setup(
    name="basecore",
    version=version,
    author="wangfeng02",
    author_email="wangfeng02@megvii.com",
    description="pack of megvii base team common used logic based on megengine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    install_requires=[
        "megfile",
        "colorama",
        "numpy",
        "pyyaml",
        "loguru",
        "easydict",
        "tabulate",
    ],
)
