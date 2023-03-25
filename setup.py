import setuptools


with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.readlines()


setuptools.setup(
    name="weighted_config",
    version="0.1.0",
    author="Ruairi OSullivan",
    author_email="ruairi.osullivan.work@gmail.com",
    description="Configuration Model for Weighted Graphs",
    long_description="Configuration Model for Weighted Graphs. Extends NetworkX",
    url="https://github.com/Ruairi-osul/weighted-config",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
