import os

from setuptools import find_packages, setup

setup(
    name="ipal-iids",
    version="1.5.2",
    packages=find_packages(exclude="tests"),
    scripts=["ipal-iids", "ipal-extend-alarms", "ipal-visualize-model"],
    install_requires=[
        "numpy",
        "tensorflow",
        "graphviz",
        "scikit-learn",
        "keras",
        "torch",
        "matplotlib",
        f"pomegranate @ file://{os.path.dirname(__file__)}/misc/pomegranate-0.14.9.zip",
        "pandas",
        "gurobipy",
        "orjson",
        "zlib-ng",
    ],
    tests_require=["pre-commit", "black", "flake8", "pytest", "pytest-cov", "isort"],
    url="https://github.com/fkie-cad/ipal_ids_framework",
    author="Konrad Wolsing",
    author_email="wolsing@comsys.rwth-aachen.de",
    long_description="Cyber-physical systems are increasingly threatened by sophisticated attackers, also attacking the physical aspect of systems. Supplementing protective measures, industrial intrusion detection systems promise to detect such attacks. However, due to industrial protocol diversity and lack of standard interfaces, great efforts are required to adapt these technologies to a large number of different protocols. To address this issue, we propose IPAL - a common representation of industrial communication as input for industrial intrusion detection systems.  This software (ipal-iids) contains the implementation of several IIDSs based on the message and state format generated by our second project transcriber",
    description="Industrial Intrusion Detection - a framework for protocol-independent industrial intrusion detection on top of IPAL.",
    keywords="IPAL IDS industrial CPS intrusion detection anomaly detection",
    classifiers=[
        "License :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    include_package_data=True,
)
