from setuptools import setup, find_packages

setup(
    name="ppfot-ids",
    version="1.0.0",
    description=(
        "Byzantine-Robust Federated Intrusion Detection "
        "via Rényi-Private Optimal Transport"
    ),
    author="Roger Nick Anaedevha, Alexander G. Trofimov, Yuri V. Borodachev",
    author_email="rogerpanel@gmail.com",
    url="https://github.com/rogerpanel/CV/tree/main/PPFOT-IDS",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "POT>=0.9.0",
        "geomloss>=0.1.1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "kagglehub>=0.1.0",
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
