from setuptools import setup, find_packages

setup(
    name="solar_optimization",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
)