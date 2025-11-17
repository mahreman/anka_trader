"""
Setup script for Otonom Trader.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="otonom-trader",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Autonomous trading system with anomaly detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/otonom-trader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "otonom-trader=otonom_trader.cli:app",
        ],
    },
)
