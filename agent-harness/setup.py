from setuptools import setup, find_namespace_packages

setup(
    name="cli-anything-medai-agent-loop",
    version="0.3.0",
    description="CLI-Anything style medical AI segmentation, ShapeKit post-processing, RadThinking trace, and lightweight agent loop prototype",
    packages=find_namespace_packages(include=["cli_anything", "cli_anything.*"]),
    python_requires=">=3.9",
    install_requires=[
        "click>=8.0.0",
        "pyyaml>=6.0",
        "numpy>=1.22,<3.0",
        "nibabel>=5.0.0",
        "tqdm>=4.60.0",
        "requests>=2.28.0",
        "scipy>=1.9.0",
        "scikit-image>=0.19.0",
        "connected-components-3d>=3.10.0",
    ],
    extras_require={
        "totalseg": ["TotalSegmentator"],
    },
    entry_points={
        "console_scripts": [
            "medai-cli=cli_anything.medai.medai_cli:main",
        ],
    },
    package_data={"cli_anything.medai": ["skills/*.md"]},
    include_package_data=True,
)
