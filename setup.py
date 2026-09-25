from setuptools import setup, find_packages

VERSION = '2.0.0'
DESCRIPTION = 'Python package for TME scoring and sex-aware immunotherapy response prediction'


# Setting up
setup(
        name="TMEImmune", 
        version=VERSION,
        author="Qilu Zhou",
        author_email="<qiluzhou@umass.edu>",
        url="https://github.com/ShahriyariLab/TMEImmune",
        description=DESCRIPTION,
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        packages=find_packages(),
        include_package_data=True,
        package_data={
        "TMEImmune": ["data/*.csv", "data/*.json", "data/*.gmt", "data/*.npz", "data/nb_biomarker/*", "data/Gide/gide_training.npz", "data/isafn/*"], 
        },
        install_requires=["pandas>=1.5.0", "numpy>=1.23.5", "rnanorm",
                          "inmoose", "lifelines", "scikit-learn", "matplotlib", "scipy", "statsmodels", "joblib", "gseapy", "torch"], 
        keywords=['python', 'TME score'],
        classifiers= [
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ],
        python_requires=">=3.10"
)