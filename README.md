# How to use the ADORE dataset for modeling

[ADORE](https://renkulab.io/projects/mltox/adore) is a benchmark dataset for modeling acute mortality in fish, crustaceans, and algae. This repository shows how to model the challenges (i.e., data subsets) of ADORE.

## A. Project description

The ADORE dataset was created as the field of ecotoxicology lacks a common dataset to drive ML research. 

This repository contains the benchmark datasets, imported from the [data repository](https://renkulab.io/projects/mltox/adore), and scripts to view the data and to model the 't-F2F' challenge with random forests. 



## B. Getting started

1. Clone the repository.

This step needs an ssh connection.

```
git clone git@gitlab.renkulab.io:mltox/adore-modeling.git
cd adore-modeling/
```

2. Install git LFS.

The data files are stored as Large File Storage (LFS) files. This step can take a few minutes.

```
git lfs install --local
git lfs pull -I "data/processed/*"
git lfs pull -I "data/chemicals/*"
git lfs pull -I "data/taxonomy/*"
```

3. Create a conda environment.

This command installs the environment directly in the project folder using the provided `environment.yml`.

```
conda env create --prefix ./conda-env --file ./environment.yml
conda activate ./conda-env
```

OR

If you prefer mamba:

```
mamba env create --prefix ./conda-env --file ./environment.yml
conda activate ./conda-env
```


4. View the dataset.

Open your favourite IDE and run the `10_view-a-dataset.py` script.

OR 

Run the script directly in the conda environment:

```
python 10_view-a-dataset.py
```



## C. Example/Usage

Models are trained and evaluated using several scripts. Here, we show this for random forest model.

1. training (cross-validation): `14_analysis_regression_rf.py`
2. evaluate training: `24_evaluate_regression_cv_rf.py`
3. testing: `34_analysis_regression_test_rf.py`
4. evaluate testing: `44_evaluate_regression_rf.py`

For other models and more detailed evaluations, check the modeling repository [mltox-model](https://renkulab.io/projects/mltox/mltox-model).
