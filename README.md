# Census Merge Other

This project merge census data with other spatial datasets.

> ## Setup

1. Create a **.env** file, insert an fill the following envirommental variables:

    ```` env
        ROOT_DATA= <path to save the data>
    ````

2. Create a  vitual enviroment and install all packages in requiments.txt.

    ```` bash
        conda create --name <env> --file requirements.txt
    ````

3. Install the project as package.

    ```` bash
        pip install -e .
    ````

> ## Usage

1. Configure the parameters in the files:

    ```` bash
    ├── data
        ├── parameters.json
    ````

2. Run src/main.py

    ```` bash
        python src/main.py
    ````

> ## Parameters description

Description of the parameters needed to execute the code.

>>### parameters.json

* **region**: The name of the region (Ex: Brazil)
* **aggregation_level**: Geographical level of data aggregation
* **data_name** The name of the data (Ex: elections)
* **save_filename** filename to save the data
* **type_merge** merging type (normal or spatial)
* **left_id_col** census data identification column
* **right_id_col** other data identification column
* **meshblock_id_col** meshblock data identification column
* **meshblock_crs**: The meshblock coordinate system

## Project Organization

```` text
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── main.py        <- Main function
    │   ├── utils.py       <- Utility functions
    │   ├── classes        <- Scripts regarding classes
    │   │   └── data.py
    │   │   └── merge.py
    │   │
    ├────
````

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
