# Learning beyond the spatial dependence structure: a regularized graph-based spatial cross-validation approach to fairly assess census data-driven election models

Spatial-contextualized modeling of voting behavior is an essential tool for understanding the electorate and the factors that shape its decision-making, including polarization and socioeconomic conjunctures. Machine learning models have outperformed classic approaches within this topic, especially in identifying patterns and relationships from high-dimensional census data. However, most reference studies do not account for the spatial dependence of the data when validating the models. Cross-validation, a widespread resampling method, limits the exploratory nature of the modeling by biasing it toward the already known spatial dependence structure. We propose RGraphSCV, a Regularized Graph-based Spatial Cross-Validation approach where spatial folds mirror pre-existing geographic boundaries and spatial dependence may occur non-contiguously across space. RGraphSCV uses a bipartite graph structure to determine a removing buffer region that isolates the test from the training set, formalizes the problem as a one-class transductive classification task, and introduces a novel label propagation method that integrates the semivariogram technique to classify nodes from the training set as part of the removing buffer. We evaluate RGraphSCV using three study cases related to recent presidential and congressional elections from Brazil, Australia, and the United States of America. Our experiments demonstrate that RGraphSCV yields less biased and more realistic results in the presence of spatial dependence, making it suitable for assessing machine learning models to identify new patterns and relationships beyond the spatial dependence structure of the data.

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


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
