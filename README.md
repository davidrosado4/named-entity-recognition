# Named Entity Recognition

The objective of this project is to fully understand the structured perceptron algorithm applied to Named Entity
Recognition (NER). NER problems are very useful in many contexts, from information retrieval to question answering
systems. The goal of this project is not to achieve the best results, but to fully understand all the details about a
simple solution.

## Collaborators

The solution was developed by the following collaborators:

- [@sergibech](https://github.com/sergibech)
- [@alexpv01](https://github.com/alexpv01)
- [@davidrosado4](https://github.com/davidrosado4)
- [@Goodjorx](https://github.com/Goodjorx)
- [@sarabase](https://github.com/sarabase)

## Repository Structure

The repository is organized as follows:

- `train models.ipynb`: a Juppyter notebook containing all the code required to train the models and store them in
  fitted models.
- `reproduce_results.ipynb`: a Jupyter a notebook that: loads the data, loads the fitted models from disk and evaluates
  the models.
- `utils.py`: a Python module that contains all the helper functions for the two previous notebooks.

## Reproducing the Results

To reproduce the results obtained by our models, follow these steps:

1. Clone this repository:

```
git clone https://github.com/sarabase/named-entity-recognition.git
```

2. Create a python environment and install the necessary requirements. Activate the environment.

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

3. If you want to see how the models are trained, open the train_models.ipynb notebook in Jupyter and execute the cells.
   This step is optional, since the fitted models are already stored in the repository.


4. If you want to see how the models are evaluated, open the reproduce_results.ipynb notebook in Jupyter and execute the
   cells. This step is optional, since the results are already stored in the repository.


