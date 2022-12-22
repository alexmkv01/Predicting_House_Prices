# House Prices Regression using Neural Networks #
This repository contains the code for a neural network model for regression, implemented using PyTorch, to predict the price of houses in California using the California House Prices Dataset. This project was completed as part of the Introduction to Machine Learning module at Imperial College London.

## Getting Started ##
To run the code in this repository, you will need to have Python3 installed with the following dependencies:
  - Pandas
  - PyTorch
  - Sklearn
  
You can install these dependencies by running the following command:
``` linuc
pip install pandas pytorch scikit-learn
```

You can then clone this repository and navigate to the local directory:
``` linux
git clone https://github.com/alexmkv01/Predicting_House_Prices.git
cd Predicting_House_Prices
```

## Data ##
The California House Prices Dataset can be found in the "housing.csv" file. It contains ten variables:

  1. longitude: longitude of the block group
  2. latitude: latitude of the block group
  3. housing median age: median age of the individuals living in the block group
  4. total rooms: total number of rooms in the block group
  5. total bedrooms: total number of bedrooms in the block group
  6. population: total population of the block group
  7. households: number of households in the block group 
  8. median income: median income of the households in the block group
  9. ocean proximity: proximity to the ocean of the block group
  10. median house value: median value of the houses of the block group

## Neural Network Architecture ## 
We have implemented a dynamic neural network architecture with parametric values for the number of hidden layers, neurons per layer, learning rate, batch size, and training epochs. The parameters were optimised in order to predict the median house value of a block group most accurately. The Regressor class includes the following methods:

  - **`_preprocessor()`**: Preprocesses the input and output data. This includes handling missing / duplicated data entires, storing parameters used for preprocessing, normalizing the input data, and encoding categorical data as 1-hot vectors.
  - **`__init__()`**: Initializes the model and defines the layers of the neural network.
  - **`forward()`**: Defines the forward pass of the neural network.
  - **`fit()`**: Trains the model on the training data.
  - **`predict()`**: Makes predictions on the test data.
  - **`save_regressor()`** and **`load_regressor()`**: Save and load the model using pickle functions.

## Hyperparameter Tuning ## 
We have performed a hyperparameter search to optimize the performance of our model. The hyperparameters considered include the learning rate, the batch size, the number of hidden layers and training iterations. The hyperparameters were optimised using a combination of manual tuning and combinatorial
search over a pre-defined search space using **`GridSearchCV`** (a technique that exhaustively tries every combination of parameters with cross-validation).

## Evaluation ## 
We have evaluated the performance of our model using root mean squared error (RMSE), r2 score, and mean absolute difference (MAD) metrics on the test data. Further information and justification is outlined in the report.

## Running the Code ## 
Simply run the python file, making sure that the "housing.csv" file is in saved in the same local directory.

## Results ##
The results of our experiments can be found in the report included in this repository. In our implementation, we were able to achieve an RMSE of ≈ 48,000, an r2 score of ≈ 0.82, which is approximately $32,000 absolute difference per prediction.

## Report ##
A report detailing our model, evaluation setup, hyperparameter tuning process, and final results can be found in the "house_prices_report.pdf" file included in this repository. The report length is 5 pages, including figures and tables. This coursework including code and report attained full marks!




