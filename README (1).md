# Linear Regression: Food Truck Profit Prediction

This project implements **univariate linear regression from scratch** using Python to predict the profit of a food truck based on the population of a city.

The notebook follows the linear regression exercise from Andrew Ng's Machine Learning course on Coursera.

## Project Overview

The project uses a dataset containing city population and corresponding food truck profit values. It fits a straight-line model to the training data using **gradient descent**, then visualizes the predicted relationship between population and profit.

## Objectives

- Load and explore the population and profit dataset.
- Implement the linear regression cost function.
- Prepare the feature matrix and target variable.
- Implement gradient descent to learn the model parameters.
- Plot the regression line against the training data.
- Visualize how the cost changes over the training iterations.

## Technologies and Libraries

- Python
- Jupyter Notebook
- NumPy
- pandas
- Matplotlib

## Project Structure

```text
project-folder/
├── Linear_Regression.ipynb
├── README.md
└── data/
    └── ex1data1.txt
```

The notebook expects the dataset at `data/ex1data1.txt`, relative to the notebook's working directory. The dataset is not included in the notebook itself.

## Dataset

The dataset is read from `data/ex1data1.txt` and contains two columns:

| Column | Description |
|---|---|
| Population | Population of a city |
| Profit | Profit of a food truck in that city |

The notebook assigns these column names while loading the file.

## Methodology

### 1. Load and explore the data

The notebook uses pandas to read the dataset and inspect it with summary statistics and a scatter plot.

### 2. Define the cost function

The `computeCost()` function calculates the mean squared error objective used by linear regression:

\[
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2
\]

where `m` is the number of training examples, `h` is the model prediction, and `theta` contains the model parameters.

### 3. Prepare the training data

A column of ones is added to the dataset to represent the intercept term. The feature matrix `X`, target vector `y`, and initial parameter matrix `theta` are then created.

### 4. Train using gradient descent

The notebook implements `gradientDescent()` to repeatedly update the parameters and reduce the cost.

The notebook uses:
- **Learning rate (`alpha`):** `0.01`
- **Number of iterations:** `1000`
- **Initial parameters:** `[0, 0]`

### 5. Visualize the results

Matplotlib is used to create:
- A scatter plot of the training data with the learned regression line.
- A cost-versus-iteration plot to show the change in error during training.

## Getting Started

### Prerequisites

Install Python and Jupyter Notebook. Install the required libraries:

```bash
pip install numpy pandas matplotlib jupyter
```

### Run the notebook

1. Clone or download this project.
2. Place `ex1data1.txt` inside a folder named `data` in the project directory.
3. Open a terminal in the project directory.
4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `Linear_Regression.ipynb`.
6. Run the notebook cells in order.

Make sure the notebook's working directory contains the `data` folder so the dataset path can be resolved.

## Output

After training, the notebook displays:
- The learned parameter matrix.
- The cost values recorded during gradient descent.
- The final cost computed using the learned parameters.
- A graph showing the fitted prediction line and training examples.
- A graph showing cost against the number of iterations.

The notebook does not provide a fixed numerical accuracy score; the learned parameters and plots are produced when the notebook is executed with the dataset.

## Reference

- Andrew Ng, *Machine Learning*, Coursera — linear regression exercise.

## Conclusion

This project demonstrates the basic workflow of univariate linear regression: defining a cost function, optimizing model parameters with gradient descent, and visualizing the fitted line. It provides a simple example of how population data can be used to estimate food truck profit.
