# 🐲 Project 1: Pokémon CP Predictor

A machine learning project to predict a Pokémon's "Total" combat power (CP) based on its "Attack" and "Defense" stats.

This is the first project in my machine learning portfolio. The primary goal was to understand the fundamentals of Linear Regression by building it **from scratch using NumPy** and then comparing its performance to the standard **scikit-learn** library implementation.



---

## 🚀 Core Concepts & Skills

This project covers the complete, foundational machine learning workflow from start to finish.

* **Model Implemented:** Linear Regression
* **"From Scratch" Algorithm:** Gradient Descent
* **Core Concepts:** Supervised Learning, Regression, Cost Functions (MSE), Vectorization
* **Data Science Workflow:**
    1.  **Data Loading:** Reading `Pokemon.csv` with Pandas.
    2.  **Data Preprocessing:** Selecting features (X) and target (y).
    3.  **Feature Scaling:** Using `MinMaxScaler` (or `StandardScaler`) to normalize data for gradient descent.
    4.  **Model Training:**
        * Building `MyLinearRegression` class from scratch with `.fit()` and `.predict()` methods.
        * Implementing the `scikit-learn` 5-step (Import, Load, Scale, Fit, Evaluate) pattern.
    5.  **Model Evaluation:** Calculating **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** to compare the "from scratch" model against the `scikit-learn` model.

---

## 🛠️ Tech Stack

* **Python**
* **NumPy:** For all from-scratch calculations, linear algebra (`np.dot`), and gradient descent.
* **Pandas:** For loading and manipulating the data.
* **Matplotlib:** For visualizing the data and results.
* **Scikit-learn:** For the library-based `LinearRegression` model, `train_test_split`, feature scaling, and `mean_squared_error`.

---

## 📂 Code Structure

* `pokemon_predictor.ipynb` (or `.py`): The main file containing all 5 steps of the workflow.
    * **Part 1: Data Preprocessing:** Loads and prepares the data.
    * **Part 2: `MyLinearRegression` Class:** The from-scratch implementation using NumPy and gradient descent.
    * **Part 3: Scikit-learn Implementation:** The 3-line `fit`/`predict` model for comparison.
    * **Part 4: Evaluation:** Compares the MSE of both models to prove the from-scratch version works.
* `Pokemon.csv`: The raw dataset used for training.
* `requirements.txt`: A list of all necessary libraries to run the project.

---

## 🏁 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/pokemon-cp-predictor.git](https://github.com/YOUR_USERNAME/pokemon-cp-predictor.git)
    cd pokemon-cp-predictor
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the script or notebook:**
    * If it's a **Jupyter Notebook (`.ipynb`)**:
        ```bash
        jupyter notebook pokemon_predictor.ipynb
        ```
    * If it's a **Python script (`.py`)**:
        ```bash
        python pokemon_predictor.py
        ```

---

## 📊 Dataset

The dataset used is the "Pokemon with stats" dataset, publicly available on Kaggle.

[Link to the Kaggle Dataset](https://www.kaggle.com/datasets/abcsds/pokemon)
