# Email Spam-Ham Classifier

This project is an email classifier that categorizes emails as either Spam or Ham (not spam) using Naive Bayes models and machine learning techniques.

<img src="/static/gif/demo.gif" alt="Gif">

## Table of Contents
1. [Introduction](#introduction)
2. [Used Technologies](#used-technologies)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Training and Algorithms](#model-training-and-algorithms)
5. [Usage](#usage)

## Introduction

Using a machine learning approach, the **Email Spam-Ham Classifier** is designed to identify whether an email is spam or ham. The project uses the Naive Bayes algorithm to build the classifier, with the model trained on a dataset of emails labeled as either spam or ham. The models were implemented manually instead of using libraries.

## Used Technologies

This project was developed using the following technologies:

[![python](https://img.icons8.com/color/50/000000/python.png)](https://www.python.org/) [![flask](https://img.icons8.com/color/50/000000/flask.png)](https://flask.palletsprojects.com/en/3.0.x/) [![javascript](https://img.icons8.com/color/50/000000/javascript.png)](https://www.javascript.com/) ![html](https://img.icons8.com/?size=50&id=20909&format=png&color=000000) ![css](https://img.icons8.com/?size=50&id=21278&format=png&color=000000) [![Google Colab](https://img.icons8.com/color/48/000000/google-colab.png)](https://colab.research.google.com/) [![sklearn](https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/70px-Scikit_learn_logo_small.svg.png)](https://scikit-learn.org/) [![nltk](https://miro.medium.com/v2/resize:fit:50/1*YM2HXc7f4v02pZBEO8h-qw.png)](https://www.nltk.org/) [![numpy](https://img.icons8.com/color/48/000000/numpy.png)](https://numpy.org/) [![pandas](https://img.icons8.com/color/50/000000/pandas.png)](https://pandas.pydata.org/)

## Dataset Preparation

The dataset for this project includes 6000 emails, with 3000 labeled as spam and 3000 labeled as ham. The preparation steps are documented in the provided Google Colab notebooks.

The dataset includes the following columns:
- `label`: 1 for spam, 0 for ham.
- `text`: The content of the email.

Additional columns added in Google Colab:
- `charCount`: Character count in the email.
- `wordCount`: The word count is in the email.
- `SentCount`: The sentence count is in the email.
- `processedText`: Processed version of the email text.

The dataset and the Google Colab notebooks used for data preparation are in the repository's `notebooks` folder:
- [emailClassifierDataProcessing.ipynb](data/emailClassifierDataProcessing.ipynb)

## Model Training and Algorithms
### Naive Bayes
Naive Bayes is a family of simple yet effective probabilistic classifiers based on Bayes' theorem. The "naive" assumption is that the features in the dataset are independent of each other given the class label.

### Bayes' Rule
Bayes' theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event. The formula is given by:
$$\[ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \]$$
Where:
- $\( P(A|B) \$) is the posterior probability of class $\( A \$) given the feature $\( B \$).
- $\( P(B|A) \$) is the likelihood of feature $\( B \$) given class $\( A \$).
- $\( P(A) \$) is the prior probability of class $\( A \$).
- $\( P(B) \$) is the prior probability of feature $\( B \$).

**Explanation of Terms:**
- **Posterior Probability ($\( P(A|B) \$))**: The probability of the class after observing the feature.
- **Likelihood ($\( P(B|A) \$))**: The probability of the feature given the class.
- **Prior Probability ($\( P(A) \$))**: The initial probability of the class before observing the feature.
- **Prior Probability of the Feature ($\( P(B) \$))**: The initial probability of the feature.
  
### Gaussian Naive Bayes
In Gaussian Naive Bayes, the continuous values associated with each class are assumed to be distributed according to a Gaussian (normal) distribution. The probability is given by:
$$\[ P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp \left( -\frac{(x_i - \mu_y)^2}{2\sigma_y^2} \right) \]$$
A small epsilon value is added to the variance to prevent division by zero.

Where:
- $\( x_i \$): The value of the $\(i\$)-th feature.
- $\( y \$): The class label (e.g., spam or ham).
- $\( \mu_y \$): The mean (average) of the $\(i\$)-th feature for the class $\( y \$).
- $\( \sigma_y^2 \$): The variance of the $\(i\$)-th feature for the class $\( y \$).
- $\( \exp \$): The exponential function.
- $\( \pi \$): Pi, a constant approximately equal to 3.14159.

**Explanation of Terms:**
- **Mean ($\(\mu_y\$))**: The average value of the feature for a particular class. It represents the central tendency of the data for that class.
- **Variance ($\(\sigma_y^2\$))**: A measure of how much the feature values vary from the mean. It quantifies the spread of the feature values.
- **Epsilon ($\(\epsilon\$))**: A small value added to the variance to prevent division by zero, ensuring numerical stability.
  
The implementation of the Gaussian Naive Bayes model can be found in the file:
- [gaussianNB.py](models/gaussianNB.py)

### Binomial Naive Bayes
In Binomial Naive Bayes, the features are assumed to be binary (word present or not). The probability is given by:
$$\[ P(x_i|y) = \frac{(n_{y,x_i} + \alpha)}{(n_y + \alpha N)} \]$$

Where:
- $\( x_i \$): The $\(i\$)-th feature (e.g., the presence or absence of a specific word).
- $\( y \$): The class label (e.g., spam or ham).
- $\( n_{y,x_i} \$): The number of occurrences of feature $\( x_i \$) in class $\( y \$).
- $\( n_y \$): The total number of occurrences of all features in class $\( y \$).
- $\( \alpha \$): The Laplace smoothing parameter is used to prevent division by zero.
- $\( N \$): The total number of features.

**Explanation of Terms:**
- **$\( n_{y,x_i} \$)**: The count of how many times the feature $\( x_i \$) appears in emails of class $\( y \$).
- **$\( n_y \$)**: The total count of all features in emails of class $\( y \$). It represents the sum of occurrences of all features for a particular class.
- **Laplace Smoothing ($\(\alpha\$))**: A technique used to handle zero probabilities by adding a constant (usually 1) to the counts. This ensures that no probability is ever zero.
- **Total Number of Features ($\( N \$))**: The total number of unique features (e.g., unique words) across all classes.
  
The implementation of the Binomial Naive Bayes model can be found in the file:
- [binomialNB.py](models/binomialNB.py)

### Text Vectorization
The Count Vectorizer from the `sklearn` library was used to transform text into numerical vectors. This process converts the processed emails into a matrix of token counts, which can then be used as input for the Naive Bayes models.

### Model Performance
1. **Gaussian Naive Bayes**: Accuracy - 0.595
2. **Binomial Naive Bayes**: Accuracy - 0.962

The Binomial Naive Bayes performed better and was chosen as the final model.

## Usage

To run the email spam-ham classifier application, follow these steps:

1. **Install Requirements:**
   - Please make sure you have Python installed.
   - Install the necessary dependencies by running:
     ```sh
     pip install -r requirements.txt
     ```

2. **Run the Application:**
   - Once the dependencies are installed, run the application using:
     ```sh
     python app.py
     ```

This will start the Flask server, and you can interact with the email classifier through the web interface.

## Additional Resources
- **Pickle Files**: Pre-trained model pickle files are available.
