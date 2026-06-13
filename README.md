Email & SMS Spam Detection System

An end-to-end Machine Learning and Natural Language Processing (NLP) pipeline built to classify text messages and emails into Ham (Legitimate) or Spam. 

This project benchmarks two different classification algorithms side-by-side, utilizes advanced text normalization techniques, and features an interactive prediction interface for testing custom inputs.

Key Project Features
* Advanced NLP Preprocessing: Cleans raw text data using regex, tokenization, lowercase normalization, standard English stop-word removal, and WordNet Lemmatization to reduce words to their semantic root forms.
* Feature Extraction: Implements TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization to convert textual features into high-dimensional numerical vectors (restricted to the top 3,000 distinct terms).
* Model Benchmarking: Automatically trains and evaluates both a probabilistic classifier (Multinomial Naive Bayes) and a linear classifier (Logistic Regression).
* Performance Visualization: Generates a real-time matplotlib bar chart comparing performance metrics directly in the environment.
* Live Inference Tester: Includes an interactive code block at the conclusion of the pipeline allowing users to feed custom strings to the trained model for instant spam risk assessment.

Pipeline Architecture:

[ Raw Message Text ] ──►

──►
[ Preprocessing ]    ──► Lowercasing, Regex Filtering, Stop-word Removal & Lemmatization
──►

──►
[ Vectorization ]    ──► Convert tokens to numerical weights using TF-IDF (3000 Features)
──►

──►
[ Benchmarking ]     ──► Train Parallel Classifiers (Naive Bayes vs. Logistic Regression)
──►

──►
[ Live Inference ]   ──► Test custom input strings against the winning model


Model Performance & Benchmarking
The models were evaluated on an 80/20 train-test split using a verified dataset containing **5,572 records**. The classification accuracy results achieved on the evaluation subset are detailed below:

| Machine Learning Model | Classification Accuracy |
| Logistic Regression | 97.22% |
| Multinomial Naive Bayes | 96.59% |

Key Takeaways:
* The Winner: Logistic Regression yielded the superior classification score on the unseen evaluation dataset with an accuracy of 97.22%.
* Efficacy: Both models displayed incredibly robust accuracy thresholds (well above 96%), proving that the advanced text cleaning and Lemmatization combination successfully highlighted strong predictive spam indicators.

Tech Stack & Dependencies
This system was built completely in Python using the local Jupyter Notebook framework. The primary core libraries include:
* Data Manipulation: `pandas`, `numpy`
* Machine Learning Framework: `scikit-learn`
* Natural Language Processing: `nltk` (Natural Language Toolkit)
* Visualization Engine: `matplotlib`, `seaborn`

How to Run the Project Locally

1. Prerequisites
Ensure you have Python installed on your machine. Install all necessary ecosystem dependencies via pip:
```bash pip install pandas numpy matplotlib seaborn scikit-learn nltk```

2. Execution Step
* Clone this repository or download the project files to a local directory.
*  Place your spam.csv file inside the exact same directory as the notebook.
*  Open your project folder in your IDE (such as VS Code) and open Email_Spam_Detection.ipynb.
*  Update the path reference in the first cell if necessary to target your local dataset layout:
            ``` df = pd.read_csv('spam.csv', encoding='latin-1')```
* Run the cells sequentially from top to bottom to observe data cleaning, model training, performance charts, and evaluation results!
