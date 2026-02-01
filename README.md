

🧠 TweetSentiment–Embeddings
End‑to‑end sentiment classification on Twitter data using sentence embeddings and a simple Logistic Regression classifier.
Three labels: Positive, Negative, Neutral. No heavy transformers training, just smart features + clean ML.

✨ Highlights
Uses sentence-transformers (all-MiniLM-L6-v2) for dense semantic embeddings instead of bag‑of‑words/TF‑IDF.

Fast and lightweight pipeline that runs comfortably in Google Colab.

Clear EDA: sentiment distribution, tweet length analysis, and class‑wise word clouds.
​

Classic ML classifier (Logistic Regression) on top of embeddings, with full evaluation (precision, recall, F1, confusion matrix).

Includes custom tweet predictions to demo how the model behaves on real examples.

📂 Dataset
Name: Twitter Tweets Sentiment Dataset

Source: Provided CSV link (compatible with Kaggle Twitter sentiment datasets)

Size: ~27k tweets

Columns:

text – raw tweet

sentiment – one of Positive, Negative, Neutral
​

The notebook loads the dataset directly from the URL using pandas.read_csv.

🧱 Project Structure
text
.
├── Sentiment_Classification_Embeddings.ipynb  # Main notebook
├── README.md                                  # Project documentation
└── (plots/outputs can be exported separately if needed)
All core logic lives in the notebook:

Data loading + EDA

Text preprocessing

Embedding generation

Model training & evaluation

Custom predictions

🚀 End‑to‑End Pipeline
1️⃣ Setup
Install and import all dependencies:

pandas, numpy for data wrangling

matplotlib, seaborn for visualizations

wordcloud for class‑wise word clouds

sentence-transformers for embeddings

scikit-learn for train/test split, Logistic Regression, and metrics

This step just makes sure the Colab/Jupyter environment has everything we need.

2️⃣ Load & Inspect the Data
Load the CSV into a DataFrame using the provided URL.

Preview a few rows with df.head() to confirm columns and sample content.

Use df['sentiment'].value_counts() to inspect the label distribution and check for class imbalance.
​

This gives a quick sanity check that the data is what we expect (tweets + labels).

3️⃣ EDA – Get a Feel for the Tweets
We run some lightweight EDA to understand the data:

Sentiment distribution bar plot – how many Positive, Negative, and Neutral tweets.

Tweet length histogram – character length distribution, colored by sentiment, to spot any length patterns.

Word clouds (per class) – three separate word clouds for Positive, Negative, and Neutral, so we can visually see which words dominate each sentiment.
​

These visuals help build intuition before touching any model.

4️⃣ Text Cleaning
We clean the raw tweets to make them model‑friendly:

Lowercase everything

Remove URLs and @mentions

Strip # from hashtags

Remove non‑alphabetic characters

Collapse multiple spaces into a single space
​

Output goes into a new clean_text column. This keeps the signal and throws away most of the noise.

5️⃣ Sentence Embeddings (No API Required)
Instead of sparse vectors, we use sentence embeddings:

Model: all-MiniLM-L6-v2 from sentence-transformers – small, fast, and good for semantic similarity & classification tasks.
​

To keep things efficient, we sample 5,000 tweets from the full dataset as a working subset.

For each clean_text, we call a get_embedding helper to produce a dense vector and store it in an embedding column.

Each tweet now lives as a fixed‑length numeric vector that captures its meaning, not just word counts.

6️⃣ Training: Logistic Regression on Embeddings
Once embeddings are ready:

Stack all embeddings into matrix X.

Encode text labels (Positive, Negative, Neutral) into numeric form using LabelEncoder to get y.

Split into train/test = 80/20 with stratify=y to preserve class ratios.

Train a Logistic Regression model (max_iter=200) on X_train, y_train.

Evaluate using classification_report(y_test, y_pred, target_names=classes) to get precision, recall, F1, and support.

This step shows how far you can go with a “classic” ML algorithm when the features are strong.

7️⃣ Evaluation: Confusion Matrix
Metrics are good, but we also want to see where the model struggles:

Compute confusion_matrix(y_test, y_pred).

Visualize it with a seaborn heatmap: actual labels on the y‑axis, predicted on the x‑axis.
​

Typical pattern: the model is strong on clearly Positive and Negative tweets, with most confusion happening around Neutral (which is naturally fuzzier).

8️⃣ Custom Tweet Predictions
To sanity‑check the model on “real” text, we:

Write 5 custom example tweets (clear positive, clear negative, and some in‑between).

Run them through the full pipeline: clean_tweet → get_embedding → clf.predict.

Print each tweet next to its predicted sentiment.

This makes it easy to show how the model behaves beyond the training/test split.

📊 What Does It Achieve?
A clean, reproducible notebook that:

Reads and explores a real Twitter sentiment dataset

Cleans noisy social media text

Uses modern embeddings with a classic classifier

Produces interpretable metrics and visuals (classification report + confusion matrix + EDA plots)

Demonstrates that with good embeddings, even a simple Logistic Regression can deliver solid sentiment classification performance.

🛠 Tech Stack
Language: Python

Environment: Google Colab / Jupyter Notebook

Core Libraries:

pandas, numpy

matplotlib, seaborn

wordcloud

sentence-transformers (all-MiniLM-L6-v2)

scikit-learn (LogisticRegression, LabelEncoder, train_test_split, metrics)

▶️ How to Run
Open Sentiment_Classification_Embeddings.ipynb in Google Colab or Jupyter.

Run all cells from top to bottom (no manual editing required).

Check that you see:

EDA plots + word clouds

Cleaned text column

Embedding generation

Classification report

Confusion matrix

Custom tweet predictions
​

📌 Future Ideas
Swap Logistic Regression for XGBoost or another classifier and compare metrics.

Try different embedding models (e.g., all-mpnet-base-v2) and benchmark performance.

Wrap the model into a simple API or Streamlit/Gradio app for live tweet sentiment demos.

👨‍💻 Author
Project implemented and documented by G R Harsha.
