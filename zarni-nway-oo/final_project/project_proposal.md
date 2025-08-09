# Is this RED or GREEN?

Sentiment Analysis of Myanmar Political News Articles on whether they are RED (Opposition), GREEN (Pro-Junta), or Neutral.

## Problem Statement

Political discourse is complex, with viewpoints often shaped by underlying beliefs. On the internet, content is easily accessible, but safeguards against biased information that can cloud judgment are lacking. This project aims to develop an NLP model and application to classify a news article at a fundamental level: as RED (Opposition), GREEN (Pro-Junta), or Neutral. Identifying the political leaning of an article will help readers make more informed judgments about the information they consume online.

## Data Collection & Preparation

For this project, a custom data scraper was created using BeautifulSoup. The dataset will consist of 300 articles (100 each) scraped from each of the following sources: DVB (Neutral), Khitthit Media (RED), and Myawady (GREEN). Only the title and body content of the articles were extracted. For tokenization and joint NER/POS tagging, Ye Kyaw Thu's myWord and myER tools will be utilized, as they are currently the better solution for Myanmar language segmentation and tagging. A script is also being developed to assist in labeling sentences as RED, GREEN, or Neutral and for model training in ML and DL.

## Methodology

1. Scrape data from news websites to gather Myanmar news in Unicode format.
2. Tokenize the cleaned data using myWord.
3. Tag the text with POS and NER attributes using myNER.
4. Develop and apply a script for labeling sentences as RED, GREEN, or Neutral.
5. Train and evaluate various Machine Learning (ML) and Deep Learning (DL) models.
6. Test the best-performing model on real-world data.

## Model Training

For ML models, Logistic Regression, Random Forest, and SVC will be evaluated to select the best performer.
For the DL model, a custom BiLSTM network will be implemented.

## Expectations

The model will take a news article as input and output a classification of whether the article leans RED, GREEN, or Neutral.

[📊 View Sentiment Analysis Results (HTML)](./r_news_dl_line_2.html)

> Click the link above to view the expected results.

The target accuracy for the model is 70% or higher in accuracy.

## Challenges

Token-level data labeling is a significant challenge, and a manual approach would be excessively time-consuming. Despite assistance from AI coding tools, developing an optimized labeling script is a challenging task. Further research into advanced labeling techniques and model training concepts is required.

## Results

Results are yet to be concluded.