# Fake News Detection Using Deep Learning Approach
# Introduction
Fake News is pervasive nowadays and is too easy to spread with social media and it is difficult for us to identify. A lot of fake news has popped up on social networks, such as Instagram, Facebook, Twitter, etc. Hence, we aim to utilize artificial intelligence algorithms to detect fake news to help people recognize it.

# Methodology
The process of fake news detection can be divided into four stages - data preprocessing, word embedding, models, and model fine-tuning.
## The Dataset
Link: https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php

The dataset we use is the ISOT Fake News dataset introduced by the ISOT Research Lab at the University of Victoria in Canada. This dataset is a compilation of several thousand fake news and truthful articles, obtained from different legitimate news sites and sites flagged as unreliable by Politifact.com.
To get insight into this dataset, we visualized it with word clouds for real and fake news respectively. Figure 1(a). shows the word cloud of the real news in the dataset, and Figure 1(b). shows one of the fake news in the dataset.

###### Figure 1(a).
![](https://i.imgur.com/07Rh4uD.png "Figure 1(a).")

###### Figure 1(b).
![](https://i.imgur.com/uSVXnyz.png "Figure 1(b).")

We can see that, in the real news word cloud, ‘Trump’, ‘say’, ‘Russia’, ‘House’, ‘North’, and Korea’ appeared frequently; while in fake news one, ‘VIDEO’, ‘Trump’, ‘Obama’, ‘WATCH’, and ‘Hillary’ appeared the most frequently. ‘Say’ appears frequently in real news but does not in fake news. ‘VIDEO’, and ‘WATCH’ appear frequently in fake news but do not in real news. From these two word clouds, we can get some important information to differentiate the two classes of data.
The original form of the dataset is two CSV files containing fake and real news respectively. We combined the dataset and split it into training, validation, and test sets with shuffling at the ratio of 60%:20%:20%. The original combined dataset contains 44,898 pieces of data, and Table 1. shows the distribution of data in the training, validation, and test sets.

###### Table 1. Distribution of Data
| Training | Validation | Test |
|:--------:|:----------:|:----:|
|    60%   |     20%    |  20% |
|   28791  |    7185    | 8981 |

## Data Preprocessing
The main goal of this part is to use NLP techniques to preprocess the input data and prepare for the next step to extract the proper features.  
	The data we use contains news titles and texts. Each of the titles is about 12.45 words long, while each of the texts is about 405.28 words long. In our project, we only use the titles for fake news detection because the texts are too large for us to train efficiently. Also, the text contains too many details and information for a piece of news, which may distract the models during training.  
	We built a preprocessing pipeline for each statement to eliminate the noise in the fake news dataset. The preprocessing pipeline includes the following 3 sub-parts:
    
1. Replaced characters that are not between a to z or A to Z with whitespace.
2. Converted all characters into lower-case ones.
3. Removed the stopwords.

We also cropped the titles into sentences with a maximum length of 42 in order to train the model on a dataset with sentences of reasonable lengths, and also eliminate titles with an extreme length that may let the model fit on unbalanced data.

## Word Embedding
After preprocessing the texts, a Keras tokenizer was used to tokenize the preprocessed text data. To ensure that each text was of a fixed length of 42 words, pre-padding was applied. Specifically, pre-padding involves adding a certain number of zero-value tokens to the beginning of each text, such that the resulting text length is equal to the desired length of 42 words. This was done to ensure that all texts have the same length. 
For the input of BERT, we need to convert the original statements into three kinds of tensors, which are token tensors, segment tensors, mask tensors. Token tensors represent the indices of tokens, which are obtained by the tokenizer. Segment tensors represent the identification of different sentences. Mask tensors represent the concentration of tokens including information after zero-padding the data into the same length.

## Models
* **Bidirectional LSTM**  
Bidirectional LSTM (BiLSTM) consists of two LSTMs: one taking the input from a forward direction, and the other in a backward direction. BiLSTM effectively increases the amount of information available to the network, improving the context available to the algorithm.

* **BERT**  
In the Natural Language Processing field, Transformers become more and more dominant. BERT the acronym for Bidirectional Encoder Representations from Transformers, is a transformer-based machine learning technique that changed the NLP world in recent years due to its state-of-the-art performance. Its two main features are that it is a deep transformer model so that it can process lengthy sentences effectively using the ‘attention’ mechanism, and it is bidirectional so that it will output based on the entire input sentence.  
We used BERT to handle the dataset and construct a deep learning model by fine-tuning the bert-based-uncased pre-trained model for fake news detection.
Training a model for natural language processing is costly and time-consuming because of the large number of parameters. Fortunately, we have pre-trained models of BERT that enable us to conduct transfer learning efficiently. We choose the pre-trained model of bert-base-uncased from a lot of models with different kinds of parameters. The chosen one consists of a base amount of parameters and does not consider cases of letters (upper-case and lower-case).

## Model Fine-Tuning
For different downstream tasks, we need to conduct different fine-tuning approaches. Thanks to HuggingFace, we have the models for different downstream tasks. In our project of fake news detection, which is the classification of statements, we used BertForSequenceClassification to fine-tune our pre-trained BERT model. The modules of the model contain a BERT module handling various embeddings, a BERT transformer encoder, a BertPooler, a dropout layer, and a linear classifier that returns logits of the 2 classes.

## Results
To evaluate the performance of the models we constructed for the fake news detection problem, we used the most commonly used metrics:
* True Positive (TP): when predicted fake news is actually fake news
* True Negative (TN): when predicted true news is actually true news
* False Negative (FN): when predicted true news is actually fake news
* False Positive (FP): when predicted fake news is actually true news

We can define the following metrics based on the value of the above 4 situations.
![](https://i.imgur.com/fWwvp5s.png)

These 4 metrics are the most widely used in the world of machine learning, especially for classification problems. It allows us to evaluate the performance of a classifier from different perspectives. Normally, ‘Accuracy’ is the most representative metric for the evaluation because it reflects the situation of classification completely.

Table 3(a). and Table 3(b). show the performance of the 4 different models we used on balanced datasets and imbalanced datasets, with different metrics mentioned above, respectively. The results show that training on imbalanced datasets is slightly better than training on balanced datasets for our fake news detection task. The results also show that Transformer-based models are considerably better than the other models. And deep learning models with more attributes perform better than those with fewer attributes. That is, BiLSTM performs better than LSTM, and CNN-BiLSTM performs better than simple BiLSTM.

###### On Balanced Dataset
|  Model | Accuracy | Precision | Recall | F1-Score |
|:------:|:--------:|:---------:|:------:|:--------:|
| BiLSTM |  0.9710  |    0.97   |  0.97  |   0.97   |
|  BERT  |  0.9874  |    0.99   |  0.99  |   0.99   |


The results shown in Table are promising. While we need to get insight into the failure cases. Figure 2. shows the insight of the data which is wrongly classified by our BERT-based model. The vertical axis is the original classes of ‘Fake’ and ‘Real’. The horizontal axis is the amount of the data. We can see that most of the wrongly classified data are originally fake (true 0), while only a small amount of the data is originally real (true 1).

![](https://i.imgur.com/EUDMV1L.png)

# Conclusions
In recent years, fake news detection plays an important role in national security and politics. In this paper, we covered the implementation of deep learning model (BiLSTM) and Transformer-based models (BERT) that have been proposed for fake news detection on the ISOT Fake News dataset. We applied the balanced datasets with preprocessing and word embedding to get word sequences, and then input these sequences into our models. We found that the experimental results of our models are close to each other but clearly show the improvement from the BiLSTM model to the BERT model. BERT is the best of all the models due to its best results and performances in both datasets.
