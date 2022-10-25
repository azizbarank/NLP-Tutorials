# Transformers As Feature Extractors

Transformers continue to be one of the most frequently used models for various NLP tasks since 2017. However, due to their high computational resource requirements and difficult maintenance, they may not be the most efficient choice out there all the time. This is especially true for simple sentiment classification tasks. In such circumstances, among the alternatives, there is the feature-based approach, where we use transformers as feature extractors for a simple model. What crucial in this approach is that since a transformer’s body weights are frozen, the hidden states need to be precomputed only once for them to be used as features for the model, meaning that the whole process does not require high computational resource compared to fine-tuning the whole model and provides an alternative to doing traditional machine learning and deep learning. 

In this post, we will do a simple binary sentiment classification task using Rotten Tomatoes movie review dataset. We will obtain it through the Hugging Face Dataset library and make use of DistilRoBERTa to provide our simple Logistic Regression model with the features it needs to be trained with. 

## Quick Intro: DistilRoBERTa

[DistilRoBERTa](https://huggingface.co/distilroberta-base) is a distilled version of the RoBERTa base model, which is an improved version of BERT due to longer training with more training data while using only Masked Language Modeling (MLM) objective. On average, DistilRoBERTa is twice as fast as RoBERTa because of having much less parameters of 82M (6 layers, 768 dimension and 12 heads).


## 1. Environment Setup

Our first step is to install the Hugging Face’s Dataset and Transformers libraries.

```python
!pip install datasets transformers
```
Then, we need to load the other needed dependencies.

```python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
```
## 2. Exploring our Dataset

We use the Rotten Tomatoes dataset for our purpose. It consists of movie reviews which are labeled as “1” and “2” that stand for “positive” and “negative”, respectively.

We use the Hugging Face Dataset library to load it.

```python
from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes")
```
Then, we can start looking at our dataset object:

```python
dataset
# DatasetDict({
#   train: Dataset({
#        features: ['text', 'label'],
#        num_rows: 8530
#    })
#    validation: Dataset({
#        features: ['text', 'label'],
#        num_rows: 1066
#    })
#    test: Dataset({
#        features: ['text', 'label'],
#        num_rows: 1066
#    })
#})
```
As can be seen, our dataset consists of three splits, each of them having two columns of “text” and “label”. 
Now, to explore our dataset further and with more convenience, let’s change its output format to a Pandas DataFrame so that we can inspect and visualize it more easily.

```python
dataset.set_format(type="pandas")
df = dataset["train"][:]
df.head()
#                                                    text   label
#0	the rock is destined to be the 21st century's ...	1
#1	the gorgeously elaborate continuation of " the...	1
#2	effective but too-tepid biopic	1
#3	if you sometimes like to go to the movies to h...	1
#4	emerges as something rare , an issue movie tha...	1
```

Then, for future use, let’s also add a column that corresponds to “positive” and “negative” for the integers of “1” and “2”, respectively.

```python
def label_str(row):
  return dataset["train"].features["label"].str(row)

df["label_name"] = df["label"].apply(label_str)
df.head()

#                                                    text   label label_name
#0	the rock is destined to be the 21st century's ...	1	pos
#1	the gorgeously elaborate continuation of " the...	1	pos
#2	effective but too-tepid biopic	1	pos
#3	if you sometimes like to go to the movies to h...	1	pos
#4	emerges as something rare , an issue movie tha...	1	pos
```

Finally, as part of exploring the dataset, we can look at its class distribution to see whether it is balanced or not. To do this, we can use Pandas and Matplotlib.

```python
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()
# neg= 5000
# pos = 5000
```

As can be seen, our dataset is pretty balanced as it contains the same amount of “positive” and “negative” labels. Therefore, we don’t have to apply any methods used for class imbalance.

## 3. Preprocessing

Now that we have enough insight on our dataset, we can start preprocessing it. In this context, preprocessing involves tokenization and then encoding our raw text strings so that we can feed them through our classification model. To achieve this, as said at the beginning of this post, we are going to use DistilRoBERTa.

First, we need to load the tokenizer of it through AutoTokenizer class of Hugging Face Transformers library.

```python
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
```
Then, since we are dealing with a dataset instead of a single string, we should define a function to apply our model for a whole tokenization.

```python
def tokenize(batch):
  return tokenizer(batch["text"], padding=True, truncation=True)
```
Now, we will use the map() method to apply this function to our dataset as a single batch.

```python
dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)
```
```python
print(dataset_encoded["train"].column_names)
#['text', 'label', 'input_ids', 'attention_mask']
```
## 4. Obtaining The Hidden States

Now that we have our token encodings, we can proceed to the next step, which is obtaining the last hidden state to fit our classification model with. Basically, to do this, we need to convert our token encodings to token embeddings and then feed them through encoder stack to get our hidden states, all of which can be done through the AutoModel class of Hugging Face Transformers. Since we used the DistilRoBERTa’s tokenizer, it would be the best to use the same model here as well.

Let’s load the model first.

```python
from transformers import AutoModel

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
```
Now, there is something to keep in mind. To get embeddings from our encoded tokens, our model needs PyTorch tensors. Only then can we apply our function to get the last hidden states. Therefore, from now on we have two things to do:

* Define a function to be applied to the whole dataset via map() method.
* Before using this function immediately, convert the encoded tokens we obtained before to PyTorch tensors.

Let’s do the first step:

```python
def extract_hidden_states(batch):
  inputs = {k:v.to(device) for k,v in batch.items()
  if k in tokenizer.model_input_names}
  with torch.no_grad():
    last_hidden_state = model(**inputs).last_hidden_state
  return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
```
Then the second step:

```python
dataset_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```
Finally, we can use the function:

```python
dataset_hidden = dataset_encoded.map(extract_hidden_states, batched=True)
```

As a result, a new column that consists of hidden state vectors is added to our dataset:

```python
dataset_hidden["train"].column_names
#['text', 'label', 'input_ids', 'attention_mask', 'hidden_state']
```

## 5. Training & Results

Since we finally have our input features, by using the corresponding labels, we can create a feature matrix. To do this, famous Python library scikit-learn can be used.

```python
X_train = np.array(dataset_hidden["train"]["hidden_state"])
X_valid = np.array(dataset_hidden["validation"]["hidden_state"])
y_train = np.array(dataset_hidden["train"]["label"])
y_valid = np.array(dataset_hidden["validation"]["label"])
```
Now, let’s fit our Logistic Regression model and see the results in terms of accuracy.

```python
from sklearn.linear_model import LogisticRegression
```

```python
clf = LogisticRegression(max_iter=3000)
clf.fit(X_train, y_train)
clf.score(X_valid, y_valid)
#0.8292682926829268
```
This seems like a really impressive result. However, to be sure how “great” our accuracy is, we can use scikit-learn’s DummyClassifier to get a baseline to compare our result with.

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)
#0.5
```
Indeed, our model did a very good job!

## Conclusion
We successfully used DistilRoBERTa’s embeddings and a Logistic Regression model to do a binary sentiment classification on movie reviews from Rotten Tomaotes. In the end, we managed to get a 83% accuracy without spending high computational resource. 

It should not be forgotten that this method is applicable to other tasks and that the other models can be used instead of DistilRoBERTa as well.
