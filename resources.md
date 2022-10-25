# Dealing with Lack of Computational Resource in NLP

For a while I was thinking of sharing my personal experience about how I, as a student, have tried to overcome the challenge of unavailability of computational resources like lack of proper GPUs, memory storage or even high latency when doing NLP tasks. So, I decided to list all of the things I have learned so far and make a blog post out of them.
Disclaimer: This post is not meant to be the exhaustive list of all of the options available out there, but rather my own personal journey throughout my education so far.

Let’s get started!

# Main methods

To put every method in order, I thought it would be the best to go from “simple” to “more complex/costly” tasks. Therefore, I’m going to continue in the following order:

* Traditional Machine Learning
* Basic Deep Learning
* Transformers as Feature Extractors
* Knowledge Distillation / Quantization / Weight Pruning
* Buying GPUs

## 1. Traditional Machine Learning:

In NLP tasks, a lot of people assume the transformers would be the best solution to their problems at hand. However, I have seen that this is a pure misunderstanding and may result in undesired problems (e.g., overly costly procedure, difficulty in maintenance of such big models). In general practice, it is suggested that people should try more simple algorithms and see if they solve the problems efficiently enough. That’s why, traditional machine learning can be considered as a good starting point.

Libraries I use to do basic ML:

a)  *Scikit-learn*: A Python library that is widely used by numerous data scientists. For simple NLP tasks like text classification, algorithms of scikit-learn (e.g., Naive Bayes, Logistic Regression, SVM) could be tried first to obtain a performance baseline. 

b)  *Gensim*: Another useful Python library based on statistical machine learning. It is used for general unsupervised tasks, especially for topic modeling with the algorithms of:

•   Latent Semantic Indexing (LSI, LSA)

•   Latent Dirichlet Allocation (LDA)  

•   Hierarchical Dirichlet Process (HDP)

Aside from these two heavily ML-based libraries, I thought I should mention the other ones that might be of use for doing specific NLP tasks despite them not being exclusively for doing ML:

a)  *NLTK*: A well-known Python library that can be used instead of big models when it comes to doing basic NLP tasks including tokenization, part of speech (POS) tagging and named entity recognition (NER).

b)  *spaCy*: NLP library based on Cython, thus with higher performance compared to NLTK. It could be an alternative for doing basic NLP tasks for more than 50 languages when the typical language models are compute-intensive.

## 2. Basic Deep Learning

Sometimes typical ML algorithms may not suffice to solve the NLP problems that require more complex approaches. In this case one might try to implement some of the well known neural network architectures such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Long-Short Term Memory (LSTM), and Gated Recurrent Units (GRUs). Although there are far simpler neural network architectures, it is known that for NLP tasks, the aforementioned ones seem to have much better performance and thus should be preferred.

There are several frameworks/libraries to implement these neural networks as well. However, it should be kept in mind that most of these frameworks use GPUs to perform tensor computations on them. Therefore, in terms of cost, they lie between traditional machine learning and using large transformer models like XLM-R. Nevertheless, since these frameworks are at the heart of nearly all of NLP today, the relatively higher cost compared to one faced in machine learning tends to be worth it most of the time.

Now, I’m going to talk about the frameworks I personally have used so far and then mention the one that could be worth learning.

a)  *PyTorch*: It is a rapidly growing framework based on Torch and was developed by Facebook’s AI Research Lab (FAIR). Since it has a similar interface to NumPy, it is Pythonic-like, which makes it easier for people with a Python background to learn it. 
Additionally, due to native support for asynchronous execution, it has a very efficient optimization for training neural networks.

b)  *TensorFlow*: It is based on the Theano framework and was developed by Google Brain. Personally, I find two of its features the most interesting:

* The full integration of Keras (with TensorFlow 2.0)
* Built-in Visualization with TensorBoard

There are other useful features of it including the ability to use it in mobile platforms thanks to TensorFlow Lite, and support for frameworks other than Python like TensorFlow.jl and TensorFlow Serving for production. 
However, it might be a little bit more difficult to learn it in comparison with PyTorch since it has a steeper learning curve. 

Mentions:

•   *Jax*: This is a framework that is basically based on *autograd* (just like PyTorch is) and *XLA* (Accelerated Linear Algebra) from TensorFlow. The latter one is especially important in that it speeds up execution and decreases the memory usage through low-level operations. 

However, despite its attractive features, it is recommended that people (especially students) first get familiar with either PyTorch or TensorFlow + Keras, because it is much more probable that a lot of them will spend their time doing standard modeling involving experiments that heavily depend on GPUs rather than TPUs exclusively. Jax is generally recommended if one is determined to use NumPy on accelerators (GPUs and TPUs) or do non-standard modeling in deep learning.  

## 3. Transformers as Feature Extractors
This method can be considered as a middle way between traditional/pure machine learning and using transformer models exclusively for the task since it is a combination of the two. Basically, a transformer model is used only to obtain the last hidden states by making token embeddings going through the encoder stack, which means that fine-tuning (training all of the model parameters) the transformer model is evaded. After getting these states, they are fed into the basic ML model to finally obtain the results. 

For further details and the implementation of this method, you can check out my previous "Transformers as Feature Extractors" post. 

## 4. Knowledge Distillation / Quantization / Weight Pruning
After some point, even using transformers only as feature extractors will not suffice to get satisfactory results. In fact, for most of the complicated real life problems, transformers are fine-tuned and deployed as a whole, rather than as the bodies of the ML model heads. 
However, this approach has the disadvantage of the possibility that fine-tuning the model will require much memory or GPU usage, or that the model will be too slow because of the number of total parameters it has even if we manage to fine-tune it.

Instead of returning to ML / Basic DL, there are several methods to make the transformer models as efficient as possible to use them despite their disadvantages, which are:

•   Knowledge Distillation

•   Quantization

•   Weight Pruning

Briefly, we can obtain accuracies that are pretty close to that of the original models we apply these methods to and have more compact and smaller models in the end. However, I have to mention that these three methods are normally for the problems occuring mainly in deployments, rather than during experiments, yet, since the deployment is one of the most frequent tasks done in NLP even by students, I thought to include them in this post besides the other solutions nevertheless.

### 1) Knowledge Distillation:
The process where the knowledge of a big model (also called “teacher”) is transferred to a smaller and more compact model (“student”) that requires less memory and computational capacity. Therefore, at the end of this approach, we can get a student model with a performance really close to its teacher while being easier to deploy. 

Before going into details about the types of knowledge distillation, I want to point out that there are already distilled models in Hugging Face Hub that can be used for various NLP tasks. Therefore, I would recommend trying them out rather than directly trying to apply the method to a random big model. In fact, I think they will be enough for most of the students and even developers to solve the NLP-related problems they would have. 

Personally, I mainly DistilBERT and DistilRoBERTa. However, for other models Hugging Face Hub can be checked out.


Though rarely, someone might still need to distill a model from scratch to make a new one. In this case, I would like to point out that there are two main approaches for this:

a)  *Task-agnostic*: Basically, it is doing the distillation during pre-training. Accordingly, the student model is going to need fine-tuning on a downstream task to be used effectively.

b)  *Task specific*: In this approach, there are actually two successive steps of distillation happening (one during pre-training, the other during the adaptation phase). The idea is that the teacher model will “help” the student through the complementary knowledge it has by augmenting the cross-entropy loss during fine-tuning.

### 2) Quantization: 
This is a method of representing the model parameters by using low precision data types of 8-bit and 16-bit instead of 32-bit floating point (FP32) on CPUs and GPUs. This will lead to much faster integer-based numerical operations involving matrices and accordingly less memory storage. 

### 3) Weight Pruning:
As the name suggests, this method is done by removing weights between the neurons, which leads to the whole network’s being sparser. There are two main approaches to apply it:

a)  *Magnitude Pruning*: The simplest method of pruning where each time the weight with the smallest “magnitude” is removed from the whole network. It works best when the task at hand is pure supervised learning.

b)  *Movement Pruning*: Another method that is based on fine-tuning, thus a better fit for tasks involving the transfer learning approach. In contrast to magnitude pruning, both weights with small and high magnitudes may be removed from the network according to their “movements” during training.


## 5. Buying GPUs

I suppose most of the students and new learners won’t consider this option at first since it will be of overkill for most of the experiments/projects. However, I thought to add this since it might come in handy at some point.

In this part, I’m going to divide the section as “If Colab Pro is available” and “If Colab Pro is not available” because of two reasons:

•   In my opinion, Colab Pro offers the best and the simplest solution to students and novice developers that are in need of better GPUs.

•   Sadly, it is only available in some specific countries, which leaves the people living in other places to consider other choices out of necessity. 

1)  If Colab Pro is Available

To begin with, I definitely think that Google Colab is the go-to when it comes to buying GPUs for several reasons:

*   It requires zero-configuration at the beginning and people with Gmail can easily start using it in just a few minutes.

*   It offers free GPU and TPU even in the free version. 

*   It is based on Jupyter Notebook, which, at least for me, makes it easier and simple to get started in projects. 

As said before, the only downside I can think of Colab Pro is that it is available in some specific countries. Therefore, people living elsewhere should consider other choices like the ones below.

2)  If Colab Pro is not available

a)  *Paperspace Gradient*

I think Gradient is the only simple choice besides Colab Pro for students and even for moderate non-company developers. It is much more easier to get grasp of it compared to other alternatives which I will mention below.

b)  *Amazon Web Services (AWS) / Google Cloud Platform (GCP) / Microsoft Azure*

I actually do not recommend these three to anyone except that they are professional developers in the enterprise. Additionally, aside from training models with GPUs, it is much more difficult and complicated to deploy these models compared to Paperspace in general. 

# Conclusion:

As of this writing, this has been my main experience when it comes to the difficulties arising from lack of computational resources and how to overcome these despite being a student myself. Thanks for reading and I hope these tips will be useful to you as well.
