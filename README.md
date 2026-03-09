**Aspect-Based Sentiment Analysis (ABSA)**

**Overview**
Aspect-Based Sentiment Analysis (ABSA) is a fine-grained sentiment analysis task that identifies specific aspects mentioned in a review and determines the sentiment associated with each aspect.

Unlike traditional sentiment analysis that assigns a single sentiment to an entire review, ABSA extracts aspect terms and predicts the sentiment related to each aspect.
This project implements multiple approaches for ABSA, combining deep learning, transformer models, and rule-based sentiment analysis techniques.

**Objectives**

1.Extract aspect terms from textual reviews
2.Determine sentiment polarity for each aspect
3.Compare traditional deep learning approaches with transformer-based models
4.Evaluate model performance using standard NLP evaluation metrics

**Approaches Used**

**1. Bi-LSTM + CRF (Sequence Labeling Approach)**

   Model Architecture
    Embedding Layer
    Bidirectional LSTM
    Conditional Random Field (CRF)
    
   Purpose
    Bi-LSTM captures contextual information from both directions of the sentence.
    CRF ensures valid sequence tagging and improves prediction consistency.
    
   Training Data
    Approximately 5000 annotated reviews were used for this experiment.
    
   Limitations
    Smaller dataset size
    Limited generalization capability compared to transformer-based models
    
**2. Transformer-Based Approach (XLM-RoBERTa)**

   Why XLM-RoBERTa?
    Transformer architecture
    Strong contextual representation
    High performance on NLP tasks
    
   Frameworks used:
    HuggingFace Transformers
    PyTorch
   
**Sentiment Extraction**

1. Aspect Sentiment Extraction using LLaMA
   For the smaller dataset used in the Bi-LSTM + CRF experiment, LLaMA was used to extract aspect-sentiment pairs.
   This approach helped generate and validate aspect-sentiment pairs for the dataset.
   
3. Sentiment Analysis using SIA
   We also experimented with Sentiment Intensity Analyzer (SIA) from the NLTK library.
   SIA generates a compound sentiment score between -1 and +1.
   Sentiment classification rule:
     compound > 0 → Positive
     compound < 0 → Negative
     compound = 0 → Neutral
   
**Dataset**

The dataset contains reviews with aspect terms and corresponding sentiments.
Example structure,
| Review                          | Aspect  | Sentiment |
| ------------------------------- | ------- | --------- |
| Food was great but service slow | food    | positive  |
| Food was great but service slow | service | negative  |

**Dataset Split**

Training Set
Validation Set
Test Set

**Evaluation Metrics**

The performance of the models was evaluated using:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix

**Technologies Used**

**Programming Language**
Python

**Libraries**
pandas
numpy
nltk
transformers
pytorch
scikit-learn
seaborn

**Models**
Bi-LSTM + CRF
XLM-RoBERTa

**Results**

The system successfully extracts aspects and predicts sentiment at the aspect level.
