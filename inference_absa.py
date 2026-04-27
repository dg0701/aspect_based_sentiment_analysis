import torch
import re
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "divya07garg01/xlmr-absa-hinglish"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


id2label = {0: "negative", 1: "positive"}


nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

def extract_aspects(review):
    doc = nlp(review)
    aspects = set()


    for chunk in doc.noun_chunks:
        aspects.add(chunk.root.lemma_.lower())


    for word in doc:
        if word.dep_ == "amod" and word.head.pos_ == "NOUN":
            aspects.add(word.head.lemma_.lower())

        if word.dep_ == "nsubj" and word.head.pos_ == "ADJ":
            aspects.add(word.lemma_.lower())

    return list(aspects)



def filter_aspects(aspects):
    return [a for a in aspects if len(a) > 2]



def pos_filter(aspects):
    valid = []
    for doc in nlp.pipe(aspects):
        if len(doc) > 0 and doc[0].pos_ in ["NOUN", "PROPN"]:
            valid.append(doc.text)
    return valid



def predict_sentiment(review, aspect):
    
    text = f"aspect: {aspect} review: {review}"
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    return id2label[pred]



def analyze_review(user_input):
    
    review = preprocess(user_input)

    
    aspects = extract_aspects(review)

    
    aspects = filter_aspects(aspects)

    
    aspects = pos_filter(aspects)

    if len(aspects) == 0:
        return {"message": "No valid aspects found"}

    results = []

    
    for aspect in aspects:
        sentiment = predict_sentiment(review, aspect)

        results.append({
            "aspect": aspect,
            "sentiment": sentiment
        })

    return results



def get_predictions(text):
    output = analyze_review(text)

    pos, neg = [], []

    # Handle case when no aspects found
    if isinstance(output, dict):
        return pos, neg

    for item in output:
        if item["sentiment"] == "positive":
            pos.append(item["aspect"])
        elif item["sentiment"] == "negative":
            neg.append(item["aspect"])

    return pos, neg
