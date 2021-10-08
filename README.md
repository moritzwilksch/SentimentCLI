# SentimentCLI
This is a CLI to use a simple sentiment analysis machine learning model
<img width="565" alt="image" src="https://user-images.githubusercontent.com/58488209/136592024-a6e74173-0edf-4a71-8e26-42338a111272.png">

# Usage
```bash
python sentistonks.py
```
or
```bash
python sentistonks.py --tweet "<TEXT HERE>"
```
# How it Works
* 3,000 tweets were collected and manually annotated as *POSITIVE*, *NEGATIVE* or *NEUTRAL*
* The sentiment model is a logistic regression trained on the preprocessed version of these tweets
* The CLI preprocesses the input and invokes the model for prediction
