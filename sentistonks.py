import click
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import string
from nltk.stem import PorterStemmer


def prepare_tweet_for_sentimodel(tweet: str) -> str:
    """Per word: to lower, stem, remove punctuation (keep emojies)"""
    ps = PorterStemmer()
    return " ".join(
        [
            ps.stem(x.lower().strip(string.punctuation + """”'’"""))
            for x in tweet.split(" ")
        ]
    )


@click.command()
@click.option(
    "--tweet", help="Tweet to classify the sentiment", prompt="Tweet text"
)
def classify_sentiment(tweet):
    """Simple program that greets NAME for a total of COUNT times."""
    tweet = prepare_tweet_for_sentimodel(tweet)
    tfidf_repr = cv.transform([tweet])
    result = model.predict(tfidf_repr)[0]
    
    if result == -1:
        click.echo(click.style("NEGATIVE", fg="red"))
    elif result == 1:
        click.echo(click.style("POSITIVE", fg="green"))
    else:
        click.echo("NEUTRAL")


if __name__ == "__main__":

    cv: TfidfVectorizer = joblib.load("count_vectorizer.joblib")
    model: LogisticRegression = joblib.load("sentiment_model.joblib")

    classify_sentiment()
