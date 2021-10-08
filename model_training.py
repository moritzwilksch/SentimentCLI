# %%
import optuna
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import string
import pandas as pd
from nltk.stem import PorterStemmer


root_path = "./"
tickers = ["TSLA", "AAPL", "AMZN", "FB", "MSFT", "TWTR", "AMD", "NFLX", "NVDA", "INTC"]

# %%
df = pd.read_csv(root_path + "SRS_sentiment_labeled.csv", index_col="Unnamed: 0")
df

# %%


def prepare_tweet_for_sentimodel(tweet: str) -> str:
    """Per word: to lower, stem, remove punctuation (keep emojies)"""
    ps = PorterStemmer()
    return " ".join(
        [
            ps.stem(x.lower().strip(string.punctuation + """”'’"""))
            for x in tweet.split(" ")
        ]
    )


# %%
df.tweet = df.tweet.apply(prepare_tweet_for_sentimodel)

cv = TfidfVectorizer(token_pattern=r"[^\s]+")
bow = cv.fit_transform(df.tweet)

# %%
RETRAIN = True
if RETRAIN:

    def objective_lr(trial):
        c = trial.suggest_float("c", 1e-5, 10, log=True)
        cv_scores = cross_val_score(
            LogisticRegression(C=c, max_iter=250),
            X=bow,
            y=df.sentiment,
            n_jobs=-1,
            cv=KFold(5, shuffle=True),
        )
        return cv_scores.mean()

    def objective_nb(trial):
        alpha = trial.suggest_float("alpha", 1e-5, 10, log=True)
        cv_scores = cross_val_score(
            MultinomialNB(alpha=alpha),
            X=bow,
            y=df.sentiment,
            n_jobs=-1,
            cv=KFold(5, shuffle=True),
        )
        return cv_scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_lr, n_trials=100)

    lr = LogisticRegression(C=study.best_params["c"], max_iter=250)
    lr.fit(bow, df.sentiment)
    model = lr
else:
    cv: TfidfVectorizer = joblib.load(root_path + "count_vectorizer.joblib")
    model: LogisticRegression = joblib.load(root_path + "sentiment_model.joblib")


#%%
rev = {v: k for k, v in cv.vocabulary_.items()}
for i, w in enumerate(["NEGATIVE", "NEUTRAL", "POSITIVE"]):
    print(f"======={'='*12}=======")
    print(f"======= {w:^10} =======")
    print(f"======={'='*12}=======")
    for i in model.coef_[i].argsort()[-20:]:
        print("• " + rev[i], end=" ")

# %%

# %%
if True:
    joblib.dump(cv, root_path + "count_vectorizer.joblib")
    joblib.dump(lr, root_path + "sentiment_model.joblib")
