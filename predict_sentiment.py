#%%
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

cv: TfidfVectorizer = joblib.load("count_vectorizer.joblib")
model: LogisticRegression = joblib.load("sentiment_model.joblib")


#%%
s = "$AAPL beats earnings again, iPhones to the mooooooon!!!"
s = "$TSLA sales down 140%, revenue is now negative and elon musk landed on mars high"
#%%
tfidf_repr = cv.transform([s])
model.predict(tfidf_repr)

#%%
