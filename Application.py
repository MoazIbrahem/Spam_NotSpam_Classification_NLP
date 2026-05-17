import re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter

import nltk
for r in ["punkt","punkt_tab","stopwords","wordnet","averaged_perceptron_tagger","averaged_perceptron_tagger_eng"]:
    nltk.download(r, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack
from xgboost import XGBClassifier

st.set_page_config(page_title="NLP Spam Detector", page_icon="📩", layout="wide")
st.title("📩 NLP Spam Detector")

with st.sidebar:
    uploaded = st.file_uploader("Upload dataset.csv", type="csv")
    section  = st.radio("Section", [
        "1 · Data & Cleaning",
        "2 · Preprocessing",
        "3 · Features",
        "4 · Models",
        "5 · Deploy",
    ])

if not uploaded:
    st.info("⬆️ Upload dataset.csv to start.")
    st.stop()

Words_dict = {'u':'you','ur':'your','r':'are','4u':'for you','gr8':'great','b4':'before','plz':'please','msg':'message','txt':'text'}
STOP = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def replace_words(t): return " ".join(Words_dict.get(w,w) for w in str(t).lower().split())
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r'https?://\S+', ' <url> ', t)
    t = re.sub(r'£|\$|€', ' <cur> ', t)
    t = re.sub(r'\b\d+\b', ' <num> ', t)
    t = re.sub(r'[^a-z\s<>]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()
def is_special(w): return bool(re.match(r'^https?://',w)) or w.isdigit()

@st.cache_data(show_spinner="Processing data…")
def load(data):
    df = pd.read_csv(pd.io.common.BytesIO(data)).drop_duplicates()
    df['text']       = df['text'].apply(replace_words).apply(clean_text)
    df['tokens']     = df['text'].apply(lambda t: re.findall(r'<num>|<url>|<cur>|[\w]+', t))
    df['tokens']     = df['tokens'].apply(lambda ws: [w for w in ws if w not in STOP])
    df['stemmed']    = df['tokens'].apply(lambda ws: [stemmer.stem(w) if not is_special(w) else w for w in ws])
    df['lemmatized'] = df['tokens'].apply(lambda ws: [lemmatizer.lemmatize(w) if not is_special(w) else w for w in ws])
    df['clean_text'] = df['lemmatized'].apply(" ".join)
    return df

@st.cache_resource
def build_features(_df):
    tfidf     = TfidfVectorizer(max_features=3000,ngram_range=(1,1),min_df=2,max_df=0.90,sublinear_tf=True)
    count_vec = CountVectorizer(max_features=3000,ngram_range=(1,1),min_df=2,max_df=0.90)
    char_vec  = TfidfVectorizer(analyzer='char',ngram_range=(2,4),max_features=2000,min_df=2,max_df=0.90)
    X1 = tfidf.fit_transform(_df['clean_text'])
    X2 = count_vec.fit_transform(_df['clean_text'])
    X3 = char_vec.fit_transform(_df['clean_text'])
    def get_pos(text):
        tags = pos_tag(word_tokenize(text))
        return [sum(1 for _,t in tags if t.startswith(p)) for p in ['NN','VB','JJ']]
    X4 = np.array(_df['clean_text'].apply(get_pos).tolist())
    X5 = np.column_stack([
        _df['clean_text'].apply(lambda x: len(x.split())),
        _df['clean_text'].apply(len),
        _df['clean_text'].apply(lambda x: sum(c.isdigit() for c in x)),
        _df['clean_text'].apply(lambda x: x.count('<url>')),
    ])
    X6 = np.column_stack([
        _df['clean_text'].apply(lambda t: max(Counter(t.split()).values())/len(t.split()) if t.split() else 0),
        _df['clean_text'].apply(lambda t: len(set(t.split()))/len(t.split()) if t.split() else 0),
    ])
    return X1, X2, X3, X4, X5, X6

@st.cache_resource
def train_models(_X1, _X2, _X3, _y):
    final = hstack([_X2, _X1, _X3])
    X_train,X_test,y_train,y_test = train_test_split(final,_y,test_size=0.2,random_state=42,stratify=_y)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        use_label_encoder=False, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    svm = CalibratedClassifierCV(LinearSVC(max_iter=3000, C=1.0), cv=3)
    svm.fit(X_train, y_train)
    return xgb, svm, X_test, y_test

df = load(uploaded.read())

# ─── 1 · DATA & CLEANING ─────────────────────────────────────────────────────
if section == "1 · Data & Cleaning":
    st.header("Data & Text Cleaning")
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Rows", len(df))
    c2.metric("Spam", int((df['text_type']=='spam').sum()))
    c3.metric("Ham",  int((df['text_type']=='ham').sum()))

    st.subheader("Sample")
    st.dataframe(df[["text","text_type"]].head(10), use_container_width=True)

    st.subheader("Class Distribution")
    fig,ax=plt.subplots(); df["text_type"].value_counts().plot(kind="bar",ax=ax,color=["#4ade80","#f43f5e"])
    plt.xticks(rotation=0); st.pyplot(fig); plt.close()

    col1,col2 = st.columns(2)
    for col,label in [(col1,"spam"),(col2,"ham")]:
        with col:
            st.subheader(f"Word Cloud — {label.title()}")
            text=" ".join(df[df["text_type"]==label]["text"])
            wc=WordCloud(width=600,height=300,background_color='white',collocations=False).generate(text)
            fig2,ax2=plt.subplots(figsize=(6,3)); ax2.imshow(wc,interpolation='bilinear'); ax2.axis("off")
            st.pyplot(fig2); plt.close()

# ─── 2 · PREPROCESSING ───────────────────────────────────────────────────────
elif section == "2 · Preprocessing":
    st.header("Preprocessing")
    c1,c2,c3=st.columns(3)
    c1.metric("Original Vocab", len(set(df['tokens'].explode())))
    c2.metric("Stemmed Vocab",  len(set(df['stemmed'].explode())))
    c3.metric("Lemmatized Vocab", len(set(df['lemmatized'].explode())))

    st.subheader("Stem vs Lemma (200 tokens)")
    orig=df['tokens'].explode().reset_index(drop=True)
    stem=df['stemmed'].explode().reset_index(drop=True)
    lemma=df['lemmatized'].explode().reset_index(drop=True)
    n=min(200,len(orig),len(stem),len(lemma))
    st.dataframe(pd.DataFrame({"original":orig[:n],"stemmed":stem[:n],"lemmatized":lemma[:n]}),use_container_width=True,height=350)

    st.subheader("Top 10 Words")
    col1,col2=st.columns(2)
    for col,col_name,title in [(col1,'stemmed',"Stemmed"),(col2,'lemmatized',"Lemmatized")]:
        with col:
            words,counts=zip(*Counter(df[col_name].explode()).most_common(10))
            fig,ax=plt.subplots(); ax.bar(words,counts); ax.set_title(f"Top 10 {title}")
            plt.xticks(rotation=45,ha='right'); st.pyplot(fig); plt.close()

    col1,col2=st.columns(2)
    with col1:
        fig,ax=plt.subplots(); ax.hist(df['text'].str.split().apply(len),bins=30); ax.set_title("Original Length"); st.pyplot(fig); plt.close()
    with col2:
        fig,ax=plt.subplots(); ax.hist(df['clean_text'].str.split().apply(len),bins=30); ax.set_title("Clean Length"); st.pyplot(fig); plt.close()

# ─── 3 · FEATURES ────────────────────────────────────────────────────────────
elif section == "3 · Features":
    st.header("Feature Engineering")

    with st.spinner("Building features (cached after first run)…"):
        X1,X2,X3,X4,X5,X6 = build_features(df)

    rows=[]
    for name,X in [("TF-IDF",X1),("Count Vectorizer",X2),("Char n-grams",X3)]:
        rows.append({"Feature":name,"Shape":str(X.shape),"Non-zero":int((X!=0).sum()),
                     "Sparsity %":round(np.mean(X.toarray()==0)*100,1),"Memory MB":round(X.data.nbytes/1024/1024,3)})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    techniques=['TF-IDF','Count','Char']
    feat=[X1.shape[1],X2.shape[1],X3.shape[1]]
    mem=[X.data.nbytes/1024/1024 for X in [X1,X2,X3]]
    spars=[(1-X.nnz/(X.shape[0]*X.shape[1]))*100 for X in [X1,X2,X3]]
    fig,axes=plt.subplots(1,3,figsize=(12,4)); colors=['#3498db','#2ecc71','#e74c3c']
    for ax,vals,title,fmt in zip(axes,[feat,mem,spars],['# Features','Memory (MB)','Sparsity %'],['{:d}','{:.2f}','{:.1f}%']):
        ax.bar(techniques,vals,color=colors); ax.set_title(title)
        for i,v in enumerate(vals): ax.text(i,v*1.02,fmt.format(v),ha='center',fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Single-Feature Accuracy")
    y=df['text_type'].map({'ham':0,'spam':1})
    def evaluate(X):
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        m=LogisticRegression(max_iter=2000,solver='liblinear'); m.fit(Xtr,ytr)
        return accuracy_score(yte,m.predict(Xte))
    scores={"TF-IDF":evaluate(X1),"Count":evaluate(X2),"Char":evaluate(X3),
            "POS":evaluate(X4),"Length":evaluate(X5),"Behavior":evaluate(X6)}
    st.dataframe(pd.DataFrame(list(scores.items()),columns=["Feature","Accuracy"]).sort_values("Accuracy",ascending=False),
                 use_container_width=True,hide_index=True)
    fig,ax=plt.subplots(); ax.bar(scores.keys(),scores.values()); ax.set_ylabel("Accuracy"); ax.set_title("Feature Comparison")
    st.pyplot(fig); plt.close()
    st.success(f"Best: **{max(scores,key=scores.get)}** ({max(scores.values()):.4f})")

# ─── 4 · MODELS ──────────────────────────────────────────────────────────────
elif section == "4 · Models":
    st.header("Model Training & Evaluation")

    with st.spinner("Building features…"):
        X1,X2,X3,X4,X5,X6 = build_features(df)

    y = df['text_type'].map({'ham':0,'spam':1})

    with st.spinner("Training models (cached after first run)…"):
        xgb, svm, X_test, y_test = train_models(X1, X2, X3, y)

    xgb_pred=xgb.predict(X_test); xgb_prob=xgb.predict_proba(X_test)[:,1]
    svm_pred=svm.predict(X_test); svm_prob=svm.predict_proba(X_test)[:,1]

    rows=[]
    for name,pred,prob in [("XGBoost",xgb_pred,xgb_prob),("SVM",svm_pred,svm_prob)]:
        rows.append({"Model":name,
                     "Accuracy":round(accuracy_score(y_test,pred),4),
                     "Precision":round(precision_score(y_test,pred),4),
                     "Recall":round(recall_score(y_test,pred),4),
                     "F1":round(f1_score(y_test,pred),4),
                     "ROC-AUC":round(roc_auc_score(y_test,prob),4)})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    tab1,tab2=st.tabs(["XGBoost","SVM"])
    for tab,name,model,pred,prob,cm_color in [(tab1,"XGBoost",xgb,xgb_pred,xgb_prob,"Purples"),(tab2,"SVM",svm,svm_pred,svm_prob,"Oranges")]:
        with tab:
            st.code(classification_report(y_test,pred,target_names=["ham","spam"]))
            col1,col2=st.columns(2)
            with col1:
                fig,ax=plt.subplots()
                ConfusionMatrixDisplay(confusion_matrix(y_test,pred),display_labels=["ham","spam"]).plot(cmap=cm_color,ax=ax)
                ax.set_title(f"{name} — Confusion Matrix"); st.pyplot(fig); plt.close()
            with col2:
                fig,ax=plt.subplots()
                RocCurveDisplay.from_estimator(model,X_test,y_test,name=name,ax=ax)
                ax.set_title(f"{name} — ROC Curve"); st.pyplot(fig); plt.close()

# ─── 5 · TRY IT ──────────────────────────────────────────────────────────────
elif section == "5 · Deploy":
    st.header("🔍  Spam or Ham?")

    with st.spinner("Building features & training model…"):
        X1,X2,X3,X4,X5,X6 = build_features(df)
        y = df['text_type'].map({'ham':0,'spam':1})
        xgb, svm, X_test, y_test = train_models(X1, X2, X3, y)

    # vectorizers needed for transform
    @st.cache_resource
    def get_vectorizers(_df):
        tfidf     = TfidfVectorizer(max_features=3000,ngram_range=(1,1),min_df=2,max_df=0.90,sublinear_tf=True)
        count_vec = CountVectorizer(max_features=3000,ngram_range=(1,1),min_df=2,max_df=0.90)
        char_vec  = TfidfVectorizer(analyzer='char',ngram_range=(2,4),max_features=2000,min_df=2,max_df=0.90)
        tfidf.fit(_df['clean_text'])
        count_vec.fit(_df['clean_text'])
        char_vec.fit(_df['clean_text'])
        return tfidf, count_vec, char_vec

    tfidf, count_vec, char_vec = get_vectorizers(df)
    chosen_model = st.radio("Model", ["XGBoost", "SVM"], horizontal=True)
    model = xgb if chosen_model == "XGBoost" else svm

    examples = [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hey, are you coming to dinner tonight?",
        "URGENT: Your account has been suspended. Verify now at http://scam.com",
        "Can you pick up some groceries on your way home?",
    ]
    st.markdown("**Quick examples:**")
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex[:35]+"…", use_container_width=True):
            st.session_state["msg"] = ex

    msg = st.text_area("Type your message:", value=st.session_state.get("msg",""), height=100, placeholder="Enter any SMS message…")

    if msg.strip():
        # preprocess
        cleaned = clean_text(replace_words(msg))
        tokens  = [w for w in re.findall(r'<num>|<url>|<cur>|[\w]+', cleaned) if w not in STOP]
        final   = " ".join(lemmatizer.lemmatize(w) if not is_special(w) else w for w in tokens)

        # transform
        x = hstack([
            count_vec.transform([final]),
            tfidf.transform([final]),
            char_vec.transform([final]),
        ])

        pred  = model.predict(x)[0]
        prob  = model.predict_proba(x)[0]
        label = "SPAM 🚨" if pred == 1 else "HAM ✅"
        color = "#f43f5e" if pred == 1 else "#4ade80"
        conf  = prob[pred]

        st.markdown(f"""
        <div style='text-align:center;padding:2rem;border-radius:12px;border:2px solid {color};background:{color}18;margin-top:1rem'>
            <div style='font-size:3rem'>{label}</div>
            <div style='font-size:1.1rem;color:#94a3b8;margin-top:8px'>
                Confidence: <b style='color:{color}'>{conf:.1%}</b>
            </div>
            <div style='font-size:0.85rem;color:#64748b;margin-top:4px'>
                Spam prob: {prob[1]:.1%} &nbsp;|&nbsp; Ham prob: {prob[0]:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Pipeline trace"):
            st.write(f"**Cleaned:** `{cleaned}`")
            st.write(f"**Tokens:** `{tokens}`")
            st.write(f"**Final text:** `{final}`")
