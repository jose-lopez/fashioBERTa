import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("Multi-Class Hebrew Text Classification")
st.markdown("### For Clothing Product Descriptions 👚")

#st.sidebar.title("Clothing Product Descriptions 👚")
DATA_URL="fashion_data.csv"

@st.cache(persist=True)
def load_data():
    df=pd.read_csv("fashion_data.csv",index_col=[0],header=[0])
    df=df.dropna()
    df=df.drop_duplicates()
    df=df.reset_index(drop=True)
    return df

@st.cache
def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(np.array(df['description']), np.array(df['category']), test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def train_MultinomialNB(cv_train_features,y_train):
    mnb = MultinomialNB(alpha=1)
    mnb.fit(cv_train_features, y_train) 
    return mnb

def generate_word_cloud(text):
    wordcloud = WordCloud(width = 600, height = 600, 
                    background_color ='white', 
                    max_words=200, 
                    min_font_size = 10,
                    font_path='david.ttf').generate(text)
    return wordcloud

df=load_data()

X_train, X_test, y_train, y_test= split_data(df)
cv=CountVectorizer(max_features=1500)
cv_train_features = cv.fit_transform(X_train)

model=train_MultinomialNB(cv_train_features,y_train)

description = st.text_input("Enter The Description")

if st.button('Predict Category'):
    cv_pred_features = cv.transform([description])
    result=model.predict(cv_pred_features)
    st.success(result[0])
#else:
    #st.write("Press the above button..")

#st.sidebar.subheader("Show random product description")
#random_category=st.sidebar.radio('Category',('dresses_women','shoes_women','coats&jackets_women'))
#st.sidebar.markdown(df.query("category == @random_category")[["description"]].sample(n=1).iat[0,0])

st.sidebar.markdown("### Number of items by category")
select=st.sidebar.selectbox('Visualization type',['Histogram','Pie chart'],key='1')

category_count=df['category'].value_counts()

#st.write(category_count)
#st.write(category_count.index[0])

category_count=pd.DataFrame({"Category":category_count.index,'Description':category_count.values})

if not st.sidebar.checkbox("Close",True):
    st.markdown("### Number of products by category")
    if select=='Histogram':
        fig=px.bar(category_count,x="Category",y='Description',color='Description',height=500)
        st.plotly_chart(fig)
    else:
        fig=px.pie(category_count,values="Description",names="Category")
        st.plotly_chart(fig)

st.sidebar.header("Word Cloud")
#word_category=st.sidebar.radio("Display word cloud for that category",('dresses_women','shoes_women','coats&jackets_women'))
word_category=st.sidebar.selectbox('Categories',category_count['Category'].values,key='4')
if not st.sidebar.checkbox("Close",True,key='3'):
    st.header('Word cloud for %s category' % (word_category))
    df_word=df[df['category']==word_category]
    cleaned_text=list()

    for i,r in df_word.iterrows():
        text=r['description']
        if isinstance(text, str):
            bidi_text = get_display(text)
            cleaned_text.append(bidi_text)

    wordcloud = generate_word_cloud(' '.join(cleaned_text))
    fig, ax = plt.subplots()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)
