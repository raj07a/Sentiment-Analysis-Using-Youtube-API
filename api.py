import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

API_KEY = "AIzaSyDV7Wfx8L4GAe6Daxfzpk97x1RECLfZ2ho"
CHANNEL_ID = "UCDDjMFHTsEerSEm2BvhcwrA"

def fetch_youtube_data(api_key, channel_id):
    # Same as before

def fetch_video_statistics(api_key, video_ids):
    # Same as before

def fetch_video_comments(api_key, video_id):
    # Same as before

def clean_text(text):
    # Same as before

def perform_sentiment_analysis(df):
    df['description'] = df['description'].apply(clean_text)
    df['description_polarity'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['description_sentiment'] = df['description_polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    return df

def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return 'positive' if score['compound'] > 0 else ('negative' if score['compound'] < 0 else 'neutral')

def main():
    st.title('YouTube Sentiment Analysis Dashboard')

    st.write('Fetching data from YouTube API...')
    
    df_videos = fetch_youtube_data(API_KEY, CHANNEL_ID)
    df_stats = fetch_video_statistics(API_KEY, df_videos['videoId'])
    df = pd.merge(df_videos, df_stats, on='videoId')
    
    comments_data = []
    for video_id in df['videoId']:
        comments = fetch_video_comments(API_KEY, video_id)
        comments_data.append(comments)
    
    df_comments = pd.concat(comments_data, ignore_index=True)
    
    df = perform_sentiment_analysis(df)
    df_comments['text'] = df_comments['text'].apply(clean_text)
    df_comments['comment_polarity'] = df_comments['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_comments['comment_sentiment'] = df_comments['comment_polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    df_comments['vader_comment_sentiment'] = df_comments['text'].apply(vader_sentiment)
    
    df['overall_sentiment_score'] = (
        df['description_polarity'] + 
        df_comments.groupby('videoId')['comment_polarity'].mean().reindex(df['videoId']).fillna(0)
    ) / 2
    df['overall_sentiment'] = df['overall_sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

    st.write("### Sample of YouTube Video Data:")
    st.write(df.head())

    st.write("### Sample Cleaned Descriptions:")
    st.write(df[['description', 'description_polarity']].head())

    st.write("### Sample Cleaned Comments:")
    st.write(df_comments[['text', 'comment_polarity']].head())

    st.write("### Sentiment Score Calculation:")
    df['comment_polarity_mean'] = df_comments.groupby('videoId')['comment_polarity'].transform('mean')
    st.write(df[['videoId', 'description_polarity', 'comment_polarity_mean', 'overall_sentiment_score']].head())

    st.write("### Sample VADER Sentiment Analysis Results:")
    st.write(df_comments[['text', 'vader_comment_sentiment']].head())

    st.header('Video Statistics')
    st.subheader('Filtered Video Data')
    st.write(df[['title', 'views', 'likes', 'dislikes', 'description_polarity', 'overall_sentiment_score', 'overall_sentiment']])

    st.subheader('Filtered Comment Data')
    st.write(df_comments[['videoId', 'author', 'text', 'comment_polarity', 'comment_sentiment', 'vader_comment_sentiment']])

    st.subheader('Video Statistics Summary')
    st.write(df[['views', 'likes', 'dislikes', 'comments']].describe())

    st.subheader('Dataset Statistics Summary')
    st.write(df.describe())

    st.subheader('Overall Sentiment Distribution')
    fig1, ax1 = plt.subplots()
    df['overall_sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62', '#8da0cb'], ax=ax1)
    ax1.set_ylabel('')
    ax1.set_title('Distribution of Overall Sentiment')
    st.pyplot(fig1)

    st.subheader('Like-Dislike Ratio Distribution')
    df['like_dislike_ratio'] = df['likes'] / df['dislikes'].replace({0: 1})
    fig2, ax2 = plt.subplots()
    sns.kdeplot(data=df, x='like_dislike_ratio', ax=ax2, fill=True, color='skyblue')
    ax2.set_title('Distribution of Like-Dislike Ratio')
    st.pyplot(fig2)

    st.subheader('Comment Polarity Distribution')
    fig3, ax3 = plt.subplots()
    sns.violinplot(data=df_comments, x='comment_polarity', ax=ax3, inner='quartile', palette='muted')
    ax3.set_title('Distribution of Comment Polarity')
    st.pyplot(fig3)

    st.subheader('Views vs. Likes')
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df, x='views', y='likes', hue='overall_sentiment', palette='viridis', ax=ax4)
    ax4.set_title('Views vs. Likes Colored by Sentiment')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    st.pyplot(fig4)

    st.subheader('Word Cloud of Comments')
    comment_words = ' '.join(df_comments['text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comment_words)
    fig5, ax5 = plt.subplots()
    ax5.imshow(wordcloud, interpolation='bilinear')
    ax5.axis('off')
    ax5.set_title('Word Cloud of Comments')
    st.pyplot(fig5)

if __name__ == "__main__":
    main()
