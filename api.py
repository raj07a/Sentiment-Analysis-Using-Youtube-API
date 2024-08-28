import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import re

API_KEY = "AIzaSyDV7Wfx8L4GAe6Daxfzpk97x1RECLfZ2ho"
CHANNEL_ID = "UCDDjMFHTsEerSEm2BvhcwrA"

# Function to fetch YouTube video data
def fetch_youtube_data(api_key, channel_id):
    videos = []
    page_token = ''
    while True:
        url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=50&pageToken={page_token}"
        response = requests.get(url).json()
        
        for item in response.get('items', []):
            if item['id']['kind'] == 'youtube#video':
                video_info = {
                    'videoId': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'publishedAt': item['snippet']['publishedAt']
                }
                videos.append(video_info)
        
        page_token = response.get('nextPageToken', '')
        if not page_token:
            break
    
    return pd.DataFrame(videos)

# Function to fetch video statistics
def fetch_video_statistics(api_key, video_ids):
    stats = []
    for video_id in video_ids:
        url = f"https://www.googleapis.com/youtube/v3/videos?key={api_key}&id={video_id}&part=statistics"
        response = requests.get(url).json()
        for item in response.get('items', []):
            stat = {
                'videoId': video_id,
                'views': int(item['statistics'].get('viewCount', 0)),
                'likes': int(item['statistics'].get('likeCount', 0)),
                'dislikes': int(item['statistics'].get('dislikeCount', 0)),
                'comments': int(item['statistics'].get('commentCount', 0))
            }
            stats.append(stat)
    return pd.DataFrame(stats)

# Function to fetch comments for a video
def fetch_video_comments(api_key, video_id):
    comments = []
    page_token = ''
    while True:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&videoId={video_id}&part=snippet&maxResults=100&pageToken={page_token}"
        response = requests.get(url).json()
        
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'videoId': video_id,
                'author': comment['authorDisplayName'],
                'text': comment['textDisplay'],
                'publishedAt': comment['publishedAt']
            })
        
        page_token = response.get('nextPageToken', '')
        if not page_token:
            break
    
    return pd.DataFrame(comments)

# Function to clean text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to perform sentiment analysis using TextBlob
def perform_sentiment_analysis(df):
    df['description'] = df['description'].apply(clean_text)
    df['description_polarity'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['description_sentiment'] = df['description_polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    return df

# Main function to fetch, process, and visualize YouTube data
def main():
    st.title('YouTube Sentiment Analysis Dashboard')

    st.write('Fetching data from YouTube API...')
    
    # Fetch video data from YouTube API
    df_videos = fetch_youtube_data(API_KEY, CHANNEL_ID)
    
    # Fetch video statistics
    df_stats = fetch_video_statistics(API_KEY, df_videos['videoId'])
    
    # Merge video data with statistics
    df = pd.merge(df_videos, df_stats, on='videoId')
    
    # Fetch comments for each video
    comments_data = []
    for video_id in df['videoId']:
        comments = fetch_video_comments(API_KEY, video_id)
        comments_data.append(comments)
    
    df_comments = pd.concat(comments_data, ignore_index=True)
    
    # Perform sentiment analysis on video descriptions
    df = perform_sentiment_analysis(df)
    
    # Perform sentiment analysis on comments
    df_comments['text'] = df_comments['text'].apply(clean_text)
    df_comments['comment_polarity'] = df_comments['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_comments['comment_sentiment'] = df_comments['comment_polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    
    # Calculate overall sentiment score
    df['overall_sentiment_score'] = (
        df['description_polarity'] +
        df_comments.groupby('videoId')['comment_polarity'].mean().reindex(df['videoId']).fillna(0)
    ) / 2
    df['overall_sentiment'] = df['overall_sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

    st.write("### Sample of YouTube Video Data:")
    st.write(df.head())

    st.header('Video Statistics')
    st.subheader('Filtered Video Data')
    st.write(df[['title', 'views', 'likes', 'dislikes', 'description_polarity', 'overall_sentiment_score', 'overall_sentiment']])

    st.subheader('Filtered Comment Data')
    st.write(df_comments[['videoId', 'author', 'text', 'comment_polarity', 'comment_sentiment']])

    st.subheader('Video Statistics Summary')
    st.write(df[['views', 'likes', 'dislikes', 'comments']].describe())

    st.subheader('Dataset Statistics Summary')
    st.write(df.describe())

    # Sentiment Distribution
    st.subheader('Overall Sentiment Distribution')
    fig1, ax1 = plt.subplots()
    df['overall_sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62', '#8da0cb'], ax=ax1)
    ax1.set_ylabel('')
    ax1.set_title('Distribution of Overall Sentiment')
    st.pyplot(fig1)

    # Like-Dislike Ratio
    st.subheader('Like-Dislike Ratio Distribution')
    df['like_dislike_ratio'] = df['likes'] / df['dislikes'].replace({0: 1})
    fig2, ax2 = plt.subplots()
    sns.kdeplot(data=df, x='like_dislike_ratio', ax=ax2, fill=True, color='skyblue')
    ax2.set_title('Distribution of Like-Dislike Ratio')
    st.pyplot(fig2)

    # Comment Polarity
    st.subheader('Comment Polarity Distribution')
    fig3, ax3 = plt.subplots()
    sns.violinplot(data=df_comments, x='comment_polarity', ax=ax3, inner='quartile', palette='muted')
    ax3.set_title('Distribution of Comment Polarity')
    st.pyplot(fig3)

    # Views vs. Likes
    st.subheader('Views vs. Likes')
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df, x='views', y='likes', hue='overall_sentiment', palette='viridis', ax=ax4)
    ax4.set_title('Views vs. Likes Colored by Sentiment')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    st.pyplot(fig4)

    # Word Cloud
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
