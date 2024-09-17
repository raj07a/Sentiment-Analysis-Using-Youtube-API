import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# API key and channel ID (Replace with your YouTube API Key)
API_KEY = "AIzaSyDV7Wfx8L4GAe6Daxfzpk97x1RECLfZ2ho"
CHANNEL_ID = "UCsT0YIqwnpJCM-mx7-gSA4Q"

# Function to fetch YouTube video data
def fetch_youtube_data(api_key, channel_id):
    videos = []
    page_token = ''
    while True:
        url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=50&pageToken={page_token}"
        response = requests.get(url)
        response_json = response.json()
        
        if response.status_code != 200:
            st.error(f"Error fetching YouTube data: {response_json}")
            return pd.DataFrame()  # Return an empty DataFrame
        
        if 'items' not in response_json:
            st.error("No 'items' found in the response.")
            return pd.DataFrame()
        
        for item in response_json.get('items', []):
            if item['id']['kind'] == 'youtube#video':
                video_info = {
                    'videoId': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'publishedAt': item['snippet']['publishedAt']
                }
                videos.append(video_info)
        
        page_token = response_json.get('nextPageToken', '')
        if not page_token:
            break
    
    return pd.DataFrame(videos)

# Function to fetch video statistics
def fetch_video_statistics(api_key, video_ids):
    stats = []
    for video_id in video_ids:
        url = f"https://www.googleapis.com/youtube/v3/videos?key={api_key}&id={video_id}&part=statistics"
        response = requests.get(url)
        response_json = response.json()
        
        if response.status_code != 200:
            st.error(f"Error fetching video statistics: {response_json}")
            continue
        
        if 'items' not in response_json:
            st.error("No 'items' found in the response.")
            continue
        
        for item in response_json.get('items', []):
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
        response = requests.get(url)
        response_json = response.json()
        
        if response.status_code != 200:
            st.error(f"Error fetching comments: {response_json}")
            return pd.DataFrame()  # Return an empty DataFrame
        
        if 'items' not in response_json:
            st.error("No 'items' found in the response.")
            return pd.DataFrame()
        
        for item in response_json.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'videoId': video_id,
                'author': comment['authorDisplayName'],
                'text': comment['textDisplay'],
                'publishedAt': comment['publishedAt']
            })
        
        page_token = response_json.get('nextPageToken', '')
        if not page_token:
            break
    
    return pd.DataFrame(comments)

# Function to perform sentiment analysis using TextBlob
def perform_sentiment_analysis(df):
    df['description_polarity'] = df['description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['description_sentiment'] = df['description_polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    return df

# Main function to fetch, process, and visualize YouTube data
def main():
    st.title('YouTube Sentiment Analysis Dashboard')

    st.write('Fetching data from YouTube API...')
    
    # Fetch video data from YouTube API
    df_videos = fetch_youtube_data(API_KEY, CHANNEL_ID)
    
    if df_videos.empty:
        st.warning("No video data found.")
        return
    
    # Fetch video statistics
    df_stats = fetch_video_statistics(API_KEY, df_videos['videoId'])
    
    if df_stats.empty:
        st.warning("No video statistics found.")
        return
    
    # Merge video data with statistics
    df = pd.merge(df_videos, df_stats, on='videoId')
    
    # Fetch comments for each video
    comments_data = []
    for video_id in df['videoId']:
        comments = fetch_video_comments(API_KEY, video_id)
        if not comments.empty:
            comments_data.append(comments)
    
    if not comments_data:
        st.warning("No comments data found.")
        return
    
    df_comments = pd.concat(comments_data, ignore_index=True)
    
    # Perform sentiment analysis on descriptions
    df = perform_sentiment_analysis(df)
    
    # Perform sentiment analysis on comments
    df_comments['comment_polarity'] = df_comments['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df_comments['comment_sentiment'] = df_comments['comment_polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    
    # Calculate overall sentiment score
    df['overall_sentiment_score'] = (df['description_polarity'] + df_comments.groupby('videoId')['comment_polarity'].mean().reindex(df['videoId']).fillna(0)) / 2
    df['overall_sentiment'] = df['overall_sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

    st.write("### Sample of YouTube Video Data:")
    st.write(df.head())

    # Data Tables
    st.header('Data Tables')

    st.subheader('Filtered Video Data')
    st.write(df[['title', 'views', 'likes', 'dislikes', 'description_polarity', 'overall_sentiment_score', 'overall_sentiment']])

    st.subheader('Filtered Comment Data')
    st.write(df_comments[['videoId', 'author', 'text', 'comment_polarity', 'comment_sentiment']])

    # Video Statistics Summary
    st.subheader('Video Statistics Summary')
    st.write(df[['views', 'likes', 'dislikes', 'comments']].describe())

    # Sentiment Distribution
    st.subheader('Overall Sentiment Distribution')
    fig1, ax1 = plt.subplots()
    df['overall_sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62', '#8da0cb'], ax=ax1)
    ax1.set_ylabel('')
    ax1.set_title('Distribution of Overall Sentiment')
    st.pyplot(fig1)

    # Like-Dislike Ratio Distribution
    st.subheader('Like-Dislike Ratio Distribution')
    df['like_dislike_ratio'] = df['likes'] / df['dislikes'].replace({0: 1})
    fig2, ax2 = plt.subplots()
    sns.kdeplot(data=df, x='like_dislike_ratio', ax=ax2, fill=True, color='skyblue')
    ax2.set_title('Distribution of Like-Dislike Ratio')
    st.pyplot(fig2)

    # Comment Polarity Distribution
    st.subheader('Comment Polarity Distribution')
    fig3, ax3 = plt.subplots()
    sns.violinplot(data=df_comments, x='comment_polarity', ax=ax3, inner='quartile', palette='muted')
    ax3.set_title('Distribution of Comment Polarity')
    st.pyplot(fig3)

    # Views vs. Likes Colored by Sentiment
    st.subheader('Views vs. Likes')
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df, x='views', y='likes', hue='overall_sentiment', palette='viridis', ax=ax4)
    ax4.set_title('Views vs. Likes Colored by Sentiment')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    st.pyplot(fig4)

    # Word Cloud of Comments
    st.subheader('Word Cloud of Comments')
    comment_words = ' '.join(df_comments['text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comment_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

if __name__ == "__main__":
    main()
