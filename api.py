import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import datetime

# API key and default channel ID (Replace with your API Key and Channel ID)
API_KEY = "AIzaSyDV7Wfx8L4GAe6Daxfzpk97x1RECLfZ2ho"
CHANNEL_ID = "UCDDjMFHTsEerSEm2BvhcwrA"

# Function to fetch YouTube channel video data
def fetch_youtube_data(api_key, channel_id):
    videos = []
    page_token = ''
    while True:
        url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=50&pageToken={page_token}"
        response = requests.get(url)
        response_json = response.json()

        if response.status_code != 200:
            st.error(f"Error fetching YouTube data: {response_json}")
            break

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

# Function to fetch specific video data
def fetch_specific_video(api_key, video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?key={api_key}&id={video_id}&part=snippet,statistics"
    response = requests.get(url)
    response_json = response.json()

    if response.status_code != 200:
        st.error(f"Error fetching video data for ID {video_id}: {response_json}")
        return None

    if 'items' in response_json and response_json['items']:
        item = response_json['items'][0]
        video_info = {
            'videoId': video_id,
            'title': item['snippet']['title'],
            'description': item['snippet']['description'],
            'publishedAt': item['snippet']['publishedAt'],
            'views': int(item['statistics'].get('viewCount', 0)),
            'likes': int(item['statistics'].get('likeCount', 0)),
            'dislikes': int(item['statistics'].get('dislikeCount', 0)),
            'comments': int(item['statistics'].get('commentCount', 0))
        }
        return pd.DataFrame([video_info])
    else:
        st.warning(f"No data found for video ID: {video_id}")
        return None

# Function to fetch comments for a video
def fetch_video_comments(api_key, video_id):
    comments = []
    page_token = ''
    while True:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&videoId={video_id}&part=snippet&maxResults=100&pageToken={page_token}"
        response = requests.get(url)
        response_json = response.json()

        # Check for errors and skip videos with disabled comments
        if response.status_code != 200:
            error_reason = response_json.get('error', {}).get('errors', [{}])[0].get('reason', '')
            if error_reason == 'commentsDisabled':
                st.warning(f"Comments are disabled for video ID: {video_id}. Skipping...")
                break
            else:
                st.error(f"Error fetching comments for video ID {video_id}: {response_json}")
                break

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

    # Add a specific video ID
    specific_video_id = "xfUQL1ByVbI"
    st.write(f"Analyzing specific video ID: {specific_video_id}")

    # Fetch specific video data
    specific_video_data = fetch_specific_video(API_KEY, specific_video_id)

    if specific_video_data is not None:
        st.write("Specific Video Data:")
        st.write(specific_video_data)

        # Fetch comments for the specific video
        specific_comments = fetch_video_comments(API_KEY, specific_video_id)

        if not specific_comments.empty:
            st.write("Specific Video Comments:")
            st.write(specific_comments)

            # Perform sentiment analysis on the specific video comments
            specific_comments['comment_polarity'] = specific_comments['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            specific_comments['comment_sentiment'] = specific_comments['comment_polarity'].apply(
                lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
            )

            # Display comment sentiment distribution
            st.subheader('Specific Video Comment Sentiment Distribution')
            fig, ax = plt.subplots()
            specific_comments['comment_sentiment'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Sentiment Distribution for Specific Video')
            st.pyplot(fig)

            # Display word cloud of comments
            st.subheader('Word Cloud of Comments')
            comment_words = ' '.join(specific_comments['text'].tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comment_words)
            fig2, ax2 = plt.subplots()
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.axis('off')
            ax2.set_title('Word Cloud of Comments')
            st.pyplot(fig2)

            # **New Visualizations**
            # 1. Sentiment Over Time (Comments vs Time)
            st.subheader('Sentiment Over Time (Comments vs Time)')
            specific_comments['publishedAt'] = pd.to_datetime(specific_comments['publishedAt'])
            specific_comments['year_month'] = specific_comments['publishedAt'].dt.to_period('M')
            sentiment_time = specific_comments.groupby('year_month')['comment_polarity'].mean().reset_index()
            fig3, ax3 = plt.subplots()
            ax3.plot(sentiment_time['year_month'].astype(str), sentiment_time['comment_polarity'], marker='o', color='green')
            ax3.set_title('Average Sentiment Over Time')
            ax3.set_xlabel('Time (Year-Month)')
            ax3.set_ylabel('Average Sentiment')
            plt.xticks(rotation=45)
            st.pyplot(fig3)

            # 2. Top Comment Authors
            st.subheader('Top Comment Authors')
            top_authors = specific_comments['author'].value_counts().head(10)
            fig4, ax4 = plt.subplots()
            top_authors.plot(kind='bar', ax=ax4, color='orange')
            ax4.set_title('Top Comment Authors')
            ax4.set_xlabel('Authors')
            ax4.set_ylabel('Number of Comments')
            st.pyplot(fig4)

            # 3. Comments Distribution by Sentiment (Pie Chart)
            st.subheader('Comments Distribution by Sentiment')
            sentiment_counts = specific_comments['comment_sentiment'].value_counts()
            fig5, ax5 = plt.subplots()
            sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62', '#8da0cb'], ax=ax5)
            ax5.set_ylabel('')
            ax5.set_title('Sentiment Distribution for Comments')
            st.pyplot(fig5)

        else:
            st.warning(f"No comments found for video ID: {specific_video_id}")
    else:
        st.warning(f"Specific video data could not be fetched.")

if __name__ == "__main__":
    main()
