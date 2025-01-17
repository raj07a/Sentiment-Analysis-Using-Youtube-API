import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from urllib.parse import urlparse, parse_qs

# YouTube API Key
API_KEY = "AIzaSyDV7Wfx8L4GAe6Daxfzpk97x1RECLfZ2ho"

# Function to resolve channel ID from username-based URL (e.g., @SonySAB)
def resolve_channel_id_from_username(api_key, username):
    url = f"https://www.googleapis.com/youtube/v3/channels?part=id&forUsername={username.replace('@', '')}&key={api_key}"
    response = requests.get(url)
    response_json = response.json()

    if response.status_code == 200 and 'items' in response_json and len(response_json['items']) > 0:
        return response_json['items'][0]['id']
    else:
        st.error("Unable to resolve channel ID from username. Please check the link or try another channel.")
        return None

# Function to extract channel ID from a YouTube channel link
def extract_channel_id(channel_url):
    parsed_url = urlparse(channel_url)
    if "@" in parsed_url.path:  # Handle username-based URL
        username = parsed_url.path.split("/")[-1]
        return resolve_channel_id_from_username(API_KEY, username)
    elif "channel" in parsed_url.path:  # Handle channel-based URL
        return parsed_url.path.split("/")[-1]
    else:
        st.error("Invalid YouTube channel link. Please provide a valid link.")
        return None

# Function to fetch video data for a channel
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
                    'title': item['snippet']['title']
                }
                videos.append(video_info)

        page_token = response_json.get('nextPageToken', '')
        if not page_token:
            break

    return pd.DataFrame(videos)

# Function to fetch comments for a video
def fetch_video_comments(api_key, video_id):
    comments = []
    page_token = ''
    while True:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&videoId={video_id}&part=snippet&maxResults=100&pageToken={page_token}"
        response = requests.get(url)
        response_json = response.json()

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
def perform_sentiment_analysis(comments):
    comments['comment_polarity'] = comments['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    comments['comment_sentiment'] = comments['comment_polarity'].apply(
        lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
    )
    return comments

# Main function
def main():
    st.title('YouTube Channel Sentiment Analysis Dashboard')

    # Step 1: Input for YouTube channel link
    channel_url = st.text_input("Enter YouTube Channel Link (e.g., https://www.youtube.com/@SonySAB):", "")
    if channel_url:
        channel_id = extract_channel_id(channel_url)
        if channel_id:
            st.success(f"Channel ID resolved: {channel_id}")

            # Step 2: Fetch videos for the channel
            video_data = fetch_youtube_data(API_KEY, channel_id)

            if not video_data.empty:
                st.write("Available Videos:")
                st.write(video_data)

                # Step 3: Create a dropdown for video selection
                video_title_to_id = dict(zip(video_data['title'], video_data['videoId']))
                selected_video_title = st.selectbox("Select a Video for Analysis:", video_data['title'])
                selected_video_id = video_title_to_id[selected_video_title]

                st.success(f"Selected Video: {selected_video_title} (ID: {selected_video_id})")

                # Step 4: Fetch comments for the selected video
                comments = fetch_video_comments(API_KEY, selected_video_id)
                if not comments.empty:
                    st.write("Sample Comments:")
                    st.write(comments.head())

                    # Step 5: Perform sentiment analysis
                    comments = perform_sentiment_analysis(comments)

                    # Sentiment Distribution
                    st.subheader('Comment Sentiment Distribution')
                    fig, ax = plt.subplots()
                    comments['comment_sentiment'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title('Sentiment Distribution')
                    ax.set_xlabel('Sentiment')
                    ax.set_ylabel('Number of Comments')
                    st.pyplot(fig)

                    # Word Cloud
                    st.subheader('Word Cloud of Comments')
                    comment_words = ' '.join(comments['text'].tolist())
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comment_words)
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(wordcloud, interpolation='bilinear')
                    ax2.axis('off')
                    ax2.set_title('Word Cloud of Comments')
                    st.pyplot(fig2)

                    # Top Comment Authors
                    st.subheader('Top Comment Authors')
                    top_authors = comments['author'].value_counts().head(10)
                    fig3, ax3 = plt.subplots()
                    top_authors.plot(kind='bar', ax=ax3, color='orange')
                    ax3.set_title('Top Comment Authors')
                    ax3.set_xlabel('Author')
                    ax3.set_ylabel('Number of Comments')
                    st.pyplot(fig3)

                else:
                    st.warning("No comments found for the selected video.")
            else:
                st.warning("No videos found for this channel.")
        else:
            st.error("Failed to resolve channel ID. Please check the link.")
    else:
        st.info("Please enter a valid YouTube channel link.")

if __name__ == "__main__":
    main()
