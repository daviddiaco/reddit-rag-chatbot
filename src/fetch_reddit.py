import praw
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def fetch_posts(subreddit="CryptoMoonShots", limit=50):
    posts = []
    for post in reddit.subreddit(subreddit).new(limit=limit):
        posts.append({
            "title": post.title,
            "selftext": post.selftext,
            "url": post.url,
            "score": post.score  # this is the upvote count (net upvotes - downvotes)
        })
    return posts



def save_posts_as_txt(posts, folder="reddit-posts"):
    os.makedirs(folder, exist_ok=True)
    for i, post in enumerate(posts):
        filename = f"{folder}/post_{i+1:03}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Upvotes: {post['score']}\n")
            f.write(post["title"] + "\n\n")
            f.write(post["selftext"] + "\n\n")
            f.write(post["url"])

if __name__ == "__main__":
    posts = fetch_posts()
    save_posts_as_txt(posts)
    print(f"Saved {len(posts)} posts as .txt files.")
