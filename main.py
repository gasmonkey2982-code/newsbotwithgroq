import html
import os
import re
import time
import feedparser
import requests
from groq import Groq

# --- CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "qwen/qwen3.8-27b"

RSS_URLS = [
    ("National", "https://www.thehindu.com/news/national/feeder/default.rss"),
    ("National", "https://indianexpress.com/section/india/feed/"),
    ("International", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Tech", "https://inc42.com/feed/"),
    ("Business", "https://www.livemint.com/rss/news"),
    ("Sports", "https://www.espncricinfo.com/rss/content/story/feeds/0.xml"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NewsSummaryBot/1.0)"
}


def validate_config():
    """Verify karta hai ki zaroori environment variables set hain ya nahi."""
    missing = [
        name for name, val in [
            ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
            ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
            ("GROQ_API_KEY", GROQ_API_KEY),
        ]
        if not val
    ]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")


def clean_html(raw_html: str) -> str:
    """HTML tags aur entities (&amp;, &#39;, etc.) saaf karta hai."""
    text = re.sub(r"<.*?>", " ", raw_html or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_article_context(url: str, fallback_summary: str) -> str:
    """Article page se shuruati context nikalta hai."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
        resp.raise_for_status()
        paragraphs = re.findall(
            r"<p[^>]*>(.*?)</p>",
            resp.text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        clean_paras = []
        for p in paragraphs:
            txt = clean_html(p)
            if (
                len(txt) > 50
                and "cookie" not in txt.lower()
                and "subscribe" not in txt.lower()
            ):
                clean_paras.append(txt)
            if len(" ".join(clean_paras)) > 250:
                break

        if clean_paras:
            return " ".join(clean_paras)[:250]
    except Exception:
        pass

    return (fallback_summary or "")[:220]


def get_news():
    """RSS feeds aur article context se news pool banata hai."""
    items = []
    seen_titles = set()
    success_count = 0

    for category, url in RSS_URLS:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)

            if feed.entries:
                for entry in feed.entries[:2]:
                    title = clean_html(entry.get("title", ""))
                    if not title or title.lower() in seen_titles:
                        continue
                    seen_titles.add(title.lower())

                    rss_content = ""
                    if entry.get("content"):
                        rss_content = clean_html(entry.content[0].get("value", ""))
                    if not rss_content:
                        rss_content = clean_html(entry.get("summary", ""))

                    link = entry.get("link", "")
                    details = (
                        fetch_article_context(link, rss_content)
                        if link
                        else rss_content[:220]
                    )

                    if details:
                        items.append(
                            f"[{category}] HEADLINE: {title}\nDETAILS: {details}"
                        )
                    else:
                        items.append(f"[{category}] HEADLINE: {title}")

                success_count += 1
                print(f"Success ({category}): {url}")
            else:
                print(f"Empty feed ({category}): {url}")
        except Exception as e:
            print(f"Failed ({category}) {url}: {str(e)}")

    print(f"Total feeds successful: {success_count}")
    return "\n\n".join(items) if items else None


def summarize_with_groq(raw_news: str) -> str:
    """GroqCloud se 10 grouped Hinglish bullet points banata hai."""
    prompt = f"""
Yeh aaj ki taaza khabrein hain unki details ke saath:
{raw_news}

INSTRUCTIONS:
1. Summary SIRF Roman script (English letters) mein likho. Devanagari/Hindi script bilkul mat use karo.
2. Language HINGLISH honi chahiye (natural casual tone).
   Example: "PUC certificate check kar lo" ✅ | "Check your PUC certificate" ❌
3. Total 10 informative bullet points banao — niche diye gaye 5 sections mein group karke (har section ke neeche 2 bullets):

🇮🇳 National
• ...
• ...

🌍 International
• ...
• ...

💻 Tech
• ...
• ...

📈 Business
• ...
• ...

🏏 Sports
• ...
• ...

4. Sirf headline repeat mat karo — DETAILS padh kar 1.5 se 2 lines mein batao ki asli baat kya hai.
5. Har bullet ke aage dubara category ka naam mat likho. Extra intro/outro mat likho.
"""
    client = Groq(api_key=GROQ_API_KEY, timeout=30.0)
    wait_times = [3, 7]
    last_error = ""

    for attempt in range(len(wait_times) + 1):
        try:
            chat_completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Aap ek concise Hinglish news editor hain jo sirf Roman script mein factual bullet summaries likhte hain.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.35,
                max_tokens=900,
            )
            text = (chat_completion.choices[0].message.content or "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            if text:
                return text
            last_error = "Empty response from Groq"
        except Exception as e:
            err_str = str(e)
            last_error = err_str[:200]
            print(f"Groq Attempt {attempt + 1} Error: {last_error}")
            if "429" in err_str or "rate_limit" in err_str.lower():
                break
            if attempt < len(wait_times):
                sleep_sec = wait_times[attempt]
                print(f"{sleep_sec}s baad retry...")
                time.sleep(sleep_sec)

    return f"AI Error: {last_error}"


def send_to_telegram(message: str):
    """Telegram par safely message bhejta hai (4096 char limit handle karte hue)."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [message[i:i + 4000] for i in range(0, len(message), 4000)]

    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                print(f"Telegram Error: {resp.text}")
        except Exception as e:
            print(f"Telegram Request Failed: {str(e)}")


if __name__ == "__main__":
    validate_config()

    print("News fetch kar raha hu...")
    raw_news = get_news()
    if not raw_news:
        print("Koi news nahi mili!")
        exit(1)

    print("Groq se summary bana raha hu...")
    summary = summarize_with_groq(raw_news)

    if summary.startswith("AI Error:"):
        print(f"Summary nahi ban payi: {summary}")
        exit(1)

    final_message = f"📰 Aaj Ki 10 Taaza Khabrein\n\n{summary}"

    print("Telegram par bhej raha hu...")
    send_to_telegram(final_message)
    print("Ho gaya!")
