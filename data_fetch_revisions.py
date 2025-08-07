import os
import pandas as pd
import pywikibot

def fetch_revisions(title: str, is_talk: bool = False):
    """Fetch revision metadata for a Wikipedia page (or its talk page)."""
    prefix = "Talk:" if is_talk else ""
    page_title = prefix + title
    site = pywikibot.Site("en", "wikipedia")
    page = pywikibot.Page(site, page_title)

    try:
        revs = list(page.revisions(content=False))
        return pd.DataFrame(revs)
    except Exception as e:
        print(f"Failed to fetch {page_title}: {e}")
        return pd.DataFrame()

def save_revisions(df: pd.DataFrame, title: str, output_dir: str):
    """Save revision DataFrame to CSV."""
    filename = title.replace(" ", "_").lower() + ".csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False, encoding='utf-8')

def ensure_dir(path: str):
    """Create output directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def main():
    page_list_path = "/Users/xixuan/Desktop/S2/wikiproject0712.csv"
    articles = pd.read_csv(page_list_path, encoding="utf-8")["title"].tolist()
    out_dir_article = "/Users/xixuan/Desktop/S2/csvs"
    out_dir_talk = "/Users/xixuan/Desktop/S2/csvs_talk"
    ensure_dir(out_dir_article)
    ensure_dir(out_dir_talk)

    for idx, title in enumerate(articles):
        for is_talk, out_dir in [(False, out_dir_article), (True, out_dir_talk)]:
            df = fetch_revisions(title, is_talk=is_talk)
            if not df.empty:
                save_revisions(df, ("Talk:" if is_talk else "") + title, out_dir)
        print(f"[{idx + 1}/{len(articles)}] Processed: {title}")

if __name__ == "__main__":
    main()
