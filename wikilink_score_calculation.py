import pandas as pd
import glob
import os

REV_PATH = "/Users/xixuanzhang/Documents/S2/s2_fifi/csvsrev_all_alsobots/"
USER_PATH = "/Users/xixuanzhang/Documents/S2/final_s2/csvs/"
OUT_PATH = "/Users/xixuanzhang/Documents/S2/s2_fifi/scoreREV/"
LINKLIST_OUT = "/Users/xixuanzhang/Documents/S2/s2_fifi/linklist_diffREVuser.csv"


# === Step 1: Scoring function ===
def get_link_scores(dfx: pd.DataFrame, pagename: str):
    """Calculate link scores for one page and save to CSV."""
    links, users, revids, linkscore = [], [], [], []

    for i in range(len(dfx)):
        link_list = str(dfx["links"][i]).split("| ")
        freq = len(link_list)
        links += link_list
        linkscore += [1 / freq] * freq
        users += [dfx["user"][i]] * freq
        revids += [dfx["revid"][i]] * freq

    df_links = pd.DataFrame({
        "revid": revids,
        "user": users,
        "links": links,
        "linkscore": linkscore
    })

    unique_links = set(links)
    scores = []

    for item in unique_links:
        test = df_links[df_links["links"] == item].sort_values(by="revid", ascending=True).reset_index(drop=True)
        test["user_shift"] = test["user"].shift()
        df_u = test[test["user"] != test["user_shift"]].copy()
        df_u["filtered_score"] = df_u.groupby("user")["linkscore"].cumsum()

        if len(df_u) > 1:
            pair_mins = [
                min(df_u["filtered_score"].iloc[i], df_u["filtered_score"].iloc[i + 1])
                for i in range(len(df_u) - 1)
            ]
            score = len(set(df_u["user"])) * (sum(pair_mins) - max(pair_mins))
        else:
            score = 0

        scores.append((item, score))

    pd.DataFrame(scores, columns=["link", "score"]).to_csv(os.path.join(OUT_PATH, f"s_{pagename}"), index=False)


# === Step 2: Apply to all revision pages ===
rev_files = [f for f in glob.glob(f"{REV_PATH}*") if not os.path.basename(f).startswith("s_")]

for file in rev_files:
    print(f"Processing {file}...")
    dfx = pd.read_csv(file, encoding="utf-8")
    user_file = file.replace("s2_fifi/csvsrev_all_alsobots", "final_s2/csvs")
    user_data = pd.read_csv(user_file, encoding="utf-8")
    user_map = user_data.groupby("revid")["user"].apply(list).to_dict()
    dfx["user"] = dfx["revid"].apply(lambda r: user_map.get(int(r), ["nosuchuser"])[0])
    get_link_scores(dfx, os.path.basename(file))


# === Step 3: Aggregate link scores for each linked article ===
score_files = glob.glob(os.path.join(OUT_PATH, "*"))
df_all = pd.concat([pd.read_csv(f) for f in score_files], ignore_index=True)

df_all["link"] = df_all["link"].astype(str).str.lower().str.strip()
df_all["link"] = df_all["link"].str.lstrip("#").str.lstrip(" ").str.replace(";", " ", regex=False)

linklist = df_all.groupby("link", as_index=False)["score"].sum()
linklist.columns = ["Id", "score"]

linklist.to_csv(LINKLIST_OUT, index=False)
print(f"[DONE] linklist saved to: {LINKLIST_OUT}")
