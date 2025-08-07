import pandas as pd
import numpy as np
import glob
import re
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
filepath = "/Users/xixuanzhang/Documents/S2/s2_fifi/csvsrev_all_alsobots/"
filepath2 = "/Users/xixuanzhang/Documents/S2/final_s2/csvs/"

# Patterns for cleaning
TAG_RE = re.compile(r'<(.*?)>')
TAG_refc = re.compile(r"-\(\(refc\d+refc\)\)-")
TAG_refd = re.compile(r"-\(\(refdel\d+refdel\)\)-")

def remove_tags(text):
    return re.sub(TAG_RE, '', text)

def remove_refc(text):
    return re.sub(TAG_refc, '', text)

def remove_refd(text):
    return re.sub(TAG_refd, '', text)

def pagesim(path_article, filepath, filepath2, thres=0.7):
    group = pd.read_csv(path_article, encoding="utf-8")
    group["revid"] = group["revid"].astype(str)
    user_map = group.groupby("revid")["user"].apply(list).to_dict()

    # Find corresponding diff file
    path_diff = filepath + path_article[len(filepath2):]
    df = pd.read_csv(path_diff, encoding="utf-8")
    df = df[::-1].reset_index()
    df = df[['revid', 'parentid', 'sentence', 'links', 'linksintext', 'refchanged', 'refdeleted']]
    df["user"] = df["revid"].apply(lambda sub: user_map.get(str(sub), ["unknown"])[0])

    df["sentence1"] = df["sentence"].apply(
        lambda sub: remove_refd(remove_refc(remove_tags(str(sub)))).replace("((", "").replace("))", "").replace("-", "").strip()
    )

    try:
        tfidf_matrix = TfidfVectorizer().fit_transform(df["sentence1"])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        np.fill_diagonal(cosine_sim, 0)

        source, target = [], []
        for k in range(len(df) - 1):
            arr = cosine_sim[k:, k:]
            row = arr[0, :]
            arrparent = np.array(df["revid"].tolist()[k:])
            indp = np.where(arrparent == df["parentid"][k])[0]

            if len(indp) > 0 and row[indp].max() > thres:
                best = np.where(row == row[indp].max())[0][0] + k
                source.append(k)
                target.append(best)
            elif row.max() > thres:
                best = np.where(row > thres)[0][0] + k
                source.append(k)
                target.append(best)
    except:
        return [], [], df

    return source, target, df

def pageseq(source, target, df):
    source = [int(x) for x in source]
    target = [int(x) for x in target]
    rest = [[x] for x in sorted(set(df.index) - set(source + target))]
    dseq = dict(zip(source, target))
    alllist, had = [], []
    currentlist = source.copy()

    while currentlist:
        current = currentlist[0]
        partlist = []
        while current in dseq:
            partlist.append(current)
            currentlist.remove(current)
            had.append(current)
            current = dseq.pop(current)
        if current not in had:
            partlist.append(current)
            had.append(current)
        alllist.append(partlist)

    alllist.extend(rest)
    alllist.sort(key=lambda x: x[0])
    return alllist

def userlist(flist, alllist, df):
    indexi = range(len(df))
    duser = dict(zip(indexi, df['user'].tolist()))
    dfuser = dict(zip(flist, ["XXXzhzh"] * len(flist)))
    duser.update(dfuser)
    return [[duser.get(x) for x in l] for l in alllist]

def realuserlist(flist, alllist, df):
    duser = dict(zip(range(len(df)), df['user'].tolist()))
    return [[duser.get(x) for x in l] for l in alllist]

def foroneuser1(u1, seq, seqlist):
    userseq = [i for i, x in enumerate(seqlist) if x == u1]
    for i in range(len(userseq)):
        idx = userseq[-1 - i]
        if len(userseq) == 1 or i == len(userseq) - 1:
            seq[idx] = "E"
        else:
            gap = userseq[-1 - i] - userseq[-2 - i]
            if gap == 1:
                seq[idx] = "A"
            elif gap == 2:
                seq[idx] = "B"
            elif 3 <= gap <= 5:
                seq[idx] = "C"
            elif gap >= 6:
                seq[idx] = "D"

def alldetails(alllist, df, path, scoredict, pageuserlist, pageuserlistreal, activegroup):
    df["comment"] = ''
    dffull = pd.read_csv(path, encoding="utf-8").set_index('revid')
    df["comment"] = df["revid"].map(dffull["comment"])

    talklist, undidlist, refclist, linkslist, taskflist, seqlistl = [], [], [], [], [], []
    links, revids, sentences, idl = [], [], [], []

    for l, real_users, masked_users in zip(alllist, pageuserlistreal, pageuserlist):
        s1 = s2 = s = s3 = 0
        revs = [df["revid"][a] for a in l]
        sents = [df["sentence1"][a] for a in l]

        for e in l:
            comment = str(df['comment'][e])
            s1 += comment.count("talk")
            s2 += comment.lower().count("undid")
            if pd.notnull(df['refchanged'][e]):
                s += df['refchanged'][e].count("<ref")

            links_raw = str(df['links'][e]).split("| ")
            for link in links_raw:
                s3 += scoredict.get(link.lower(), 0)
            s3 = s3 / max(1, len(links_raw))

        links.append('| '.join(set(links_raw)))
        talklist.append(s1)
        undidlist.append(s2)
        refclist.append(s)
        linkslist.append(s3 / len(l))
        idl.append(revs)
        revids.append(revs[0])
        sentences.append(sents)

        active_count = sum(1 for e in real_users if e in activegroup)
        taskflist.append(active_count)

        seq = ["0"] * len(real_users)
        foroneuser_map = {u for u in set(real_users)}
        for u in foroneuser_map:
            foroneuser1(u, seq, real_users)

        for i in range(len(real_users)):
            if masked_users[i] == "XXXzhzh" or not isinstance(masked_users[i], str):
                seq[i] = "E"
            seq[i] += "1" if real_users[i] in activegroup else "0"

        seqlistl.append("-".join(seq))

    return talklist, undidlist, refclist, linkslist, taskflist, seqlistl, links, revids, sentences, idl

# Load external data
active = pd.read_csv("/Users/xixuanzhang/Documents/S2/final_s2/climatechange_u.csv")
active2 = pd.read_csv("/Users/xixuanzhang/Documents/S2/final_s2/wikiproject_u.csv")
active3 = pd.read_csv("/Users/xixuanzhang/Documents/S2/final_s2/activeinactive_u.csv")
activegroup = list(set(active["user"].tolist() + active2["user"].tolist() + active3["user"].tolist()))

nodelist = pd.read_csv('/Users/xixuanzhang/Documents/S2/s2_fifi/nodelist_diffREVuser.csv')
scoredict = dict(zip(nodelist['Id'].values, nodelist['score'].values))

# Process all articles
readlist = glob.glob("/Users/xixuanzhang/Documents/S2/final_s2/csvs/*.csv")

# Output containers
seqlistl, pagelist, firstrevidlist, sentencel = [], [], [], []
refclist, linkslist, taskflist, talklist, undidlist, idsl, links = [], [], [], [], [], [], []

for n, path in enumerate(readlist):
    print(n, path)
    source, target, df = pagesim(path, filepath, filepath2)

    try:
        alllist = pageseq(source, target, df)
        flist = df[['user']].drop_duplicates().reset_index()['index'].tolist()
        pageuserlist = userlist(flist, alllist, df)
        pageuserlistreal = realuserlist(flist, alllist, df)

        details = alldetails(alllist, df, path, scoredict, pageuserlist, pageuserlistreal, activegroup)
        t0, u0, r0, l0, tf0, sl0, lk0, fr0, st0, id0 = details

        talklist += t0
        undidlist += u0
        refclist += r0
        linkslist += l0
        taskflist += tf0
        seqlistl += sl0
        links += lk0
        firstrevidlist += fr0
        sentencel += st0
        pagelist += [path[len(filepath2):-4]] * len(sl0)
        idsl += id0
    except Exception as e:
        print("Failed:", path, "Error:", e)

# Final output
dfnew = pd.DataFrame({
    'page': pagelist,
    'firstrevid': firstrevidlist,
    'seq': seqlistl,
    'links': links,
    'score': linkslist,
    'taskforce': taskflist,
    'talk': talklist,
    'conf': undidlist,
    'refc': refclist
})

dfnew2 = pd.DataFrame({
    'page': pagelist,
    'firstrevid': firstrevidlist,
    'revid_seq': idsl,
    'sentence': sentencel
})

#dfnew.to_pickle("/Users/xixuanzhang/Documents/S2/s2_fifi/evo_sentence_new_fifiREVuser.pkl")
#dfnew2.to_pickle("/Users/xixuanzhang/Documents/S2/s2_fifi/evo_sentence_ids_fifiREVuser.pkl")
