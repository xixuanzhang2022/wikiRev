import requests
import re
import spacy
from bs4 import BeautifulSoup
from wikitextparser import parse

nlp = spacy.load("en_core_web_sm")

def is_valid_sentence(text):
    """
    Returns a list of valid sentences from the input text.
    A valid sentence:
    - Starts with a title-cased word
    - Ends with punctuation
    - Has at least one noun and one verb
    """
    doc = nlp(text)
    valid = []
    for sent in doc.sents:
        if sent[0].is_title and sent[-1].is_punct:
            noun_ok = any(tok.pos_ in ["NOUN", "PROPN", "PRON"] for tok in sent)
            verb_ok = any(tok.pos_ == "VERB" for tok in sent)
            if noun_ok and verb_ok:
                valid.append(str(sent))
    return valid

def call_wikipedia_api(revid, parentid):
    """
    Calls Wikipedia API to get HTML diff between two revisions.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': "compare",
        'format': "json",
        'fromrev': parentid,
        'torev': revid
    }
    r = requests.get(url, params=params)
    return r.json().get('compare', {}).get('*')

def eachrev(revid, parentid):
    """
    Parses the revision diff to extract sentences with link edits and citation edits.
    Returns:
        ssum (int): Number of link-containing sentences
        sl (list): Sentences with link edits
        slinklist (list): Target articles
        slinklist2 (list): Link display texts
        alllinks (list): All link targets
        allscores (list): Score per link (1 / len(linklist))
        refcl (list): Changed references
        refdl (list): Deleted references
    """
    text = call_wikipedia_api(revid, parentid)
    if not text:
        return 0, [], [], [], [], [], [], []

    soup = BeautifulSoup(text, "lxml")
    deletions = soup.find_all("del", class_="diffchange diffchange-inline")
    parents = list(set(s.parent for s in deletions))

    ssum, alllinks, allscores = 0, [], []
    sl, slinklist, slinklist2 = [], [], []
    refcl, refdl = [], []
    refcdict, refddict = {}, {}
    count_refc = count_refd = 0

    for parent in parents:
        text_html = str(parent)
        raw_text = parent.text

        # Extract links
        links = re.findall(r"\[\[(.*?)\]\]", raw_text)
        flat_links = [l for l in links if "|" not in l or l.count("|") == 1]
        flat_links = [l for l in flat_links if not any(k in l.lower() for k in ["file:", "image:"])]
        labels = [l.split("|", 1)[1] if "|" in l else l for l in flat_links]
        targets = [l.split("|", 1)[0] if "|" in l else l for l in flat_links]
        link_dict = dict(zip(labels, targets))

        changed = text_html.replace('<del class="diffchange diffchange-inline">', "-(d(-").replace("</del>", "-)d)-")
        for r in links:
            if any(k in r.lower() for k in ["file:", "image:"]):
                changed = changed.replace(f"[[{r}]]", "")

        parsed_text = BeautifulSoup(changed, "lxml").text
        refs = parse(parsed_text).get_tags()
        ref_strings = [r.string for r in refs]

        # Detect changed citations
        for ref in [r for r in ref_strings if "-(d(-" in r or "-)d)-" in r]:
            safe = ref.replace("<", "&lt;").replace(">", "&gt;")
            marker = f"-((refc{count_refc}refc))-"
            changed = changed.replace(safe, marker)
            refcdict[marker] = ref.replace("-(d(-", "((").replace("-)d)-", "))")
            count_refc += 1

        # Detect fully deleted citations
        for d in re.findall(r'<del class=\"diffchange diffchange-inline\">(.*?)</del>', text_html):
            if "&lt;" in d and "&gt;" in d and "ref" in d:
                for r in ref_strings:
                    if r in d:
                        marker = f"-((refdel{count_refd}refdel))-"
                        changed = changed.replace(r, marker)
                        refddict[marker] = r
                        count_refd += 1

        # Clean markup to prepare for sentence parsing
        changed = re.sub(r'(\.|[?!])((&lt;.*?&gt;)+(\{\{.*?\}\}))(\s)', r'\2\1 ', changed)
        changed = re.sub(r'(\.|[?!])((&lt;.*?&gt;)+)(\s)', r'\2\1 ', changed)
        changed = changed.replace("-(d(-", "((").replace("-)d)-", "))")

        try:
            sentences = is_valid_sentence(parse(changed).plain_text())
        except:
            print("error: no sentence")
            return 0, [], [], [], [], [], [], []

        for sent in sentences:
            if "((" in sent or "))" in sent:
                clean = sent.replace("((", "").replace("))", "")
                in_links = [link_dict[l] for l in labels if l in clean]
                if in_links:
                    score = [1 / len(in_links)] * len(in_links)
                    ssum += 1
                    sl.append(sent)
                    slinklist.append(" | ".join(in_links))
                    slinklist2.append(" | ".join([l for l in labels if l in clean]))
                    alllinks += in_links
                    allscores += score
                    refcl.append(" | ".join([refcdict[r] for r in re.findall(r'\-\(\(refc\d+refc\)\)\-', sent)]))
                    refdl.append(" | ".join([refddict[r] for r in re.findall(r'\-\(\(refdel\d+refdel\)\)\-', sent)]))

    return ssum, sl, slinklist, slinklist2, alllinks, allscores, refcl, refdl


if __name__ == "__main__":
    output = eachrev("794386526", "852661894")
    print(output)
