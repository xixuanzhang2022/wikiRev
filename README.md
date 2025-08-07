A toolkit for mining, parsing, and modeling revision data from Wikipedia and other wiki platforms, using both Python and R.
The code in this repository was originally developed to support the data collection and analyses presented in the paper:
**Zhang, X. (2025). Decoding revision mechanisms in Wikipedia: Collaboration, moderation, and collectivities. *New Media & Society*. [https://doi.org/10.1177/14614448251336418](https://doi.org/10.1177/14614448251336418)**

**Revision Data Collection**

data_fetch_revisions.py: Downloads and structures page revision histories.

**Sentence-Level Diff Parsing**

data_sentencediff_parser.py: Extracts sentence-level differences between revisions.

**Sequence Modeling**

sequence_modeling.py: Models edit progression using structured sequence data.

**Content Embedding (BERT-based)**

content_finetune_bert.py: Fine-tunes BERT on revision content.

content_apply_bert.py: Applies the fine-tuned model for downstream tasks.

**Statistical & Meta Analysis (R)**

frailty_model.R and meta_analysis.R: Perform survival and meta-analysis on revision-based features.

**User and Link Insights**

user_wiki_rights.py: Analyzes user rights and editorial roles.

wikilink_score_calculation.py: Scores internal wiki link structures.

