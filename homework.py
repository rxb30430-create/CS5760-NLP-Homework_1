
"""
Homework 1 — CS5760 Natural Language Processing
Author: Rakesh Boragani 
student id 700763043
Instructions: Install spaCy and the small English model before running parts that use spaCy:
    pip install spacy
    python -m spacy download en_core_web_sm

"""

import re
from collections import Counter, defaultdict
import math

# -----------------------------
# Q1: Regex solutions + tests
# -----------------------------
def q1_regex_tests():
    print("Q1: Regex examples and tests\n" + "-"*40)

    # 1. U.S. ZIP codes (whole token)
    zip_re = re.compile(r"\b\d{5}(?:[- ]\d{4})?\b")
    tests = ["12345", "12345-6789", "12345 6789", "a12345", "123456", "01234-5678"]
    print("ZIP regex:", zip_re.pattern)
    for t in tests:
        print(t, "->", bool(zip_re.search(t)))

    # 2. Words that do NOT start with a capital letter. Allow internal apostrophes/hyphens.
    # Using ASCII letters; adjust to Unicode if needed.
    not_cap_re = re.compile(r"\b(?![A-Z])[A-Za-z][A-Za-z'-]*\b")
    tests2 = ["apple", "don't", "state-of-the-art", "Hello", "eMail", "o'neill", "3cats"]
    print("\nNot-starting-with-cap regex:", not_cap_re.pattern)
    for t in tests2:
        print(t, "->", bool(not_cap_re.search(t)))

    # 3. Numbers: optional sign, optional thousands separators, optional decimal, optional scientific notation
    number_re = re.compile(r"[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
    tests3 = ["123", "+1,234.56", "-0.12", "1.23e-4", "1234", "12,34", "1,234,567.89e+03"]
    print("\nNumber regex:", number_re.pattern)
    for t in tests3:
        print(t, "->", bool(number_re.fullmatch(t)))

    # 4. Spelling variants of "email" (space or hyphen, include en-dash)
    email_re = re.compile(r"(?i)\be[\-\s\u2013]?mail\b")  # \u2013 is en-dash
    tests4 = ["email", "e-mail", "e mail", "E–mail", "EMAIL", "e--mail"]
    print("\nEmail variants regex:", email_re.pattern)
    for t in tests4:
        print(t, "->", bool(email_re.search(t)))

    # 5. go/goo/gooo... as a word with optional trailing punctuation ! . , ?
    go_re = re.compile(r"\bgo+\b[!.,?]?")
    tests5 = ["go", "goo!", "gooo?", "going", "goooo.", "go," ]
    print("\nGo interjection regex:", go_re.pattern)
    for t in tests5:
        print(t, "->", bool(go_re.search(t)))

    # 6. Lines that end with a question mark possibly followed only by closing quotes/brackets and spaces
    endq_re = re.compile(r"\?\s*['\"\)\]\»]*\s*$")
    lines = [
        'Is this correct?',
        'Really?")',
        "Who? ]",
        "No? something",
        "Question? '",
        "End? »  "
    ]
    print("\nEnd-of-line-question regex:", endq_re.pattern)
    for ln in lines:
        print(repr(ln), "->", bool(endq_re.search(ln)))


# ---------------------------------------
# Q2: Tokenization (naive, manual, spaCy)
# ---------------------------------------
def q2_tokenization_demo(paragraph=None):
    print("\nQ2: Tokenization demo\n" + "-"*40)
    if paragraph is None:
        paragraph = ("The city of Paris is beautiful. However, it’s crowded during summer. "
                     "Don’t forget to visit the Eiffel Tower!")

    print("Paragraph:\n", paragraph, "\n")

    # 1. Naive space-based tokenization
    naive_tokens = paragraph.split()
    print("Naive space tokens:\n", naive_tokens, "\n")

    # 2. Manual corrections (simple demonstration):
    # We'll split punctuation and separate clitics (English-focused). A more complete rule set is larger.
    manual_tokens = []
    clitic_pattern = re.compile(r"(?i)(\w+)(n't|'s|'re|'ve|'ll|'d|’s|’t)$")
    for tok in naive_tokens:
        # split trailing punctuation
        m = re.match(r"^(.+?)([.!,?;:]+)$", tok)
        if m:
            core, punct = m.groups()
        else:
            core, punct = tok, ""
        # handle clitics: e.g., Don't -> Do + n't  (note: simple heuristic)
        mc = clitic_pattern.match(core)
        if mc:
            base, clitic = mc.groups()
            manual_tokens.append(base)
            manual_tokens.append(clitic)
        else:
            manual_tokens.append(core)
        if punct:
            manual_tokens.append(punct)
    print("Manual-corrected tokens:\n", manual_tokens, "\n")

    # 3. spaCy tokenization (if available)
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(paragraph)
        spacy_tokens = [t.text for t in doc]
        print("spaCy tokens:\n", spacy_tokens, "\n")
    except Exception as e:
        print("spaCy not available or model not installed. To compare, install spaCy and run:\n"
              "pip install spacy\npython -m spacy download en_core_web_sm\nThen re-run this script.\n")
        spacy_tokens = None

    # 4. Compute differences (if spaCy tokens present)
    if spacy_tokens is not None:
        print("Tokens differing between manual and spaCy (positionally):")
        diffs = []
        # We'll compare sets and also list tokens that are not in both
        set_manual = set(manual_tokens)
        set_spacy = set(spacy_tokens)
        print("Manual-only tokens:", sorted(set_manual - set_spacy))
        print("spaCy-only tokens:", sorted(set_spacy - set_manual))
    print("\nMultiword expressions (examples and why to keep them as single tokens):")
    mwes = [
        ("New York City", "Place name; splitting loses identity"),
        ("kick the bucket", "Idiom; meaning not compositional"),
        ("prime minister", "Title; treated as single semantic unit")
    ]
    for m, reason in mwes:
        print(f"- {m}: {reason}")

    print("\nReflection (example 5 sentences):")
    reflection = (
        "Tokenization is tricky because punctuation, clitics (like n't, 's), and multiword expressions "
        "interact. English contractions need special handling to separate the clitic for tasks like POS tagging. "
        "MWEs should be single tokens when their meaning is fixed (e.g., place names). "
        "Compared to some languages (agglutinative ones), English has moderate morphological complexity. "
        "Overall, punctuation and MWEs complicate tokenization and require language-specific rules."
    )
    print(reflection)

# -----------------------------------------------
# Q3: Manual BPE (toy corpus) + mini-BPE implementation
# -----------------------------------------------
def q3_manual_bpe():
    print("\nQ3: Manual BPE (toy corpus) and mini-BPE learner\n" + "-"*40)
    corpus = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
    words = corpus.split()

    # Add end-of-word marker '_'
    words_eow = [w + "_" for w in words]
    print("Corpus words with end-of-word marker:\n", words_eow[:20], "...")
    # Initial vocabulary = set of characters + '_'
    vocab = set()
    for w in words_eow:
        for ch in list(w):
            vocab.add(ch)
    print("Initial vocabulary (chars):", sorted(vocab))

    # Function to compute bigram counts over current tokenized words
    def get_pair_counts(tokenized_words):
        pairs = Counter()
        for tw in tokenized_words:
            for i in range(len(tw)-1):
                pairs[(tw[i], tw[i+1])] += 1
        return pairs

    # Initialize tokenized words as list of lists (chars)
    tokenized = [list(w) for w in words_eow]

    # We'll do 3 manual merge steps demonstrating the process
    for step in range(1, 4):
        pairs = get_pair_counts(tokenized)
        most_common, freq = pairs.most_common(1)[0]
        print(f"\nStep {step}: most frequent pair = {most_common} (freq={freq})")
        a, b = most_common
        new_symbol = a + b  # concatenation representing merged symbol
        # Apply merge
        new_tokenized = []
        for tw in tokenized:
            i = 0
            new_tw = []
            while i < len(tw):
                if i < len(tw)-1 and tw[i] == a and tw[i+1] == b:
                    new_tw.append(new_symbol)
                    i += 2
                else:
                    new_tw.append(tw[i])
                    i += 1
            new_tokenized.append(new_tw)
        tokenized = new_tokenized
        vocab.add(new_symbol)
        print("New token added to vocabulary:", new_symbol)
        # show snippets (first two lines)
        print("Updated corpus snippet (first 6 token lists):")
        for t in tokenized[:6]:
            print(t)

    # 3.2 Mini-BPE learner (coded)
    print("\nMini-BPE learner (automated):\n")
    # We'll implement a simple BPE learner function
    def learn_bpe(words_list, num_merges=10):
        tokenized = [list(w + "_") for w in words_list]
        merges = []
        vocab = set(ch for w in tokenized for ch in w)
        for it in range(num_merges):
            pairs = get_pair_counts(tokenized)
            if not pairs:
                break
            (a,b), freq = pairs.most_common(1)[0]
            merges.append(((a,b), freq))
            new_symbol = a + b
            # apply merge
            new_tokenized = []
            for tw in tokenized:
                i = 0
                new_tw = []
                while i < len(tw):
                    if i < len(tw)-1 and tw[i] == a and tw[i+1] == b:
                        new_tw.append(new_symbol)
                        i += 2
                    else:
                        new_tw.append(tw[i])
                        i += 1
                new_tokenized.append(new_tw)
            tokenized = new_tokenized
            vocab.add(new_symbol)
            print(f"Merge {it+1}: pair={(a,b)} freq={freq} -> new symbol='{new_symbol}' vocab_size={len(vocab)}")
        return merges, tokenized, vocab

    merges, tokenized_final, final_vocab = learn_bpe(words, num_merges=10)

    # Segment words: new, newer, lowest, widest, newestest (invented)
    def segment_word(word, merges):
        # greedy longest-match segmentation from merges (simulate BPE apply order)
        tokens = list(word + "_")
        merge_symbols = [a+b for ((a,b),_) in merges]
        # Apply merges in order
        for ((a,b),_) in merges:
            new_symbol = a + b
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(new_symbol)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    sample_words = ["new", "newer", "lowest", "wider", "newestest"]
    print("\nSegmentation of sample words after merges:")
    for sw in sample_words:
        seg = segment_word(sw, merges)
        print(f"{sw} -> {seg}")

    print("\nMini-BPE reflection (5-6 sentences):")
    reflection = (
        "Subword tokens help with OOV by allowing unknown words to be represented as sequences of known subwords. "
        "For example, an unseen derived form can be split into a stem and a suffix learned from other words. "
        "This reduces the number of completely unknown tokens the model must handle. "
        "Often suffixes like 'er' or 'est' (plus end-of-word marker) appear as meaningful subwords. "
        "However, BPE can also produce subwords that do not align with linguistic morphemes, which can be a downside."
    )
    print(reflection)

    # 3.3 Train BPE on a short paragraph (we'll reuse paragraph from Q2)
    paragraph = ("The city of Paris is beautiful. However, it’s crowded during summer. "
                 "Don’t forget to visit the Eiffel Tower!")
    para_words = re.findall(r"\b\w+\b", paragraph.lower())  # simple word list
    print("\nTraining BPE on a small paragraph (lowercased tokens):", para_words)
    merges_para, tokenized_para, vocab_para = learn_bpe(para_words, num_merges=30)
    # Show top 5 merges
    print("\nTop 5 merges learned on paragraph:")
    for i, ((a,b), freq) in enumerate(merges_para[:5], start=1):
        print(f"{i}. {(a,b)} freq={freq}")
    # Show five longest subword tokens (by length) from vocabulary
    sorted_vocab = sorted(vocab_para, key=lambda x: -len(x))
    print("\nFive longest subword tokens learned:", sorted_vocab[:5])
    # Segment 5 different words (including a rare and a derived form)
    words_to_segment = ["paris", "beautiful", "crowded", "eiffel", "forget"]
    print("\nSegmentations (paragraph BPE):")
    for w in words_to_segment:
        seg = segment_word(w, merges_para)
        print(f"{w} -> {seg}")

    print("\nReflection on paragraph BPE (5-8 sentences):")
    reflection_para = (
        "On a small text, BPE often learns frequent character sequences like common prefixes or suffixes. "
        "Subwords included stems (e.g., 'beaut' in 'beautiful'), suffix-like endings, and sometimes whole short words. "
        "Pros: reduces OOV and can capture morphological regularities. "
        "Cons: on very small data, merges can be noisy and overfit to specific words; linguistic morphemes are not guaranteed. "
        "For English, BPE handles inflectional morphology reasonably well; for highly agglutinative languages additional considerations apply."
    )
    print(reflection_para)


# -------------------------------------------
# Q4: Edit distance (Sunday -> Saturday)
# -------------------------------------------
def q4_edit_distance():
    print("\nQ4: Edit distance between 'Sunday' and 'Saturday'\n" + "-"*40)
    s1 = "Sunday"
    s2 = "Saturday"

    def levenshtein(a, b, sub_cost=1, ins_cost=1, del_cost=1):
        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            dp[i][0] = i*del_cost
        for j in range(1, m+1):
            dp[0][j] = j*ins_cost
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost_sub = 0 if a[i-1] == b[j-1] else sub_cost
                dp[i][j] = min(dp[i-1][j] + del_cost,
                               dp[i][j-1] + ins_cost,
                               dp[i-1][j-1] + cost_sub)
        return dp

    # Model A: sub=1, ins=1, del=1
    dpA = levenshtein(s1, s2, sub_cost=1, ins_cost=1, del_cost=1)
    distA = dpA[len(s1)][len(s2)]
    # Backtrace one alignment (from bottom-right)
    def backtrace(dp, a, b, sub_cost, ins_cost, del_cost):
        i, j = len(a), len(b)
        ops = []
        while i>0 or j>0:
            if i>0 and j>0 and dp[i][j] == dp[i-1][j-1] + (0 if a[i-1]==b[j-1] else sub_cost):
                if a[i-1] == b[j-1]:
                    ops.append(("match", a[i-1], b[j-1]))
                else:
                    ops.append(("sub", a[i-1], b[j-1]))
                i -= 1; j -= 1
            elif i>0 and dp[i][j] == dp[i-1][j] + del_cost:
                ops.append(("del", a[i-1], "-"))
                i -= 1
            else:
                ops.append(("ins", "-", b[j-1]))
                j -= 1
        return list(reversed(ops))

    opsA = backtrace(dpA, s1, s2, 1,1,1)
    print("Model A (sub=1,ins=1,del=1) distance:", distA)
    print("One edit sequence (A):")
    for op in opsA:
        print(op)

    # Model B: sub=2, ins=1, del=1
    dpB = levenshtein(s1, s2, sub_cost=2, ins_cost=1, del_cost=1)
    distB = dpB[len(s1)][len(s2)]
    opsB = backtrace(dpB, s1, s2, 2,1,1)
    print("\nModel B (sub=2,ins=1,del=1) distance:", distB)
    print("One edit sequence (B):")
    for op in opsB:
        print(op)

    print("\nReflection (4-5 sentences):")
    refl = (
        "The two models can give different distances because substitutions are penalized more in Model B. "
        "For Sunday -> Saturday, inserting characters and deleting can become preferable to substitutions under Model B. "
        "Insertions/deletions are more useful when substitution cost is high. "
        "In applications: spell-checkers often prefer lower substitution cost because typos are usually single-character substitutions; "
        "DNA alignment may set substitution costs differently based on biological mutation rates and may use affine gap penalties."
    )
    print(refl)


# -----------------------------
# main execution
# -----------------------------
if __name__ == "__main__":
    q1_regex_tests()
    q2_tokenization_demo()
    q3_manual_bpe()
    q4_edit_distance()
