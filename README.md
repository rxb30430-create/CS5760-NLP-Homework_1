# CS5760-NLP-Homework_1
Homework 1- Natural Language processing(Regex, Tokenization, BEP, Edit distance)

Author: Rakesh Boragani
Student ID :700763043
CRN:13312

## Files
- `homework.py` — single Python script implementing Q1–Q4, with comments and example output.
- `README.md` — this file.

## Setup
Install spaCy and the English model (if you want the spaCy tokenization comparison in Q2):

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Then run the script:

1. *Regular Expressions (Regex):* 
   - Matching ZIP codes, numbers, emails, and custom patterns.
   - Testing regex patterns on sample inputs.

2. *Tokenization:*
   - Naive (space-based) tokenization.
   - Manual tokenization handling punctuation and clitics.
   - Tokenization using *spaCy*.
   - Comparison of manual vs spaCy tokenization.
   - Discussion on multiword expressions and tokenization challenges.

3. *Byte Pair Encoding (BPE):*
   - Manual BPE on a toy corpus.
   - Implementation of a mini BPE learner.
   - Training BPE on a small paragraph and segmenting sample words.
   - Reflection on BPE advantages and limitations.

4. *Edit Distance:*
   - Levenshtein distance computation between "Sunday" and "Saturday".
   - Alignment operations (substitution, insertion, deletion).
   - Discussion on different cost models and their impact.


```bash
python3 homework.py
```



