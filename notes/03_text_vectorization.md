# Text Vectorization

## One-Hot Encoding

Consider three sentences:

- s1 = "I am Spiderman"
- s2 = "I am not Spiderman"
- s3 = "We are Venom"

Tokenizing (and lowercasing) these sentences and collecting unique tokens in order of appearance gives the vocabulary:

`['i', 'am', 'spiderman', 'not', 'we', 'are', 'venom']` (v = 7)

Using this vocabulary, each sentence is represented as a binary vector of length 7 (1 indicates presence of the token):

- s1: [1, 1, 1, 0, 0, 0, 0] # `i`, `am`, `spiderman`
- s2: [1, 1, 0, 1, 0, 0, 0] # `i`, `am`, `not`
- s3: [0, 0, 0, 0, 1, 1, 1] # `we`, `are`, `venom`

Advantages and disadvantages:

| Advantage                    | Disadvantage                        |
| ---------------------------- | ----------------------------------- |
| Simple and easy to implement | Produces very sparse vectors        |
| Deterministic representation | High dimensionality -> overfitting  |
|                              | No semantic similarity preserved    |
|                              | Out-of-vocabulary (OOV) words issue |

## Bag of Words (BoW)

Example sentences:

- s1 = "He is a cool guy"
- s2 = "She is a cool girl"
- s3 = "The guy and girl are cool"

After removing stopwords (e.g., `he`, `is`, `a`, `she`, `the`, `and`), the remaining vocabulary is:

`['cool', 'guy', 'girl']`

Count-based BoW (vocabulary order: `['cool', 'guy', 'girl']`):

- s1: [1, 1, 0]
- s2: [1, 0, 1]
- s3: [1, 1, 1]

Binary BoW would use 0/1 instead of counts; normalized BoW can use term frequency.

Advantages and disadvantages:

| Advantage                        | Disadvantage                    |
| -------------------------------- | ------------------------------- |
| Simple and intuitive             | Still produces sparse vectors   |
| Fixed-size vectors for ML models | Loses word order / context      |
| Easy to compute and interpret    | No semantic similarity captured |
|                                  | OOV words handling required     |

## N-grams

An n-gram is a contiguous sequence of n tokens. For example:

- s1 = "The food is good"
- s2 = "The food is not good"

Keeping relevant tokens `['food', 'not', 'good']`, the unigram vectors are:

- s1 (unigrams): [1, 0, 1] # `food`, `not`, `good`
- s2 (unigrams): [1, 1, 1]

If we include bigrams (token pairs), consider the feature set:

`['food', 'not', 'good', 'food good', 'food not', 'not good']`

Then vectors become (order as above):

- s1: [1, 0, 1, 1, 0, 0]
- s2: [1, 1, 1, 0, 1, 1]

In scikit-learn's `CountVectorizer` / `TfidfVectorizer` you can control n-gram range with `ngram_range=(1,1)` (unigrams), `(1,2)` (unigrams + bigrams), etc.

## TF–IDF (Term Frequency–Inverse Document Frequency)

TF–IDF weights terms by how important they are in a document relative to the corpus.

- Term Frequency (TF) for term t in sentence d: $\\mathrm{TF}(t,d)=\\frac{\\text{count}(t,d)}{\\text{number of terms in }d}$
- Inverse Document Frequency (IDF) for term t: $\\mathrm{IDF}(t)=\\log\\left(\\dfrac{N}{\\mathrm{df}(t)}\\right)$, where $N$ is the number of documents and $\\mathrm{df}(t)$ is the number of documents containing t.

Example (corrected):

- s1 = "good hero"
- s2 = "good villain"
- s3 = "good villain hero"

Vocabulary: `['good', 'hero', 'villain']`, with $N=3$ documents.

TF table:

| word    | s1  | s2  | s3  |
| ------- | --- | --- | --- |
| good    | 1/2 | 1/2 | 1/3 |
| hero    | 1/2 | 0   | 1/3 |
| villain | 0   | 1/2 | 1/3 |

Document frequencies: $\\mathrm{df}(good)=3$, $\\mathrm{df}(hero)=2$, $\\mathrm{df}(villain)=2$.

IDF values:

- $\\mathrm{IDF}(good)=\\log(3/3)=0$
- $\\mathrm{IDF}(hero)=\\log(3/2)$
- $\\mathrm{IDF}(villain)=\\log(3/2)$

TF–IDF entries (TF multiplied by IDF):

|     | good | hero                      | villain                   |
| --- | ---- | ------------------------- | ------------------------- |
| s1  | 0    | $\\tfrac{1}{2}\\log(3/2)$ | 0                         |
| s2  | 0    | 0                         | $\\tfrac{1}{2}\\log(3/2)$ |
| s3  | 0    | $\\tfrac{1}{3}\\log(3/2)$ | $\\tfrac{1}{3}\\log(3/2)$ |

Because `good` appears in every document its IDF is zero, so it does not contribute to TF–IDF.
```
