# Summary
- Take AMS-labeled subset of arXMLiv dataset
  - Extend with "fixed meaning" heading labels if desired (introduction, related work, conclusion,...)
- Suvey environments, arriving at a shortlist classification targets
- Use the first paragraph under each heading, with the environment name as a classification label
- Split data into 80/20 training/testing set
- Experiment with training RNN models which given a paragraph recognize the classification label
  - Embed paragraphs as padded fixed width "480 word" windows
  - Each word is embedded into a 300 dimensional GloVe vector, via the arXMLiv GloVe embeddings
- Improve classification scheme
  - Learn from confusion matrixes to find strong separability, as well as linguistic nearness between the classes
  - Reduce 50 label target to 13 label target (comprised of 25 of the original classes)
    - we regroup into classes with strong linguistic similarity,
    - but thus increase the semantic ambiguity of the merged classes (e.g. `proposition` is a lot vaguer in purpose than its constituents `theorem` or `lemma`).
- Arrive at a Keras `BiLSTM(128)→BiLSTM(64)→LSTM(64)` model, with a Dense(13) softmax output.
- Report a 0.91 F1-score on the 13 label classification task.

# Experimental Design, 2019

Start with the 1.2 million documents from the arXMLiv 08.2018 dataset.

## AMS label survey; Structural headings

First, filter only the document that have AMS theorem markup from their latex sources, and manually survey the various environments used.

The initial survey is available as a [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1KdGFlc49WAJLehveASLZ94ykBTPzTcSub_omLsf8JoY/edit?usp=sharing), and was performed on the 08.2017 version of arXMLiv.

Analyzing the report suggests that 527 distinct `{environment}` names are used 100 or more times in arXiv (08.2017), and they cover 98% of all labeled paragraphs available. There is a huge long tail contributing the last 2% of paragraphs, totaling over 19,000 distinct names, which we mostly discard to avoid incurring unnecessary noise.

We observe that the most frequent names can then be mapped into a smaller conceptual subset - 44 statement kinds in particular, as well as the "other" label for any labels that do not fit those classes. This selection has been codified in the [llamapun::ams::AmsEnv](https://github.com/KWARC/llamapun/blob/8fe63ed148d6dacef7e544b966e88f73e9d86086/src/ams.rs#L122) datatype, and we cover the first ~1000 environment "nicknames", mapping them into the 44 targets.

Additionally, we shortlist 12 of the "fixed meaning" headings in scientific documents as an extension to the labeled dataset, as codified in the [llamapun::ams::StructuralEnv](https://github.com/KWARC/llamapun/blob/8fe63ed148d6dacef7e544b966e88f73e9d86086/src/ams.rs#L28) datatype.

Hence the final classes for the raw extracted data, as collected by llamapun, are:

- AMS:
    - "affirmation",
    - "answer",
    - "assumption",
    - "bound",
    - "case",
    - "claim",
    - "comment",
    - "condition",
    - "conjecture",
    - "constraint",
    - "convention",
    - "corollary",
    - "criterion",
    - "definition",
    - "demonstration",
    - "expansion",
    - "experiment",
    - "expectation",
    - "explanation",
    - "fact",
    - "hint",
    - "issue",
    - "lemma",
    - "notation",
    - "note",
    - "notice",
    - "observation",
    - "principle",
    - "problem",
    - "proof",
    - "proposition",
    - "question",
    - "rule",
    - "solution",
    - "step",
    - "summary",
    - "theorem"

- Structural:
    - "abstract",
    - "conclusion",
    - "exercise",
    - "introduction",
    - "method",
    - "overview",
    - "relatedwork",

- Joint (both from AMS and Structural markup):
    - "acknowledgement",
    - "discussion",
    - "example",
    - "keywords",
    - "remark",
    - "result",


This results in 43 named classes from AMS environments + 13 named structural classes, 6 of which collected jointly, for a total of **50 distinct classification targets**.

## Representation

Using llamapun's corpus traversal and (math-rich) plain-text normalization tools, we represent each paragraph as a space-separated sequence of alphanumeric words, where:
 * Sentence and word tokenization over a logical paragraph are performed by llamapun, and are aware of display equations, as well as other structural markup
 * words are lowercased and stripped of non-alphanumeric characters. Possesives are not currently split, so `It's` becomes `its`.
 * paragraphs with known latexml errors, or words longer than 25 characters (which are indicative of math mode errors) are discarded completely
 * several generic inline artefacts are normalized to generic replacements:
   - numeric literals become `NUM` (all caps)
   - citation markup becomes `citationelement`
   - `\ref` internal references become a `ref` placeholder word.
 * formulas are represented as a sequence of word lexemes (a feature of latexml), e.g. `x=0` becomes `italic_x RELOP_equals NUM`.

An example definition would look like (indented line-breaks added for readability, these are 2 sentences, each being a single line):
```
let italic_p italic_q be two distributions on finite metric space italic_M
an emd closeness tester is an algorithm which
    takes as input samples from italic_p and italic_q together with a real number
        italic_varepsilon RELOP_greater_than NUM
    and guarantees that NUM
    if
        italic_p RELOP_equals italic_q
    then it accepts with probability at least
        NUM MULOP_divide NUM
    and NUM
    if
        italic_E italic_M italic_D OPEN_( italic_p PUNCT_, italic_q CLOSE_) RELOP_greater_than italic_varepsilon
    then it rejects with probability at least NUM MULOP_divide NUM
```
(Note: this example represents `that 1) if ... and 2) if...` by using a single sentence and switching the enumeration markers by NUM)

Collecting the data as induced by these filters results in obtaining 10.5 million paragraph files from arXiv, each line of which contains a single sentence.

**TODO: Continue updating from here**
---

## Experimental analysis (outdated -- to be renewed)

At this point we can train a first few naive models on a subset of the data and obtain initial intuitions about the difficulties in classification. In particular, a very informative confusion matrix is:

![50 class confusion matrix](https://raw.githubusercontent.com/dginev/arxiv-statement-classification/master/figures/confusion_matrix_normalized_50class.png)

which was obtained from evaluating a BiLSTM model on the 50-class setup.

Next, we remind of the original goal of the current run of this specific experiment. It is to obtain a classification method for "definitional paragraphs", while optionally preserve any additional separability between the other available labels. In an effort to increase the utility of this classifier, and starting with a vague specification of "separability between mathematical statements", we can take a few liberties in the classification setup. Namely:
 - We are allowed to ignore unrelated data: data with unrelated AMS markup can be both discarded by llamapun's extraction, as well as discarded if the raw data has significant problems upon manual inspection (as it would only introduce noise)
  - We are allowed to meaningfully merge label classes: a great example why such a frivolous technique is permissible is to consider that we can only train a classifier between statement kind A and statement kind B if they have a different linguistic footprint. In particular, classifying a paragraph as a `theorem` vs a `lemma` would be impossible to meaningfully do for a mathematician who has no additional context of the paper, but is looking at a single paragraph in isolation.


It becomes apparent that there are in fact very high correlations in certain classes, and after enumerating them a linguistic similarity pattern starts to emerge:

An up-to-date description of the various relationships between the 50 classes, can be seen in the respective [jupyter notebook](https://github.com/dginev/arxiv-statement-classification/blob/master/pre_analysis_50_class_bilstm.ipynb)

At time of writing, it appears that there is a grouping in 13 classes (starting from 25 of the original 50) that is separable with high confidence by our best baselines (BILSTM encoder-decoder model).

## Model Training

Having the classification scheme informed by the confusion matrix, allows us to arrive at highly separable classes, without performing any unreasonable techniques with data curation. It is not surprising that certain classes of statements, which are linguistically similar, but contextually distinct (lemma vs theorem, discussion vs conclusion, etc) may require additional context to label accurately.

For the first iteration of this experiment, we remain focused on identifying, with as much success as possible, individual definitional paragraphs, with a model that relies on the standalone content of the paragraph. As such, reducing the labeling scheme to the most separable classes allows to test the limits of the single-paragraph classification approach.

We train a number of baseline models, as seen in the jupyter notebooks contained in the repository's root directory, which confirm our intuition that this is a well-posed task aligned with the state of art in text classification.

## Model Evaluation

The best result achieved to date has been with a `BiLSTM(128)→BiLSTM(64)→LSTM(64)`, with a `Dense(`12)` softmax output, for an F1 score of `0.91`, as shown below. The model was capped at 900k paragraphs per label class (before merging), and used the most frequent 1 million words in the GloVe vocabulary. The input was padded to paragraphs of `480` words, and embedded into `300`-dimensional GloVe vectors.

![13-class confusion](https://raw.githubusercontent.com/dginev/arxiv-statement-classification/master/figures/confusion_matrix_normalized_13class.png)
