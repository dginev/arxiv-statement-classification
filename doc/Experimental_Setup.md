# Summary
- Take AMS-labeled subset of arXMLiv dataset
- Suvey environments, arriving at a shortlist classification targets
- Use the first paragraph under each heading, with the environment name as a classification label
  - Extend with "fixed meaning" heading labels if desired (introduction, related work, conclusion,...)
- Split data into 80/20 training/testing set
- Experiment with training RNN models which given a paragraph recognize the classification label
  - Embed paragraphs as padded fixed width "480 word" windows
  - Each word is embedded into a 300 dimensional GloVe vector, via the arXMLiv GloVe embeddings
  - Cap data at 1 million paragraphs per class, due to hardware limitations
- Improve classification scheme
  - Learn from confusion matrixes to find strong separability, as well as linguistic nearness between the classes
  - Reduce 28+other label target to 8+other label target,
  - we regroup into classes with strong linguistic similarity,
  - but thus increase the semantic ambiguity of the merged classes (e.g. `proposition` is a lot vaguer in purpose than its constituents `theorem` or `lemma`).
- Arrive at a Keras `BiLSTM(128)→BiLSTM(64)→LSTM(64)` model, with a Dense(8) softmax output.
- Report a 0.95 F1-score on the 8 label classification task. 

# Experimental Design, 2018

Start with the 1.2 million documents from the arXMLiv 08.2018 dataset.

## AMS label survey; Structural headings

First, filter only the document that have AMS theorem markup from their latex sources, and manually survey the various environments used.

The initial survey is available as a [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1KdGFlc49WAJLehveASLZ94ykBTPzTcSub_omLsf8JoY/edit?usp=sharing), and was performed on the 08.2017 version of arXMLiv. 

Analyzing the report suggests that 527 distinct `{environment}` names are used 100 or more times in arXiv (08.2017), and they cover 98% of all labeled paragraphs available. There is a huge long tail contributing the last 2% of paragraphs, totaling over 19,000 distinct names, which we mostly discard to avoid incurring unnecessary noise.

We observe that the most frequent names can then be mapped into a smaller conceptual subset - 22 statement kinds in particular, as well as the "other" label for any labels that do not fit those classes. This selection has been codified in the [llamapun::ams::AmsEnv](https://github.com/KWARC/llamapun/blob/master/src/ams.rs#L108) datatype, and we cover the first ~1000 environment names, as mapped into the 23 targets.

Additionally, we shortlist 11 of the "fixed meaning" headings in scientific documents as an extension to the labeled dataset, as codified in the [llamapun::ams::StructuralEnv](https://github.com/KWARC/llamapun/blob/master/src/ams.rs#L27) datatype.

Hence the final classes for the experiment, as collected by llamapun, are:

- AMS:
    - "algorithm",
    - "assumption",
    - "case",
    - "condition",
    - "conjecture",
    - "corollary",
    - "definition", 
    - "fact",
    - "lemma",
    - "notation",
    - "paragraph",
    - "problem",
    - "proof",
    - "proposition",
    - "question", 
    - "step", 
    - "theorem"

- Structural:
    - "abstract",
    - "conclusion",
    - "discussion",
    - "introduction",
    - "method",
    - "relatedwork",

- Joint (both from AMS and Structural markup):
    - "example",
    - "caption",
    - "remark",
    - "result",
    - "acknowledgement",

- Other: first paragraph under a heading, which has *no* AMS markup, and no known structural markup from the heading.

This results in 22 named classes from AMS environments + 11 named structural classes, 5 of which collected jointly, for a total of **28 classification targets** (and "other"). 

## Representation

Using llamapun's corpus traversal and (math-rich) plain-text normalization tools, we represent each paragraph as a space-separated sequence of alphanumeric words, where: 
 * Sentence and word tokenization over a logical paragraph are performed by llamapun, and are aware of display equations, as well as other structural markup
 * words are lowercased and stripped of non-alphanumeric characters
 * paragraphs with known latexml errors, or words longer than 25 characters (which are indicative of math mode errors) are discarded completely
 * several generic inline artefacts are normalized to generic replacements:
   - numeric literals become `NUM`
   - citation markup becomes `citationelement`  
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

Collecting the data as induced by these filters results in obtaining 12,029,317 paragraph files from arXiv, each line of which contains a single sentence.

## Experimental analysis

At this point we can train a first few naive models on a subset of the data and obtain initial intuitions about the difficulties in classification. In particular, a very informative confusion matrix is:

![28 class confusion matrix](figures/confusion_matrix_normalized.png)

which was obtained from evaluating a BiLSTM model on the 28-class setup.

Next, we remind of the main goal of the current run of this specific experiment. It is to obtain a classification method for "definitional paragraphs", while optionally preserve any additional separability between the other available labels. In an effort to increase the utility of this classifier, and starting with a vague specification of "separability between mathematical statements", we can take a few liberties in the classification setup. Namely:
 - We are allowed to ignore unrelated data - as mentioned data with unrelated AMS markup can be both discarded by llamapun's extraction, as well as discarded by us if the raw data has significant problems upon manual inspection (as it would only introduce noise)
  - We are allowed to meaningfully merge label classes - a great example why such a frivolous technique is permissible is to consider that we can only train a classifier between statement kind A and statement kind B if they have a different linguistic footprint. In particular, classifying a paragraph as a `theorem` vs a `lemma` would be impossible to meaningfully do for a mathematician who has no additional context of the paper, but is looking in a single paragraph in isolation.
  

It becomes apparent that there are in fact very high correlations in certain classes, and after enumerating them a linguistic similarity pattern starts to emerge:

1. Abstracts are highly correlated with introductions, and as they were only of tangential interest for this experiment, focused on AMS labels, we drop them entirely.
2. Acknowledgements are separable with very high confidence, and we keep them as a non-mathematical control in the label set.
3. Algorithms are separable with high confidence, but upon manual inspection of the paragraphs, it turns out they are rather unfortunate artifacts. These are not the usual algorithms one sees in computer science papers, but are instead short snippets describing procedural information, in ad-hoc markup (via amsmath). For this reason, we drop them from our final modeling target of interest, which is focused on separating mathematical statements. 
4. Assumptions are highly separable, and Conditions are highly misinterpreted as assumptions. However, a merged class of the two remains significantly confused with Proposition (not shown here). As both assumption and condition can be seen as fragments of several higher level statements, we take the safe route and drop them from the data, focusing on "logical" statements (proofs, theorems, definitions...).
5. Captions are also reasonably separable, yet again - upon manual inspection of the data - it becomes clear they are very peculiar artifacts that are not comparable to a usual caption of a figure or table. Hence, we drop them from the experiment - follow-up work can separately extract all correctly marked up captions of articles (via `\caption`) and add them to the classification setup.
6. Case, Proof and Step are generally tending to the best populated Proof label. As we expect a descriptive language of derivation, and enumeration, in these types of constructs, we merge them together under the umbrella "Proof" name.
7. Conclusion - noisily correlated with several classes, hence discard, as not central to experiment
8. Conjecture, Corollary, Fact, Lemma, Proposition, Theorem - high confusion correlation, with an obvious concession to make. All six labels fit in the most general of the labels - they are all propositions, of various notability, and various degrees of being known. We hence merge them into a single "Proposition" class.
9. Definitions are highly separable, and we keep them as-is, as the main classification target of the experiment.
10. Discussion is significantly confused with Conclusion, as well as Remark. All three are indeed various kinds of discussion within an article, and may be nicely separable if merged. However, as this is a broader class of statements, further work may be needed to properly curate the data. For now we will drop them from the experiment, as tangential.
11. Example is strongly separable and we keep it unchanged.
12. Introduction is strongly separable and we keep it as a non-mathematical control in the label set.
13. Method is both very scarce in available data, but also strongly confused with a range of classes, suggesting noisy data. We drop it from the experiment.
14. Notations are not sufficiently separable, and are confused with definitions. An option would be to merge them with definition, but as we try to develop a precise classifier for the definition label, which introduces mathematical concepts, rather than only syntactic rules, we instead opt to drop them from the experiment.
15. paragraph is highly confused, and upon manual observation contains a large variety of noisy data - from captions to lemmas. Thus, we drop that subset of data from the experiment as uninformative.
16. Problem is strongly separable, and we keep it unchanged.
17. Question is not strongly separable, and is highly confused with Notation, which we already dropped, so we also drop it from the experiment.
18. Related work is strongly separable, so we keep it as-is, as a non-mathematical control label
19. Result is very highly confused, drop from the experiment.

Thus, we arrive at the following final classes for the **AMS Paragraph Classification (focus: definition discovery)** experiment,
summarizing 15 original labels into 8 classes, and ignoring the rest of the available label data:

0. acknowledgement
1. proof = case + proof + step
2. proposition = lemma + theorem + corollary + proposition + conjecture + fact
3. definition
4. example
5. introduction
6. problem
7. related work

where an "other" label is implied when the classification process returns with a score smaller than a threshold (value between 0.7-0.75 appears to be appropriate).

## Model Training

Having the classification scheme informed by the confusion matrix, allows us to arrive at highly separable classes, without performing any unreasonable techniques with data curation. It is not surprising that certain classes of statements, which are linguistically similar, but contextually distinct (lemma vs theorem, discussion vs conclusion, etc) may require additional effort to label accurately.

For the first iteration of this experiment, we remain focus on identifying with as much success as possible individual definitional paragraphs, with a model that relies on the standalone content of the paragraph. As such, reducing the labeling scheme to the most separable classes allows to test the limits of the single-paragraph classification approach. The general "other" class would contain a large variety of additional content, much of which we can't classify apriori, so a negative control over all dropped data may be in order. How should a 8-class model treat the discarded labels? We leave that as an open question for the next stage of such experiments.

For this specific experiment, we are interested in achieving a good result on training a classifier between the 8 labels.

We propose experimenting with a variety of standard models in Keras: Dense MPC, CNN, LSTM/BiLSTM/GRU, and various topologies of each.

## Model Evaluation

The best result achieved to date has been with a `BiLSTM(128)→BiLSTM(64)→LSTM(64)`, with a `Dense(8)` softmax output, for an F1 score of `0.95`, as shown below. The model was capped at 900k paragraphs per label class (before merging), and used the most frequent 1 million words in the GloVe vocabulary. The input was padded to paragraphs of `480` words, and embedded into `300`-dimensional GloVe vectors.

![8-class confusion](figures/confusion_8class_1m_normalizeds.png)