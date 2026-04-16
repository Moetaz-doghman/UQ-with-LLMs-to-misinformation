# ISOT UQ Evaluation Report

## Run Summary
- Total examples: 200
- Accuracy: 0.695
- Parse failure rate: 0.000
- Number of incorrect predictions: 61

## How To Read The UQ Metrics
- Higher uncertainty should ideally correspond to model mistakes.
- AUROC > 0.5 means the method detects errors better than random ranking.
- AUROC near 1.0 means very strong error detection.
- AUROC near 0.5 means the method is close to random.
- AUROC below 0.5 means the ranking is misleading or inverted for this run.
- ROC curves can look like staircases instead of smooth arcs when the dataset is small or when the UQ method returns only a few distinct score values.
- Verbalized1S/2S are intentionally kept for a second experiment with a dedicated prompt that asks the model to report its confidence explicitly.

## Method Interpretations
- `Eccentricity_NLI_score_entail`: AUROC=0.628 | mean incorrect - mean correct = 0.061. Measures how spread out sampled answers are in a semantic graph. Higher values mean the generations occupy more distant semantic regions.
- `EigValLaplacian_NLI_score_entail`: AUROC=0.627 | mean incorrect - mean correct = 0.061. Measures how semantically diverse the sampled answers are. Higher scores mean the model's sampled answers disagree more in meaning.
- `LexicalSimilarity_rougeL`: AUROC=0.531 | mean incorrect - mean correct = 0.007. Uses surface-form similarity across sampled answers. Lower similarity between samples implies higher uncertainty.
- `LexicalSimilarity_rouge1`: AUROC=0.531 | mean incorrect - mean correct = 0.007. Surface-form overlap using ROUGE-1. Lower overlap between samples implies higher uncertainty.
- `NumSemSets`: AUROC=0.531 | mean incorrect - mean correct = 0.061. Counts how many distinct semantic groups appear among sampled answers. Higher values mean the model explores more incompatible meanings.
- `DegMat_Jaccard_score`: AUROC=0.530 | mean incorrect - mean correct = 0.010. A Jaccard-overlap variant of DegMat. It captures how tightly grouped sampled answers are at the lexical level.
- `EigValLaplacian_Jaccard_score`: AUROC=0.530 | mean incorrect - mean correct = 0.014. A lexical-overlap variant of EigValLaplacian. It measures disagreement through token overlap instead of NLI.
- `Eccentricity_Jaccard_score`: AUROC=0.530 | mean incorrect - mean correct = 0.061. A lexical-overlap variant of Eccentricity. Higher values mean sampled answers are dispersed in lexical space.
- `DegMat_NLI_score_entail`: AUROC=0.433 | mean incorrect - mean correct = 0.018. Uses graph connectivity over semantic similarities. It reflects how tightly grouped the sampled answers are.
- `KernelLanguageEntropy`: AUROC=0.433 | mean incorrect - mean correct = 0.020. Computes entropy over a heat kernel built from semantic relations between samples. Higher values mean more uncertainty in the semantic output space.
- `LUQ`: AUROC=0.433 | mean incorrect - mean correct = 0.022. Long-text uncertainty score derived from entailment and contradiction logits between sampled responses. Higher values mean less semantic agreement.

## Main Takeaway
- Best method in this run: `Eccentricity_NLI_score_entail` with AUROC 0.628.

## Wrong But Apparently Confident Examples
- Example 0: gold=FAKE, predicted=REAL, title=WATCH: Chuck Todd Puts The Screws To Mitch McConnell For Not Giving Merrick Garland A Chance
- Example 1: gold=FAKE, predicted=REAL, title=ANGRY Venezuelans CHASE Their President…Bang On Pots And Yell [Video]
- Example 2: gold=FAKE, predicted=REAL, title=Cruz Trying To Hold Fiorina’s Hand Is More Awkward Than A Middle School Dance (VIDEO)
- Example 9: gold=FAKE, predicted=REAL, title=Dem Challenger To Paul Ryan Has Raised A Massive Amount Of Money In The Last 24 hours
- Example 13: gold=FAKE, predicted=REAL, title=WATCH: Democrats Release Video PERFECTLY Highlighting Hypocrisy Of Trump’s Appointments
- Example 16: gold=FAKE, predicted=REAL, title=Hillary Talks Taco Trucks, SHREDS Trump, And The Crowd Goes WILD (VIDEO)
- Example 24: gold=FAKE, predicted=REAL, title=INSIDE TRUMP’S CHARITY BALL Tonight At Beautiful Mar-A-Lago…Protests Outside [Video]
- Example 25: gold=FAKE, predicted=REAL, title=(VIDEO) DESPERATION? CLINTON CAMPAIGN PANDERS TO RADICAL BLACK GROUPS
- Example 32: gold=FAKE, predicted=REAL, title=BREAKING! INVESTIGATION: Hillary Clinton Did NOT Comply With Records Rules
- Example 26: gold=FAKE, predicted=REAL, title=WATCH: Sean Spicer In December STRONGLY Opposed Exactly What He Did Today