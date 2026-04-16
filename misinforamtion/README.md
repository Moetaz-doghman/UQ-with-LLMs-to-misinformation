# ISOT UQ Evaluation

Ce dossier contient un pipeline d'evaluation pour comparer plusieurs methodes d'uncertainty estimation sur une tache de misinformation classification avec le dataset ISOT.

## Fichiers principaux

- [run_isot_uq_eval.py](/C:/Users/doghm/Desktop/projet-infox/lm-polygraph/misinforamtion/run_isot_uq_eval.py)
  Script principal d'evaluation.
- [data/isot/Fake.csv](/C:/Users/doghm/Desktop/projet-infox/lm-polygraph/misinforamtion/data/isot/Fake.csv)
  Exemples `FAKE`.
- [data/isot/True.csv](/C:/Users/doghm/Desktop/projet-infox/lm-polygraph/misinforamtion/data/isot/True.csv)
  Exemples `REAL`.
- [.env.example](/C:/Users/doghm/Desktop/projet-infox/lm-polygraph/misinforamtion/.env.example)
  Exemple de fichier pour la cle API.

## Ou mettre la cle API

Le script lit d'abord `OPENAI_API_KEY` dans l'environnement.

Tu peux aussi la mettre dans:
- [misinforamtion/.env](/C:/Users/doghm/Desktop/projet-infox/lm-polygraph/misinforamtion/.env)

Format:

```env
OPENAI_API_KEY=ta_cle_openai_ici
```

Le script charge automatiquement ce fichier avant de construire `BlackboxModel.from_openai(...)`.

## Execution

Depuis la racine du repo:

```powershell
python misinforamtion\run_isot_uq_eval.py --sample-per-class 20
```

Avec un nom de run explicite:

```powershell
python misinforamtion\run_isot_uq_eval.py --sample-per-class 20 --run-name pilot_20
```

## Nouvelle structure des sorties

Chaque execution cree un dossier de run dans:
- [misinforamtion/outputs](/C:/Users/doghm/Desktop/projet-infox/lm-polygraph/misinforamtion/outputs)

Exemple:
- `misinforamtion/outputs/isot_gpt-4.1-mini_YYYYMMDD_HHMMSS/`

Ce dossier contient:

- `examples_compact.csv`
  Vue compacte, facile a lire.
- `examples_full.csv`
  Version detaillee avec article complet et prompt.
- `uq_scores_long.csv`
  Format long, pratique pour analyse et visualisation.
- `method_summary.csv`
  Resume par methode: AUROC, moyennes, medians, delta correct/incorrect.
- `wrong_but_confident.csv`
  Exemples faux avec incertitude faible.
- `report.md`
  Resume interpretatif du run.
- `plots/`
  Dossier des visualisations.

## Visualisations generees

Dans `plots/`, le script produit:

- `auroc_bar.png`
  Compare les methodes sur la detection d'erreurs.
- `roc_curves.png`
  Montre la capacite de chaque methode a separer les predictions correctes et incorrectes.
- `uncertainty_boxplots.png`
  Compare les distributions de scores pour les exemples corrects vs incorrects.
- `confusion_matrix.png`
  Montre les erreurs de classification `REAL/FAKE`.
- `mean_uncertainty_rank_scatter.png`
  Montre ou tombent les erreurs dans le classement global d'incertitude.

## Comment interpreter les methodes

- `EigValLaplacian`
  Plus les generations echantillonnees divergent semantiquement, plus l'incertitude monte.
- `NumSemSets`
  Compte le nombre de groupes semantiques distincts dans les generations.
- `LexicalSimilarity_rougeL`
  Mesure la similarite de surface entre generations. Faible similarite = plus d'incertitude.
- `DegMat`
  Regarde a quel point les generations restent connectees dans le graphe semantique.

## Comment interpreter les graphiques

- Si `AUROC > 0.5`, la methode aide a detecter les erreurs.
- Si `AUROC` se rapproche de `1.0`, la methode est forte.
- Si les boxplots montrent des scores plus eleves pour `Incorrect` que pour `Correct`, c'est bon signe.
- Si beaucoup de points rouges apparaissent tres bas dans `mean_uncertainty_rank_scatter.png`, cela veut dire que le modele fait des erreurs avec une incertitude trop faible.

## Remarque importante

`MonteCarloSequenceEntropy` n'est pas utilisee ici, car dans cette codebase LM-Polygraph elle depend de statistiques white-box et n'est donc pas adaptee a ce setup OpenAI black-box.
