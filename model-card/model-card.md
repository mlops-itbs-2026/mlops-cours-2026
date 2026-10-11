# Model Card — `scoring-model` v3 (scoring de crédit à la consommation)

Binôme : corrigé enseignant · Dataset : `german-credit` · Date : 5 octobre 2026 ·
Statut : **prototype pédagogique — ne pas déployer pour de vraies décisions de crédit**

> Format : Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019), sections 1 à 9,
> complétées par la sécurité de la chaîne (section 10) et le cadre réglementaire (section 11).
> Chiffres : `reports/fairness_baseline.json`, `reports/fairness_mitigated.json`,
> `reports/shap_importance.json`, `reports/metrics.json` (Lab 1).

## 1. Détails du modèle

- **Nom / version** : `scoring-model` version 3, alias `@champion` du Registry MLflow
  (`models:/scoring-model@champion`) ; run MLflow `d3a0e901e8124008b07946ae68544205`
  (expérience `scoring`, tag `launched_by=dvc`).
- **Type** : Pipeline scikit-learn 1.9.1 = prétraitement décrit par `dataset.yaml`
  (imputation médiane + standardisation des 7 numériques, imputation + one-hot des 13
  catégorielles, 61 colonnes) + `LogisticRegression(C=0.1, class_weight="balanced")`.
- **Sortie** : probabilité que la demande soit un « mauvais payeur » (`credit_risk = bad`).
  **Décision** : refus si P(bad) ≥ **0,318** (paramètre `decision_threshold` du run). Ce
  seuil minimise le coût métier de `dataset.yaml` (faux négatif = 5, faux positif = 1)
  pour un modèle entraîné avec des poids de classes équilibrés.
- **Artefacts** : modèle MLflow sérialisé en skops (types hors liste de confiance :
  `numpy.dtype`, autorisé) ; export ONNX opset 17 (`packaging/model/model.onnx`, empreinte
  dans `metadata.json`) servi par l'API FastAPI `scoring-api` (Lab 4), image durcie
  `scoring-api:distroless` (Lab 8).
- **Traçabilité** : données versionnées par DVC (`dvc.lock`, md5 du brut dans le tag
  `data_md5` du run), code et paramètres dans Git, pipeline `dvc repro` reproductible.
- **Auteurs / contact** : équipe projet du cours MLOps (binôme). **Licence** : code MIT
  (usage pédagogique), données CC BY 4.0.
- **Citation des données** : Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine
  Learning Repository. https://doi.org/10.24432/C5NC77

## 2. Usage prévu

- **Usage primaire** : support pédagogique du cours MLOps — illustrer, sur un cas réaliste,
  la chaîne versionnement → packaging → serving → monitoring → gouvernance d'un modèle de
  scoring de crédit.
- **Utilisateurs visés** : étudiants et enseignants du cours.
- **Usage envisageable après mise en conformité** (hors de ce projet) : aide à la décision
  pour un analyste crédit, qui reste décisionnaire et peut passer outre le score.
- **Usages interdits (hors périmètre)** :
  - décider **automatiquement** d'accorder ou de refuser un crédit à une personne réelle ;
  - fixer un prix, un taux ou un plafond de crédit ;
  - tout usage sur une autre population que celle des données (Allemagne, 1973-1975) ;
  - toute utilisation des attributs sensibles pour cibler, exclure ou tarifer un groupe ;
  - recrutement, assurance, logement ou tout autre domaine que le crédit à la consommation.

## 3. Facteurs

- **Attributs sensibles** (section `sensitive` de `dataset.yaml`) :
  - `age`, découpé en deux groupes `<25` / `>=25` ;
  - `personal_status` : sexe et situation familiale **mélangés** dans une seule variable ;
  - `foreign_worker` : travailleur étranger (`yes` / `no`).
- **Le modèle utilise directement ces trois attributs** comme variables d'entrée.
- **Proxies** (`reports/fairness_mitigated.json`) : sans la colonne, le groupe se devine
  quand même à partir des autres variables — AUC 0,77 pour l'âge (`housing`,
  `personal_status`, `employment`), 0,71 pour `personal_status` (`age`, `housing`,
  `employment`), 0,78 pour `foreign_worker` (`duration`, `property_magnitude`, `purpose`).
- **Facteurs d'environnement** : conjoncture économique, politique de crédit de la banque
  et réglementation de 1973-1975, sans rapport avec aujourd'hui.

## 4. Métriques

### Performance globale (jeu de test, 200 demandes, seuil 0,318)

| Métrique | Valeur |
|---|---|
| ROC AUC (métrique primaire, gate 0,70) | 0,809 |
| Précision moyenne (PR AUC) | 0,646 |
| Rappel des mauvais payeurs (TPR) | 0,883 (53 / 60) |
| Précision | 0,421 |
| F1 | 0,570 |
| Accuracy | 0,600 |
| Taux de refus | 63 % |
| Coût métier (5 × FN + 1 × FP) | **108** (7 FN, 73 FP) |
| Référence « tout refuser » / « tout accepter » | 140 / 300 |

### Performance par groupe (au seuil 0,318)

Refus = prédiction « bad » ; FPR = part des **bons** payeurs refusés à tort.

#### age

| Groupe | n | Défaut réel | Refus [IC 95 %] | TPR | FPR | Coût/demande |
|---|---|---|---|---|---|---|
| <25 | 39 | 49% | 77% [63% ; 90%] | 95% | 60% | 0.44 |
| >=25 | 161 | 25% | 60% [52% ; 67%] | 85% | 51% | 0.57 |

#### personal_status

| Groupe | n | Défaut réel | Refus [IC 95 %] | TPR | FPR | Coût/demande |
|---|---|---|---|---|---|---|
| female div/dep/mar | 61 | 33% | 72% [60% ; 81%] | 95% | 61% | 0.49 |
| male div/sep | 9 | 33% | 67% [33% ; 100%] | 100% | 50% | 0.33 |
| male mar/wid | 12 | 25% | 83% [57% ; 100%] | 67% | 89% | 1.08 |
| male single | 118 | 29% | 56% [46% ; 65%] | 85% | 44% | 0.53 |

#### foreign_worker

| Groupe | n | Défaut réel | Refus [IC 95 %] | TPR | FPR | Coût/demande |
|---|---|---|---|---|---|---|
| no | 6 | 17% | 50% [0% ; 100%] | 100% | 40% | 0.33 |
| yes | 194 | 30% | 63% [56% ; 70%] | 88% | 53% | 0.55 |

### Indicateurs d'équité

| Attribut | Écart de parité (refus) | Ratio d'acceptation (règle des 4/5) | Écart equalized odds |
|---|---|---|---|
| age | 0,173 | **0,57** | 0,094 |
| personal_status | 0,274 | **0,38** | 0,448 |
| foreign_worker | 0,134 | **0,73** | 0,126 |

## 5. Données d'évaluation

- 200 demandes (20 %) tirées du fichier brut par un split **stratifié** sur la cible
  (`random_state = 42`, `dataset.yaml`) : 60 mauvais payeurs (30 %), 140 bons.
- Même prétraitement que l'entraînement, appliqué dans le Pipeline (aucune fuite).
- **Effectifs faibles** pour plusieurs groupes : 39 moins de 25 ans, 9 et 12 dans deux
  modalités de `personal_status`, **6** travailleurs non étrangers. Les taux de ces groupes
  ont des intervalles de confiance très larges (jusqu'à [0 % ; 100 %]).

## 6. Données d'entraînement

- Statlog German Credit (UCI), 1 000 demandes de crédit à la consommation d'une banque
  allemande (1973-1975), 20 variables, cible `credit_risk` (`bad` 30 %). Licence CC BY 4.0.
- 800 demandes (80 %) pour l'entraînement, versionnées par DVC (`data/processed/train.csv`).
- Échantillon **non représentatif** d'une population actuelle : les mauvais payeurs ont été
  sur-représentés à la constitution du jeu, et la période est très ancienne.
- **Qualité de la variable sensible** : d'après la documentation du jeu corrigé « South
  German Credit » (UCI 522, Grömping 2019), le codage d'origine est erroné et le sexe ne
  peut pas être retrouvé à partir de `personal_status`. L'analyse par ce groupe mesure donc
  des écarts entre libellés peut-être faux.
- Données personnelles pseudonymes (pas d'identifiant direct) ; publiées pour la recherche.

## 7. Analyse quantitative

**Disparités mesurées** (`reports/fairness_baseline.txt`) : les moins de 25 ans sont
refusés à 77 % contre 60 % (ratio d'acceptation 0,57, sous le seuil de 0,8 de la règle des
4/5). Leur taux réel de défaut est plus élevé (49 % contre 25 %), mais parmi les **bons**
payeurs, les jeunes sont aussi plus souvent refusés à tort (FPR 60 % contre 51 %). Pour
`personal_status`, l'écart le plus fort (FPR 89 % pour `male mar/wid`) porte sur 12
personnes : non concluant. Pour `foreign_worker`, 6 personnes : non mesurable.

**Atténuation** (`reports/fairness_mitigated.txt`, attribut `age`) :

| Stratégie | Coût | Δ coût | Écart de parité | Ratio d'acceptation | Écart EO | Attribut en production |
|---|---|---|---|---|---|---|
| Champion | 108 | — | 0,173 | 0,57 | 0,094 | oui |
| Sans la colonne `age` | 111 | +2,8 % | 0,154 | 0,60 | 0,094 | non |
| ThresholdOptimizer, objectif accuracy | 239 | +121,3 % | 0,054 | 0,94 | 0,025 | oui |
| ThresholdOptimizer, objectif coût | 114 | +5,6 % | 0,083 | 0,80 | 0,012 | oui |
| ExponentiatedGradient, objectif coût | 116 | +7,4 % | 0,065 | 0,83 | 0,118 | non |

- Retirer la colonne ne suffit pas (« fairness through unawareness ») : les proxies
  reconstituent l'information (AUC 0,77), l'écart de parité baisse à peine.
- Le post-traitement réglé sur le coût métier ramène l'écart equalized odds de 0,094 à
  0,012 pour +5,6 % de coût, mais il **applique un seuil différent selon l'âge** : un
  traitement explicitement différencié, à justifier juridiquement avant tout usage.
- ExponentiatedGradient n'utilise pas l'âge en production et améliore la parité (ratio
  0,83) pour +7,4 % de coût ; sur 39 jeunes, l'écart equalized odds mesuré en test (0,118)
  ne confirme pas la contrainte respectée à l'entraînement. Ses décisions sont tirées au
  sort parmi plusieurs modèles (classifieur randomisé) : deux dossiers identiques peuvent
  recevoir des réponses différentes.
- **Décision** : pas de mitigation déployée dans ce prototype ; recommandation d'un
  ré-entraînement sous contrainte (in-processing) validé sur davantage de données.

**Explicabilité** (`reports/shap_importance.json`, LinearExplainer, contributions en
log-odds) : variables les plus influentes `checking_status` (0,53), `savings_status`
(0,28), `duration` (0,26), `installment_commitment` (0,25), `credit_history` (0,25).
Attributs sensibles : `personal_status` 7e (0,23), `age` 15e (0,06), `foreign_worker` 17e
(0,04) sur 20. Explications locales (`reports/explanations_local.md`) : `personal_status =
male div/sep` figure parmi les 3 premiers motifs du refus le plus net (demande 819) ; pour
le cas limite (demande 278, P = 0,317, accordée), **changer seulement l'âge** pour la
médiane des moins de 25 ans ferait basculer la décision en refus.

## 8. Considérations éthiques

- Le modèle reproduit des décisions de crédit historiques : les écarts observés selon
  l'âge, le statut personnel et la nationalité peuvent refléter des discriminations passées
  autant qu'un risque réel.
- Il utilise directement des attributs protégés (âge, sexe et situation familiale via
  `personal_status`, nationalité via `foreign_worker`) : en droit européen, fonder un refus
  de crédit sur ces critères est une discrimination directe, sauf justification objective.
- Le coût asymétrique (FN = 5) conduit à refuser 63 % des demandes, dont 73 bons payeurs
  sur 140 : un coût social important, supporté surtout par les jeunes.
- Les explications SHAP décrivent le **modèle**, pas une causalité : « vos économies sont
  faibles » est une corrélation apprise, pas un conseil fiable pour obtenir un crédit.

## 9. Mises en garde et recommandations

- **NE PAS** utiliser pour de vraies décisions de crédit, ni de façon entièrement
  automatisée (RGPD art. 22).
- **NE PAS** généraliser à une autre population, une autre époque ou un autre produit.
- Retirer les attributs protégés des entrées **et** traiter les proxies, puis ré-auditer.
- Évaluer l'équité sur un jeu de test plus grand (les petits groupes rendent les écarts
  non concluants), suivre les métriques par groupe en production (Lab 7).
- Réviser la matrice de coûts avec le métier : elle fixe le taux de refus.
- Toute décision défavorable doit pouvoir être expliquée (motifs principaux,
  `governance/explain_local.py --id …`) et revue par un humain.

## 10. Sécurité de la chaîne

- Image `scoring-api:distroless` (base Docker Hardened Images Python 3.12, sans shell,
  utilisateur 65532) : rapports `reports/trivy_critical.txt` (image du Lab 4) et
  `reports/trivy_distroless.txt`.
- Modèle servi au format ONNX (aucun code exécuté au chargement) ; ModelScan :
  `reports/modelscan_onnx.txt`, `reports/modelscan_pickle.txt` (pickle piégé détecté,
  CRITICAL `posix.system`) ; skops du Registry audité (`numpy.dtype` seul type autorisé).
- Kubernetes : ServiceAccount `scoring-sa` sans jeton monté, Role limité à `get` sur le
  Secret `s3-credentials` (`k8s/rbac-scoring.yaml`) ; identifiants S3 synchronisés par
  External Secrets depuis la source de vérité, aucune valeur secrète dans Git
  (`k8s/mlflow-creds-source.yaml`).

## 11. Cadre réglementaire

- **AI Act (règlement (UE) 2024/1689)** : un système d'évaluation de la solvabilité ou de
  score de crédit de personnes physiques est **à haut risque** (annexe III, point 5 b). Le
  règlement « Digital Omnibus » (UE) 2026/1744, publié au JO le 24 juillet 2026, reporte
  l'application des obligations des systèmes de l'annexe III au **2 décembre 2027**.
- **RGPD art. 22** : le calcul d'un score de crédit sur lequel le prêteur s'appuie de
  manière déterminante est une décision individuelle automatisée (CJUE, C-634/21, SCHUFA,
  7 décembre 2023) ; l'explication de la « logique sous-jacente » doit être compréhensible
  (CJUE, C-203/22, 27 février 2025).
- Détail des obligations, preuves produites et écarts : `reports/compliance-ai-act.md`.
