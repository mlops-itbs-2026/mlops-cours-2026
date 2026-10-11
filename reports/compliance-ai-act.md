# Rapport de conformité — AI Act et RGPD — `scoring-model` v3

Binôme : corrigé enseignant · Date : 5 octobre 2026 · Système : API de scoring de crédit
(`scoring-api`, InferenceService `scoring-model`, namespace `scoring`)

## 1. Classification du risque

- **Usage pédagogique (ce projet)** : aucune décision réelle, pas de mise sur le marché ni
  de mise en service auprès de personnes concernées : hors champ des obligations.
- **Usage réel visé par le cas d'étude** : évaluer la solvabilité de personnes physiques ou
  établir leur score de crédit = système **à haut risque**, annexe III, point 5 b) du
  règlement (UE) 2024/1689 (seule exception : la détection de fraude financière).
- **Calendrier (octobre 2026)** : interdictions (art. 5) applicables depuis le 2 février
  2025, obligations des modèles d'IA à usage général depuis le 2 août 2025, transparence
  (art. 50) depuis le 2 août 2026. Le règlement « Digital Omnibus » (UE) 2026/1744 (JO du
  24 juillet 2026, en vigueur le 27 juillet 2026) reporte les obligations des systèmes à
  haut risque de l'annexe III au **2 décembre 2027** (annexe I : 2 août 2028).
- **Déjà applicable aujourd'hui** : RGPD art. 22 (décision individuelle automatisée) et
  art. 15 (accès, logique sous-jacente). Un score de crédit sur lequel le prêteur s'appuie
  de manière déterminante est une telle décision (CJUE C-634/21 SCHUFA, 7 décembre 2023).
- **À venir** : directive (UE) 2023/2225 sur le crédit aux consommateurs, applicable à
  partir du 20 novembre 2026 (évaluation de solvabilité, droit à une intervention humaine
  en cas de traitement automatisé).

## 2. Checklist AI Act

| Obligation (système à haut risque) | Comment le projet y répond | Preuve |
|---|---|---|
| Gestion des risques (art. 9) | Gates de validation (Lab 3), monitoring et alertes (Lab 7), audit d'équité (Lab 8) | `reports/fairness_*.txt`, alertes Prometheus |
| Données et gouvernance des données (art. 10) | Contrat `dataset.yaml`, versionnement DVC, validation GX, analyse des biais par groupe | `dvc.lock`, `gx/`, `reports/fairness_baseline.json` |
| Documentation technique (art. 11, annexe IV) | Model Card versionnée avec le code | `model-card/model-card.md` |
| Journalisation (art. 12, conservation ≥ 6 mois côté déployeur, art. 26) | Logs de l'API collectés dans Loki (Lab 7) | politique de rétention Loki |
| Transparence, notice d'utilisation (art. 13) | Usages prévus et interdits, limites, seuil documentés | Model Card, sections 2 et 9 |
| Contrôle humain (art. 14) | Score = aide à la décision ; explication par demande pour l'analyste | `governance/explain_local.py`, `reports/explanations_local.md` |
| Exactitude, robustesse, cybersécurité (art. 15) | Métriques par groupe, image durcie scannée, ONNX au lieu de pickle, RBAC, secrets externalisés | `reports/trivy_*.txt`, `reports/modelscan_*.txt`, `k8s/*.yaml` |
| Système de gestion de la qualité (art. 17) | CI/CD, revues de code, tests (Lab 6) | `.github/workflows/ci.yml` |
| Enregistrement dans la base de l'UE (art. 49) | Non réalisé (prototype) | — |
| Analyse d'impact sur les droits fondamentaux (art. 27, déployeur d'un système 5 b) | Non réalisée | — |
| Explication des décisions individuelles (art. 86) | Motifs principaux par demande | `reports/explanations_local.md` |

## 3. Preuves produites

- Sécurité de l'image : `reports/trivy_full.txt`, `reports/trivy_critical.txt` (image du
  Lab 4), `reports/trivy_distroless.txt` (image durcie `scoring-api:distroless`).
- Sécurité du modèle : `reports/modelscan_onnx.txt`, `reports/modelscan_pickle.txt`.
- Équité : `reports/fairness_baseline.{txt,json}`, `reports/fairness_mitigated.{txt,json}`.
- Explicabilité : `reports/shap_summary_bar.png`, `reports/shap_beeswarm.png`,
  `reports/shap_importance.json`, `reports/shap_waterfall_*.png`,
  `reports/explanations_local.md`.
- Documentation : `model-card/model-card.md`.
- Kubernetes : `k8s/rbac-scoring.yaml` (ServiceAccount sans jeton, Role minimal),
  `k8s/mlflow-creds-source.yaml` (SecretStore + ExternalSecret, aucun secret dans Git).
- Traçabilité : runs MLflow (paramètres, métriques, `data_md5`), `dvc.lock`, historique Git.

## 4. Écarts identifiés

- **Contrôle humain** : aucune interface ni procédure permettant à un analyste de revoir,
  contester ou annuler une décision ; pas de runbook d'intervention.
- **Équité non concluante** : petits groupes (6 à 39 personnes en test) ; écart
  d'acceptation sous la règle des 4/5 pour l'âge et `personal_status` ; attributs protégés
  utilisés directement par le modèle ; variable `personal_status` mal codée dans la source.
- **Données** : période 1973-1975, mauvais payeurs sur-représentés, population
  allemande : non représentatives d'une clientèle actuelle.
- **Organisation** : pas de comité d'éthique, pas d'AIPD (RGPD art. 35), pas d'analyse
  d'impact sur les droits fondamentaux (AI Act art. 27), pas de conservation de la
  documentation 10 ans (art. 18), pas d'enregistrement UE, pas d'évaluation de conformité.
- **Explications** : SHAP explique le modèle, pas la causalité ; aucune explication
  contrefactuelle (« que changer pour être accepté ? ») n'est fournie au demandeur.

## 5. Recommandations

1. Avant tout usage réel : AIPD, analyse d'impact sur les droits fondamentaux, évaluation
   de conformité, enregistrement dans la base de l'UE, désignation des responsables.
2. Ré-entraîner sans les attributs protégés, traiter les proxies, appliquer une
   atténuation in-processing (ExponentiatedGradient) et ré-auditer sur un jeu plus large.
3. Mettre en place une revue humaine obligatoire des refus et une voie de contestation.
4. Suivre les métriques d'équité par groupe en production (Lab 7) avec des seuils d'alerte.
5. Générer la Model Card et ce rapport dans la CI à chaque nouveau champion (Lab 6).
6. Audit annuel, conservation de la documentation technique 10 ans, journaux ≥ 6 mois.

## Sources

- Règlement (UE) 2024/1689 (AI Act), annexe III point 5 b), art. 27 et 86 :
  https://artificialintelligenceact.eu/annex/3/
- Règlement (UE) 2026/1744 (« Digital Omnibus » sur l'IA), JO du 24 juillet 2026 :
  https://cdp.cooley.com/digital-ai-omnibus-delays-key-deadlines-introduces-new-rules/
- CJUE C-634/21 (SCHUFA, 7 décembre 2023) :
  https://www.aoshearman.com/en/insights/ao-shearman-on-data/cjeu-rules-that-a-credit-score-constitutes-automated-decision-making-under-the-gdpr
- CJUE C-203/22 (Dun & Bradstreet Austria, 27 février 2025) :
  https://www.taylorwessing.com/en/insights-and-events/insights/2025/03/ecj-on-scoring
- Directive (UE) 2023/2225 (crédit aux consommateurs) :
  https://oeil.europarl.europa.eu/oeil/en/document-summary?id=1764052
