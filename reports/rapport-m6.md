# Rapport — Lab 6 : CI/CD/CT de bout en bout

Binôme : corrigé enseignant · Fork : `enseignant/mlops-cours-2026` · Dataset : `german-credit` ·
Image en prod à la fin du lab : `ghcr.io/enseignant/mlops-cours-2026/scoring-api:2.1.9` ·
`@champion` : v5

> Corrigé type. Les chiffres de la gate sont ceux d'un run réel (lot à dérive × 1,5, puis × 2) ;
> les durées sont des ordres de grandeur sur un portable 4 cœurs / 16 Go : mesurez les vôtres.

## 1. Architecture

```text
 vous ── git push ──► GitHub (fork) ──► GitHub Actions : ruff, pytest, build, test de fumée
                          ▲   │                          └─► push ghcr.io/…/scoring-api:2.1.N
                          │   └─ commit du tag (job deploy-staging) ─┐
                          │                                          ▼
                          │                   Argo CD (cluster) tire k8s/overlays/* toutes les 60 s
                          │                      ├─► scoring-staging : ISVC scoring-model (0-1 pod)
                          │                      └─► scoring : ISVC scoring-model, canary KServe
 bot GitOps (poste) ──────┘ commits : packaging/model, overlay prod, canaryTrafficPercent
        ▲
 webhook CT (poste, :8095) ◄── événement de dérive (Evidently 0.7, rapport rangé dans MLflow)
        │
        └─► KFP ct-pipeline : ingest → validate → prepare → train → evaluate
                              → register (@challenger) → gate (@champion) ──► MLflow
```

Personne ne « pousse » vers le cluster : GitHub Actions publie une image et écrit dans Git ;
Argo CD tire Git ; Kubernetes tire l'image depuis GHCR.

## 2. Temps du cycle

| Étape | Durée |
|---|---|
| Dérive détectée → run KFP lancé | 6 s (rapport Evidently 3 s, dépôt dans MLflow, POST) |
| Pipeline CT (ingest → gate), 7 tâches | 4 min 10 s |
| Commit du modèle → image publiée (CI : tests 2 min 30, build + test de fumée 1 min 40) | 4 min 30 s |
| Image publiée → staging servi (Argo CD ≤ 70 s, révision Knative, réveil depuis 0 pod) | 1 min 50 s |
| Canary 10 % en place, analyse naïve, promotion à 100 % appliquée | 2 min 40 s |
| **Cycle complet** | **≈ 13 min 30 s** |

La CI et le pipeline CT dominent. Pistes : cache uv et cache des couches Docker (déjà actifs),
ne lancer que les tests utiles sur un commit qui ne touche que `packaging/model/`, webhook
GitHub → Argo CD (impossible vers kind, possible avec un cluster joignable), modèle plus lourd
= pipeline CT plus long : le ré-entraînement dominerait alors tout le reste.

## 3. Gates et décisions

- Gate de promotion, lot à dérive × 1,5 : challenger v2 ROC AUC 0,8094, coût 108 ; champion v1
  évalué sur les mêmes données récentes : 0,8050, coût 124 → **promu** (coût non dégradé,
  ROC AUC ≥ 0,70). Le champion, appris sur l'ancienne population, décide moins bien sur la
  nouvelle ; le ré-entraînement retrouve exactement le niveau d'origine (108).
- Test de la gate avec des étiquettes mélangées : ROC AUC 0,44 → **refusé**, `@champion`
  inchangé. La règle de coût seule aurait laissé passer (150 ≤ 183 : le champion aussi se
  trompe sur ces étiquettes) : il faut plusieurs règles.
- Canary : 200 requêtes à 10 % → 21 réponses de la nouvelle image ; analyse naïve (50
  prédictions sur la route `latest-`, 0 erreur) → promotion à 100 %.

## 4. Points d'attention

- **Webhook** : jeton obligatoire (`Authorization: Bearer`, comparaison à temps constant),
  écoute sur `127.0.0.1` seulement ; un seul ré-entraînement à la fois et 10 min minimum
  entre deux (une alerte qui boucle ne lance pas dix pipelines).
- **Gate automatique** : acceptable ici parce que le canary limite l'exposition (10 %) et
  que tout est réversible par un commit ; pour un modèle de crédit en production réelle
  (AI Act, haut risque), une validation humaine (PR) reste attendue avant 100 %.
- **Retour arrière** : trois niveaux — trafic (`canaryTrafficPercent: 0`, instantané, la
  révision précédente est encore chaude), Registry (`@champion` restauré depuis le tag
  `previous_champion`), Git (`git revert` du commit de prod).
- **Secrets** : aucun dans Git (`CT_WEBHOOK_TOKEN` dans `.env`, `GITHUB_TOKEN` éphémère
  fourni par Actions). Le bot pousse avec les droits Git de l'étudiant : trop large.

## 5. Différences avec la prod

- Bot CT dans le cluster (Deployment), avec une GitHub App limitée au dépôt et une PR
  plutôt qu'un commit direct sur `main` (protection de branche).
- Données récentes réellement étiquetées (délai d'observation des défauts de paiement :
  plusieurs mois en crédit), pas un lot simulé.
- Analyse du canary sur des métriques (latence, erreurs, distribution des scores : Lab 7),
  promotion progressive 10 → 50 → 100 %.
- Secrets managés (External Secrets : Lab 8), images signées et scannées (Lab 8), clusters
  et Argo CD séparés pour staging et prod, audit (qui a promu quoi, quand).

## 6. Captures d'écran

Argo CD (deux Applications Synced/Healthy), run `ct-…` dans KFP avec le rapport Markdown de
la gate, onglet Actions (3 jobs verts), `kubectl get isvc -n scoring` à 90/10.
