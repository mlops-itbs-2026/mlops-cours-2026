# Rapport — Lab 7 : monitoring du modèle de scoring

> Corrigé. Les chiffres de drift, de taux de refus et d'analyse de canary sont ceux mesurés en
> préparation du lab ; les horodatages et identifiants sont à remplacer par les vôtres.

## 1. Architecture déployée

```text
            trafic (monitoring/traffic.py)
                       │  Host: scoring-model-predictor.scoring.example.com
                       ▼
 Kourier :30080 ──► InferenceService scoring-model (Knative, révisions 0000N, canary)
                       │ /metrics (port 8000)            │ stdout : 1 ligne JSON / prédiction
          PodMonitor scoring-api                    Grafana Alloy (API Kubernetes)
          (+ labels revision, isvc)                      │
                       ▼                                 ▼
                  Prometheus ◄── Pushgateway ◄──── CronJob scoring-drift (Evidently, PSI)
                   │      │                          ▲ référence : train.csv versionné (DVC, S3)
     PrometheusRule│      │                          └─ courant : demandes lues dans Loki
                   ▼      ▼                                                  ▲
           Alertmanager  Grafana (scoring-overview) ◄──────────── Loki ──────┘
                   │
                   ▼ AlertmanagerConfig scoring-slack → Slack / webhook.site

 Poste : calc_drift.py + webhook CT (Lab 6, :8095) → KFP ct-pipeline → gate → commit canary
 Git (k8s/overlays/prod) → Argo CD → ISVC ; canary_analysis.py (Prometheus) → commit suivant
```

## 2. Instrumentation et dashboard

| Métrique | Type | Labels | Question |
|---|---|---|---|
| `http_requests_total` | Counter | method, handler, code | trafic, erreurs 5xx, requêtes rejetées (422) |
| `ml_inference_latency_seconds` | Histogram | — | latence P95/P99 de `/predict` |
| `ml_predictions_total` | Counter | model, version, outcome | volume et part de refus (`outcome="1"`) |
| `ml_prediction_score` | Histogram | — | distribution des scores |
| `ml_predictions_by_group_total` | Counter | attribute, group, outcome | refus par groupe sensible (âge, statut, nationalité) |
| `ml_model_info` | Info | model, model_version, api_version, threshold | version et seuil servis par révision |

Captures : `scoring-overview` en charge normale (≈ 20 req/s, P95 ≈ 5 ms, refus ≈ 63 %, PSI
tous < 0,15) puis pendant le drift (refus ≈ 78 %, heatmap des scores décalée vers le haut, PSI
de `age` et `credit_amount` en rouge, refus des moins de 25 ans en hausse).

## 3. Alertes

| Alerte | Condition (PromQL) | Sévérité | Runbook |
|---|---|---|---|
| ScoringHighLatency | `scoring:latency_p95:5m{namespace="scoring"} > 0.2` pendant 5 min | warning | `monitoring/runbooks.md#scoringhighlatency` |
| ScoringErrorBudgetBurn | ratio 5xx > 14,4 × 0,5 % sur 1 h et 5 min, ou > 6 × 0,5 % sur 6 h et 30 min | critical | `#scoringerrorbudgetburn` |
| ScoringDataDrift | `ml_drift_detected == 1` (PSI > 0,25) pendant 5 min | warning | `#scoringdatadrift` |
| ScoringPositiveRateShift | écart du taux de refus 5 min / 6 h > 10 points | warning | `#scoringpositiverateshift` |
| ScoringNoReplicaAvailable | aucun pod prêt en production pendant 2 min | critical | `#scoringnoreplicaavailable` |
| ScoringDriftJobStale | pas de publication du job de drift depuis 45 min | warning | `#scoringdriftjobstale` |

Burn rate plutôt que seuil brut : un pic d'erreurs d'une minute ne consomme presque rien du
budget mensuel du SLO (0,5 %) et ne mérite pas de page ; un burn rate de 14,4 sur 1 h en
consomme 2 % en une heure. La seconde fenêtre (5 min) fait retomber l'alerte dès la fin de
l'incident.

## 4. Démo drift → alerte → ré-entraînement

| Horodatage | Événement |
|---|---|
| T0 | début du trafic dérivé (`--drift age=0.6 --drift credit_amount=2.5`), refus 63 % → 78 % |
| T0 + 6 min | job `manual-run-2` : `age` PSI 4,19, `credit_amount` PSI 1,44, 2/20 variables en dérive |
| T0 + 7 min | `ScoringDataDrift` (age, credit_amount) et `ScoringPositiveRateShift` en Pending |
| T0 + 12 min | alertes Firing, notification reçue sur webhook.site (`[FIRING] ScoringDataDrift`) |
| T0 + 13 min | `calc_drift.py` depuis le poste → webhook CT : HTTP 202, run KFP `ct-pipeline` lancé |

## 5. Analyse du canary

Canary du modèle ré-entraîné (révision `00005`) contre la stable (`00004`), fenêtre de 5 min :
observabilité, volume (1 161 / 339 requêtes), erreurs (0 %), latence (P95 4,8 ms des deux
côtés), scores (PSI 0,057) et décisions (60,8 % contre 62,2 %) passent : décision `CONTINUE`
(palier 50 %) puis `PROMOTE`, appliquées par `canary_analysis.py --apply --push`.

Canary défectueux (image du Lab 5, sans `/metrics`) : `observabilité` en échec (`up=0`),
décision `ROLLBACK`, commit `gitops(prod): canary 0%`, trafic revenu à 100 % sur la stable.
Le cas « seuil erroné » (mêmes scores, refus 85,5 % contre 62,2 %) est arrêté par le
garde-fou `décisions`.

## 6. Améliorations à prévoir

- Garder chaque palier du canary plusieurs heures et calculer un intervalle de confiance sur
  l'écart de taux de refus, au lieu d'un seuil fixe de 10 points.
- Faire tourner le service de ré-entraînement dans le cluster et le brancher directement sur
  Alertmanager (`webhookConfigs`), avec une gate humaine pour la production.
- Joindre les labels réels (remboursement) quand ils arrivent pour suivre la performance et
  le coût métier, pas seulement des indicateurs avancés.
- Dashboards par équipe (exploitation, data science, conformité) et alertes d'équité par
  groupe sensible, en lien avec le Lab 8.
- Pseudonymiser le journal des prédictions et le stocker hors de Loki (data lake) si la
  rétention doit dépasser 72 h.
