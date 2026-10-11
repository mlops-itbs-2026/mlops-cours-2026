# Rapport — Lab 5 : serving avec KServe

Binôme : corrigé enseignant · Dataset : `german-credit` · Modèle : `scoring-model` v1 ·
Seuil de décision servi : 0,3182 (param `decision_threshold` du champion)

Valeurs indicatives d'un laptop 8 cœurs / 16 Go (cluster kind à 3 nœuds, Locust sur le même
poste) : vos chiffres seront différents, c'est leur analyse qui est évaluée.

## 1. Résumé

L'image FastAPI + ONNX Runtime du Lab 4 est servie par KServe 0.21 sous la forme de
l'InferenceService `scoring-model` (namespace `scoring`) en mode **Knative** : Knative Serving
crée une révision par version, Kourier route les appels reçus sur `localhost:30080` d'après
l'en-tête `Host`, et l'autoscaler de Knative (KPA) ajuste de 0 à 5 pods sur le débit (cible
200 requêtes/s par pod). En fin de lab, la version 2.0.0 (route `/version` en plus) a été
promue à 100 % après un canary 90/10.

## 2. Benchmark de latence

Requêtes `POST /predict` d'une demande réelle de `test.csv`, attente de 0,1 à 0,3 s entre deux
requêtes d'un utilisateur ; 2 min (100 utilisateurs) puis 3 min (500 et 1 000).

| Run | Requêtes | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Erreurs (%) |
|---|---|---|---|---|---|---|
| run_100 | 52800 | 440.2 | 6 | 24 | 60 | 0.00 |
| run_500 | 161500 | 897.4 | 120 | 450 | 900 | 0.02 |
| run_1000 | 165200 | 917.8 | 650 | 1600 | 2800 | 0.85 |

Point de saturation : entre 500 et 1 000 utilisateurs, le débit plafonne (≈ 900 requêtes/s)
alors que la charge double : les requêtes supplémentaires ne font qu'attendre, le P99 est
multiplié par 3 et les premières erreurs (503 de la passerelle, délais dépassés) apparaissent.
Le RPS max soutenable avec un P99 < 500 ms est d'environ 800 requêtes/s pour 5 pods. À noter :
Locust, sur le même poste, consomme lui-même 1 à 2 cœurs et participe à la saturation.

## 3. Scale-up observé

| Run | Pods (max observé) | CPU moyen par pod | Temps pour atteindre ce nombre de pods |
|---|---|---|---|
| run_100 | 4 | ≈ 450m | ≈ 15 s |
| run_500 | 5 (maxReplicas) | ≈ 900m (limite : 1) | ≈ 20 s |
| run_1000 | 5 (maxReplicas) | ≈ 1000m, CPU bridé | immédiat (déjà 5) |

Avec une cible de 200 requêtes/s par pod, le KPA vise 140 requêtes/s (70 %) : 440 requêtes/s
demandent 4 pods. Le mode « panique » (fenêtre de 6 s) fait monter très vite au début de
chaque run. Scale-to-zero : 0 pod ≈ 2 min après la dernière requête (fenêtre de 60 s, délai de
réduction de 30 s, délai de grâce de 30 s) ; le démarrage à froid suivant a coûté ≈ 4 s
(pod planifié, image déjà présente sur le nœud, chargement du modèle, sondes de Knative).

## 4. Analyse canary

- Répartition observée sur 200 requêtes : 181 × 1.0.0 / 19 × 2.0.0.
- Attendu : 180 / 20. Avec une loi binomiale (n = 200, p = 0,1), l'écart-type est de
  √(200 × 0,1 × 0,9) ≈ 4,2 requêtes : 19 est à 0,25 écart-type de la valeur attendue, le split
  est conforme. Sur 100 requêtes seulement, observer entre 4 et 16 serait tout aussi normal.
- Décision : promotion à 100 %. La v2 ne change ni le modèle ni le seuil (mêmes
  probabilités, `model_version` identique), aucune erreur ni hausse de latence pendant le
  canary. En production, il faudrait des critères chiffrés (taux d'erreur, P95, distribution
  des scores) et une durée minimale d'observation : c'est ce qu'automatisera le Lab 6.

## 5. Points d'amélioration

- **Cible de l'autoscaler** : la mesurer plutôt que la deviner (un pod seul, charge croissante,
  RPS au moment où le P95 décroche), puis prendre ≈ 70 % de cette valeur.
- **Métrique** : la concurrence (requêtes en cours) réagit mieux que le RPS quand la latence
  varie ; le CPU est un mauvais signal pour une API très légère (le goulot est l'unique worker
  uvicorn).
- **maxReplicas et ressources** : relever `maxReplicas`, ou donner 2 workers uvicorn par pod ;
  ajuster les requêtes CPU pour que le scheduler ne sur-remplisse pas un nœud kind.
- **Démarrage à froid** : `minReplicas: 1` en journée, scale-to-zero la nuit ; image plus
  légère ; `initial-scale`.
- **Batching** : regrouper les demandes (bonus : × 10 en prédictions/s), ou Triton avec batching
  dynamique.

## 6. (Optionnel) Comparatif des serveurs et bonus batch

20 utilisateurs pendant 1 min, un pod par serveur (1 CPU max), même passerelle Kourier.

| Serveur | Protocole | P50 (ms) | P95 (ms) | P99 (ms) | Image | Mémoire du pod |
|---|---|---|---|---|---|---|
| FastAPI + ONNX Runtime (`scoring-model`) | V1 `/predict` | 4 | 9 | 15 | ≈ 330 Mo | ≈ 150 Mio |
| BentoML + scikit-learn (`scoring-bento`) | V1 `/predict` | 9 | 18 | 30 | ≈ 500 Mo | ≈ 300 Mio |
| Triton + ONNX Runtime (`scoring-triton`) | V2 | 5 | 10 | 16 | ≈ 15 Go | ≈ 400 Mio |

Les trois serveurs renvoient les mêmes probabilités (écart < 10⁻⁶ entre Triton et l'API du
Lab 4) ; seuls l'API FastAPI et BentoML décident eux-mêmes au seuil de 0,318, Triton laisse la
décision au client (`serving/v2_client.py`). Sur un si petit modèle, Triton n'apporte rien en
latence et coûte une image énorme : son intérêt (batching dynamique, GPU, multi-modèles)
apparaît sur des modèles lourds. BentoML paie le surcoût du pipeline scikit-learn (≈ 4 ms par
appel, voir le Lab 4).

Bonus batch (`BATCH_SIZE=50`, 20 utilisateurs) : ≈ 85 requêtes/s × 50 = ≈ 4 250 prédictions/s,
contre ≈ 95 en unitaire à charge égale : le coût fixe d'une requête HTTP domine largement celui
du modèle.
