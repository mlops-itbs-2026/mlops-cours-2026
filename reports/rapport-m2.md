# Rapport comparatif — Kubeflow Pipelines vs Airflow (Lab 2)

<!-- Exemple de rapport rempli (corrigé enseignant). Les durées dépendent de la machine :
     ordres de grandeur observés sur un portable 8 vCPU / 16 Go, Docker Engine, kind 3 nœuds. -->

**Binôme :** exemple corrigé · **Date :** octobre 2026 · **Machine :** 8 vCPU, 16 Go,
Ubuntu 24.04 · **Dataset :** german-credit

## 1. Contexte

Le pipeline de credit scoring du Lab 1 (DVC) a été porté sur deux orchestrateurs. Il compte six
étapes : ingestion, validation du contrat de données, préparation (split stratifié),
entraînement d'une régression logistique tracée dans MLflow, évaluation au seuil de coût
minimal, enregistrement dans le Model Registry. Les deux versions exécutent la même image
`localhost:5001/scoring-pipeline:1.0.0` et les mêmes commandes `python -m src.<étape>` que
`dvc.yaml` : chaque step est un pod du cluster kind. Seule la façon de relier les étapes
change.

## 2. Résultats quantitatifs

| Mesure | KFP | Airflow |
|---|---|---|
| Installation de zéro (min) | 9 (pull des 14 images) | 5 (compose officiel) |
| Temps jusqu'au premier run réussi (min) | 35 (image + composants) | 15 (image déjà prête) |
| Durée d'un run complet (min) | 3,5 | 3 |
| Relance avec les mêmes paramètres (min) | 1,5 (3 steps en cache) | 3 (tout est relancé) |
| Hot-reload : modifier un step et voir le résultat (min) | 0,5 en local, 4 sur le cluster | 3 (rebuild image, relance) |
| Lignes de code propres à l'orchestrateur | 280 (composants + pipeline) | 70 (DAG) + 60 (S3, XCom) |
| Mémoire consommée par l'outil (Go, `docker stats`) | 2,4 (pods kubeflow) | 3,6 (7 conteneurs) |

## 3. Analyse qualitative

### Developer experience

KFP impose plus de cérémonie : décorateurs, types d'artefacts, compilation en YAML, upload.
En échange, la signature d'un composant documente ses entrées et sorties, et `kfp.local`
permet de tester tout le pipeline en 30 secondes sans cluster. Airflow se lit plus vite (un
opérateur par étape, `>>` pour l'ordre), mais rien n'est typé : une faute de frappe dans un
template Jinja n'apparaît qu'à l'exécution.

### Scaling

Dans les deux cas un step est un pod, avec ses propres requests et limits : le scaling
dépend du cluster, pas de l'outil. KFP rend le parallélisme naturel (deux tâches sans
dépendance partent ensemble) et le cache évite de refaire les étapes de données. Airflow en
Docker Compose ajoute un worker Celery qui doit rester en vie pendant chaque pod.

### Debugging

KFP affiche, pour chaque artefact, son producteur, ses métadonnées (md5 du fichier brut,
run MLflow) et ses métriques, avec une matrice de confusion : le lineage est à un clic.
Côté Airflow, les logs du pod remontent bien dans l'UI, mais pour retrouver un fichier
intermédiaire il faut connaître le préfixe S3 du run.

## 4. Recommandation

Pour une équipe déjà sur Kubernetes dont l'essentiel du travail est de l'entraînement de
modèles, KFP : artefacts typés, cache et lineage répondent directement aux questions
d'audit. Pour une organisation qui orchestre surtout des flux de données hétérogènes (ETL,
SQL, notifications), Airflow, en gardant la logique ML dans une image versionnée appelée par
le KubernetesPodOperator, comme ici. Condition commune : un code de projet indépendant de
l'orchestrateur, sinon la migration coûte une réécriture.

## 5. Limites de la comparaison

Un seul dataset de 1 000 lignes, un cluster kind sur un portable, un utilisateur : rien sur
la montée en charge réelle, le multi-utilisateur de Kubeflow, le KubernetesExecutor
d'Airflow, ni un Airflow installé par Helm. Les temps incluent des pulls d'images qui
disparaissent au deuxième essai, et notre expérience préalable de KFP au fil du lab biaise
la mesure du « premier pipeline ».
