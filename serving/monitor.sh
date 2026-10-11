#!/usr/bin/env bash
# Tableau de bord « live » de l'autoscaling, à lancer dans un terminal dédié pendant les
# tests de charge. Usage : ./serving/monitor.sh [isvc]   (défaut : scoring-model)
ISVC=${1:-scoring-model}
NS=scoring
SEL="serving.kserve.io/inferenceservice=${ISVC}"
while true; do
  clear
  echo "=== $(date +%T) — InferenceService ${NS}/${ISVC} ==="
  kubectl get isvc "$ISVC" -n "$NS"
  echo ""
  echo "=== Autoscaler Knative (KPA) : DESIREDSCALE = pods voulus, ACTUALSCALE = pods prêts ==="
  kubectl get kpa -n "$NS" -l "$SEL" 2>/dev/null
  echo ""
  echo "=== Pods ==="
  kubectl get pods -n "$NS" -l "$SEL"
  echo ""
  echo "=== CPU / mémoire (metrics-server) ==="
  kubectl top pods -n "$NS" -l "$SEL" --containers 2>/dev/null
  sleep 2
done
