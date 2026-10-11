# Fichier : tests/test_k8s.py
"""Tests des manifests GitOps (k8s/) : ce qu'Argo CD appliquera, sans cluster.

Rendu par Kustomize (binaire kustomize, ou kubectl kustomize : présents sur les runners
GitHub) ; sans eux, un rendu Python minimal (namespace, images, patches) vérifie les mêmes
propriétés, pour que les tests tournent aussi sur un poste sans outils Kubernetes.
"""

import copy
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from pipelines import gitops

ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = ROOT / "k8s" / "overlays"


def _merge(target: dict, patch: dict) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def render_python(folder: Path) -> list[dict]:
    """Sous-ensemble de Kustomize utilisé par ce repo : resources, namespace, images, patches."""
    kust = yaml.safe_load((folder / "kustomization.yaml").read_text(encoding="utf-8"))
    docs = []
    for res in kust.get("resources", []):
        path = (folder / res).resolve()
        assert path.exists(), f"{folder}: ressource introuvable {res}"
        if path.is_dir():
            docs += render_python(path)
        else:
            docs += [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]
    for doc in docs:
        if "namespace" in kust:
            doc["metadata"]["namespace"] = kust["namespace"]
        for container in doc.get("spec", {}).get("predictor", {}).get("containers", []):
            for image in kust.get("images", []):
                if container["image"] == image["name"]:
                    container["image"] = f"{image['newName']}:{image['newTag']}"
    for patch in kust.get("patches", []):
        if "path" in patch:  # patch stratégique : même kind et même nom que la cible
            body = yaml.safe_load((folder / patch["path"]).read_text(encoding="utf-8"))
            targets = [d for d in docs if (d["kind"], d["metadata"]["name"]) ==
                       (body["kind"], body["metadata"]["name"])]  # fmt: skip
            assert targets, f"{patch['path']} ne cible aucune ressource"
            for doc in targets:
                _merge(doc, {k: v for k, v in body.items() if k != "metadata"})
        else:  # patch JSON 6902 en ligne (op replace / add)
            sel = patch["target"]
            targets = [d for d in docs if d["kind"] == sel["kind"]
                       and d["metadata"]["name"] == sel["name"]]  # fmt: skip
            assert targets, f"patch en ligne : aucune cible {sel}"
            for doc in targets:
                for op in yaml.safe_load(patch["patch"]):
                    *parents, last = op["path"].strip("/").split("/")
                    node = doc
                    for key in parents:
                        node = node[key]
                    assert op["op"] in ("replace", "add") and (op["op"] == "add" or last in node)
                    node[last] = op["value"]
    return docs


def render(env: str) -> list[dict]:
    folder = OVERLAYS / env
    if shutil.which("kustomize"):
        cmd = ["kustomize", "build", str(folder)]
    elif shutil.which("kubectl"):
        cmd = ["kubectl", "kustomize", str(folder)]
    else:
        return render_python(folder)
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return [d for d in yaml.safe_load_all(out) if d]


@pytest.mark.parametrize("env", ["staging", "prod"])
def test_overlay(env):
    docs = render(env)
    assert [(d["kind"], d["metadata"]["name"]) for d in docs] == [
        ("InferenceService", "scoring-model")
    ]
    isvc = docs[0]
    assert isvc["metadata"]["namespace"] == gitops.NAMESPACES[env]
    # mode Knative : le seul où canaryTrafficPercent existe (Lab 5) ; fixé à la création
    assert isvc["metadata"]["annotations"]["serving.kserve.io/deploymentMode"] == "Knative"
    predictor = isvc["spec"]["predictor"]
    name, tag = gitops.read_image(ROOT, env)
    assert predictor["containers"][0]["image"] == f"{name}:{tag}"  # images: bien appliqué
    if env == "prod":
        assert predictor["canaryTrafficPercent"] == gitops.read_canary(ROOT)
        assert 0 <= predictor["canaryTrafficPercent"] <= 100
        assert predictor["minReplicas"] >= 1
    else:
        assert "canaryTrafficPercent" not in predictor
        assert predictor["minReplicas"] == 0  # staging : scale-to-zero


def test_rendu_python_identique_a_kustomize():
    """Le rendu de secours dit la même chose que Kustomize (quand il est installé)."""
    if not (shutil.which("kustomize") or shutil.which("kubectl")):
        pytest.skip("ni kustomize ni kubectl sur ce poste")
    for env in ("staging", "prod"):
        assert render(env) == render_python(OVERLAYS / env)


def test_applications_argocd():
    apps = list(yaml.safe_load_all((ROOT / "k8s" / "argocd-app.yaml").read_text()))
    assert sorted(a["metadata"]["name"] for a in apps) == ["scoring-prod", "scoring-staging"]
    for app in apps:
        assert app["apiVersion"] == "argoproj.io/v1alpha1" and app["kind"] == "Application"
        env = app["metadata"]["name"].removeprefix("scoring-")
        assert app["spec"]["source"]["path"] == f"k8s/overlays/{env}"
        assert app["spec"]["source"]["targetRevision"] == "main"
        assert app["spec"]["destination"]["namespace"] == gitops.NAMESPACES[env]
        assert app["spec"]["syncPolicy"]["automated"] == {"prune": True, "selfHeal": True}


def test_gitops_modifie_sans_casser(tmp_path):
    """set_image / set_canary : seules les valeurs changent, commentaires conservés."""
    shutil.copytree(ROOT / "k8s", tmp_path / "k8s")
    before = (tmp_path / gitops.CANARY).read_text().splitlines()
    gitops.set_image(tmp_path, "prod", "ghcr.io/etudiant/mlops-cours-2026/scoring-api", "2.1.7")
    gitops.set_canary(tmp_path, 10)
    assert gitops.read_image(tmp_path, "prod") == (
        "ghcr.io/etudiant/mlops-cours-2026/scoring-api",
        "2.1.7",
    )
    assert gitops.read_canary(tmp_path) == 10
    after = (tmp_path / gitops.CANARY).read_text().splitlines()
    changed = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
    assert len(changed) == 1 and "canaryTrafficPercent" in changed[0][1]
    isvc = render_python(tmp_path / "k8s" / "overlays" / "prod")[0]
    assert isvc["spec"]["predictor"]["canaryTrafficPercent"] == 10
    with pytest.raises(SystemExit):
        gitops.set_canary(tmp_path, 120)
