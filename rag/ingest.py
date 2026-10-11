# Fichier : rag/ingest.py
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import COLLECTION, RAG_DIR, REPO_ROOT, get_embeddings, qdrant_location

parser = argparse.ArgumentParser(description="Indexe des documents dans Qdrant")
parser.add_argument(
    "--with-labs", action="store_true", help="ajoute les énoncés labs/*/README.md du fork"
)
parser.add_argument(
    "--project",
    action="store_true",
    help="indexe la doc de VOTRE projet (README, model card, docs/ et ADR)",
)
args = parser.parse_args()

# 0. Choisir les fichiers à indexer
if args.project:
    files = [REPO_ROOT / "README.md", REPO_ROOT / "dataset.yaml"]
    files += sorted(REPO_ROOT.glob("model-card/*.md")) + sorted(REPO_ROOT.glob("docs/**/*.md"))
else:
    files = sorted((RAG_DIR / "docs").glob("*.pdf")) + sorted((RAG_DIR / "docs").glob("*.md"))
    if args.with_labs:
        files += sorted(REPO_ROOT.glob("labs/*/README.md"))
files = [f for f in files if f.is_file()]

# 1. Charger les documents : 1 Document par page de PDF, 1 par fichier texte
docs = []
for f in files:
    print(f"Loading {f.relative_to(REPO_ROOT)}...")
    if f.suffix == ".pdf":
        docs.extend(PyPDFLoader(str(f)).load())
    else:
        docs.extend(TextLoader(str(f), encoding="utf-8").load())
print(f"Total: {len(files)} fichiers, {len(docs)} pages/documents chargés")
if not docs:
    raise SystemExit("Aucun document : copiez des PDF dans rag/docs/ ou utilisez --with-labs")

# 2. Découper en chunks de ~1 000 caractères (~250 tokens) avec un overlap de 150
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(docs)
for c in chunks:  # chemin relatif au repo : plus lisible dans les sources et dans Phoenix
    c.metadata["source"] = str(Path(c.metadata["source"]).resolve().relative_to(REPO_ROOT))
print(f"Chunks générés: {len(chunks)}")

# 3. Embeddings bge-small-en-v1.5 (même modèle pour l'ingestion ET les requêtes !)
embeddings = get_embeddings()

# 4. Upsert dans Qdrant (la collection est recréée à chaque ingestion)
vstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION,
    force_recreate=True,
    **qdrant_location(),
)
print(f"Ingested {len(chunks)} chunks into Qdrant (collection {COLLECTION})")
