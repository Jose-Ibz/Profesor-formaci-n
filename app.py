# app_rag_multi_web.py — Chatbot Profesor (multi‑cursos + webs oficiales)
# -----------------------------------------------------------------------
# Añade INGESTA desde webs oficiales por curso y las combina con las transcripciones .txt.
# Mantiene RAG local con embeddings. Permite elegir cursos y re‑crawl.
#
# Requisitos:
#   pip install streamlit openai python-dotenv gdown numpy pyyaml requests beautifulsoup4 tldextract
#   .env: OPENAI_API_KEY=xxxxx
#   courses.yaml  -> Drive por curso (opcional)
#   web_sources.yaml -> Webs permitidas por curso (ver ejemplo al final)
#   Ejecuta: streamlit run app_rag_multi_web.py
# -----------------------------------------------------------------------

import os
import io
import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import deque

import yaml
import numpy as np
import streamlit as st
import gdown
import requests
import tldextract
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------
# Configuración
# --------------------------------------------------
AUTO_REFRESH = os.getenv("AUTO_REFRESH", "0") == "1"   # Si es 1: al arrancar descarga cursos/web y reindexa solo
HIDE_ADMIN   = os.getenv("HIDE_ADMIN", "0") == "1"     # Si es 1: oculta los botones de mantenimiento para alumnos
LEN_MIN_TEXT = int(os.getenv("LEN_MIN_TEXT", 80))  # antes 200; algunas webs (p.ej. Garmin) rinden poco HTML
VERBOSE_CRAWL = os.getenv("VERBOSE_CRAWL", "0") == "1"
load_dotenv()
client = OpenAI()

MODEL_CHAT = os.getenv("TEACHER_MODEL", "gpt-4o-mini")
MODEL_EMB = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

BASE_DIR = Path(os.getenv("BASE_DIR", "."))
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")          # data/<curso>/*.txt (transcripciones)
WEB_DIR = BASE_DIR / os.getenv("WEB_DIR", "data_web")         # data_web/<curso>/*.txt (web)
INDEX_DIR = BASE_DIR / os.getenv("INDEX_DIR", "index")
MANIFEST_PATH = BASE_DIR / os.getenv("COURSES_YAML", "courses.yaml")
WEB_MANIFEST_PATH = BASE_DIR / os.getenv("WEB_SOURCES_YAML", "web_sources.yaml")

for d in (DATA_DIR, WEB_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)
INDEX_PATH = INDEX_DIR / "embeddings_index.pkl"

# Chunking / búsqueda
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 6))
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", 9000))

# Crawl
DEFAULT_MAX_PAGES = int(os.getenv("MAX_PAGES", 12))
TIMEOUT = float(os.getenv("HTTP_TIMEOUT", 15))
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# --------------------------------------------------
# Utilidades de cursos/manifest
# --------------------------------------------------

def load_manifest() -> Dict:
    if not MANIFEST_PATH.exists():
        return {"courses": []}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"courses": []}


def load_web_manifest() -> Dict[str, Dict]:
    if not WEB_MANIFEST_PATH.exists():
        return {"courses": []}
    with open(WEB_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"courses": []}


def list_local_courses() -> List[str]:
    cursos = set()
    for root in (DATA_DIR, WEB_DIR):
        for p in sorted(root.glob("*")):
            if p.is_dir() and (any(p.glob("*.txt"))):
                cursos.add(p.name)
    return sorted(cursos)


def ensure_course_dir(root: Path, course_name: str) -> Path:
    p = root / course_name
    p.mkdir(parents=True, exist_ok=True)
    return p

# --------------------------------------------------
# Descarga de Drive (transcripciones)
# --------------------------------------------------

def download_course_from_drive(course_name: str, folder_id: str):
    target = ensure_course_dir(DATA_DIR, course_name)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    st.write(f"📥 Descargando '{course_name}' desde Drive…")
    gdown.download_folder(url=url, output=str(target), quiet=True, use_cookies=False)


def download_all_from_manifest():
    manifest = load_manifest()
    courses = manifest.get("courses", [])
    if not courses:
        st.warning("No hay cursos definidos en courses.yaml")
        return
    for c in courses:
        name = c.get("name")
        fid = c.get("folder_id")
        if not name or not fid:
            st.warning(f"Entrada inválida en manifest: {c}")
            continue
        download_course_from_drive(name, fid)
    st.success("✅ Descarga completada.")

# --------------------------------------------------
# Crawl de webs oficiales (por curso)
# --------------------------------------------------

def same_registrable_domain(a: str, b: str) -> bool:
    ea, eb = tldextract.extract(a), tldextract.extract(b)
    return (ea.domain, ea.suffix) == (eb.domain, eb.suffix)


def sanitize_filename(url: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", url)
    return name[:180]


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # eliminar scripts/estilos/nav
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return text


def fetch_url(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, allow_redirects=True)
        ctype = resp.headers.get("Content-Type", "")
        if VERBOSE_CRAWL:
            st.write(f"GET {url} -> {resp.status_code} {ctype}")
        if resp.status_code == 200 and "text/html" in ctype:
            return resp.text
    except Exception as e:
        if VERBOSE_CRAWL:
            st.write(f"Error GET {url}: {e}")
        return None
    return None
    return None


def crawl_course_web(course: str, seeds: List[str], max_pages: int = DEFAULT_MAX_PAGES, same_domain_only: bool = True) -> int:
    """Crawl BFS simple desde seeds. Guarda cada página en data_web/<curso>/*.txt"""
    if not seeds:
        return 0
    target = ensure_course_dir(WEB_DIR, course)
    visited: Set[str] = set()
    q = deque(seeds)
    saved = 0

    while q and saved < max_pages:
        url = q.popleft().strip()
        if not url or url in visited:
            continue
        visited.add(url)
        html = fetch_url(url)
        if not html:
            continue

        text = extract_text_from_html(html)
        if len(text) < LEN_MIN_TEXT:
            # Demasiado poco contenido (posible web muy dinámica). Se omite.
            continue

        fname = sanitize_filename(url) + ".txt"
        (target / fname).write_text(text, encoding="utf-8")
        saved += 1

        # extraer enlaces
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("/") and seeds[0].startswith("http"):
                    # convertir a absoluto
                    base = re.match(r"^https?://[^/]+", seeds[0])
                    if base:
                        href = base.group(0) + href
                if not href.startswith("http"):
                    continue
                if same_domain_only and not same_registrable_domain(href, seeds[0]):
                    continue
                if href not in visited and len(q) < 1000:
                    q.append(href)
        except Exception:
            pass

    return saved


def update_web_from_manifest(selected_courses: List[str]) -> int:
    webmf = load_web_manifest()
    courses = webmf.get("courses", [])
    index_by_name = {c.get("name"): c for c in courses}
    total_saved = 0
    for name in selected_courses:
        entry = index_by_name.get(name)
        if not entry:
            st.warning(f"No hay fuentes web para el curso: {name}")
            continue
        seeds = entry.get("seeds", [])
        maxp = int(entry.get("max_pages", DEFAULT_MAX_PAGES))
        saved = crawl_course_web(name, seeds, max_pages=maxp)
        total_saved += saved
        st.write(f"🌐 {name}: {saved} páginas guardadas")
    return total_saved

# --------------------------------------------------
# Lectura y chunking
# --------------------------------------------------

def read_all_texts() -> List[Tuple[str, str, str, str]]:
    """Devuelve [(course, source_type, filename, text)] de data/ y data_web/."""
    out: List[Tuple[str, str, str, str]] = []
    # Transcripciones
    for course_dir in sorted(DATA_DIR.glob("*")):
        if not course_dir.is_dir():
            continue
        course = course_dir.name
        for fp in sorted(course_dir.glob("*.txt")):
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
                if txt.strip():
                    out.append((course, "transcript", fp.name, txt))
            except Exception as e:
                st.warning(f"No se pudo leer {fp}: {e}")
    # Web
    for course_dir in sorted(WEB_DIR.glob("*")):
        if not course_dir.is_dir():
            continue
        course = course_dir.name
        for fp in sorted(course_dir.glob("*.txt")):
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
                if txt.strip():
                    out.append((course, "web", fp.name, txt))
            except Exception as e:
                st.warning(f"No se pudo leer {fp}: {e}")
    return out


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

# --------------------------------------------------
# Indexado / búsqueda
# --------------------------------------------------

def build_index():
    docs = read_all_texts()
    if not docs:
        raise RuntimeError("No hay .txt en data/<curso> ni en data_web/<curso>")

    chunks: List[str] = []
    metadatas: List[Dict] = []
    for course, source_type, fname, text in docs:
        for i, ch in enumerate(chunk_text(text)):
            chunks.append(ch)
            metadatas.append({
                "course": course,
                "source": fname,
                "source_type": source_type,
                "chunk_id": i,
            })

    st.write("🧮 Generando embeddings…")
    embs: List[List[float]] = []
    BATCH = 256
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        resp = client.embeddings.create(model=MODEL_EMB, input=batch)
        embs.extend([d.embedding for d in resp.data])

    M = np.array(embs, dtype=np.float32)
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-10)

    index = {"embeddings": M, "metadatas": metadatas, "chunks": chunks, "model": MODEL_EMB}
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    st.success(f"✅ Índice creado: {len(chunks)} fragmentos")


def load_index():
    if not INDEX_PATH.exists():
        return None
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def search_similar(query: str, selected_courses: Optional[List[str]] = None, top_k: int = TOP_K):
    index = load_index()
    if index is None:
        raise RuntimeError("No existe índice. Crea el índice primero.")

    q_emb = client.embeddings.create(model=index["model"], input=[query]).data[0].embedding
    q = np.array(q_emb, dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-10)

    M = index["embeddings"]
    sims = M @ q

    # Filtrado por curso
    mask = np.ones(len(index["metadatas"]), dtype=bool)
    if selected_courses:
        courses_set = set(selected_courses)
        mask = np.array([m["course"] in courses_set for m in index["metadatas"]], dtype=bool)

    idx_all = np.argsort(-sims)
    results = []
    for idx in idx_all:
        if not mask[idx]:
            continue
        meta = index["metadatas"][idx]
        results.append({"score": float(sims[idx]), **meta, "text": index["chunks"][idx]})
        if len(results) >= top_k:
            break
    return results


def build_context_snippet(passages: List[Dict], max_chars: int = MAX_PROMPT_CHARS) -> str:
    parts = []
    total = 0
    for p in passages:
        header = (
            f"\n[COURSE: {p['course']} | SOURCE: {p['source']} | TYPE: {p['source_type']} "
            f"| CHUNK: {p['chunk_id']} | SCORE: {p['score']:.3f}]\n"
        )
        body = p["text"].strip()
        add = header + body + "\n"
        if total + len(add) > max_chars:
            break
        parts.append(add)
        total += len(add)
    return "".join(parts)


def build_teacher_prompt(context_block: str, question: str, modo_profesor: bool = True) -> str:
    style = (
        "Explica de forma clara, paso a paso y pedagógica. Estructura con títulos cortos, listas y notas de 'Errores comunes'. "
        "Usa ejemplos y advierte si la información pudiera estar desactualizada en web."
        if modo_profesor else
        "Responde de forma concisa y directa, solo con datos de las fuentes."
    )
    return f"""
Actúa como profesor técnico náutico. Responde ÚNICAMENTE con el contenido del contexto (transcripciones y webs oficiales permitidas). 
Si no hay suficiente información, dilo.
{style}

--- CONTEXTO ---
{context_block}
--- FIN CONTEXTO ---

Pregunta del alumno:
❓ {question}

Prioriza transcripciones frente a web si hay conflicto. Cierra con **Fuentes** listando COURSE, SOURCE, TYPE y CHUNK.
"""

# --------------------------------------------------
# UI Streamlit
# --------------------------------------------------

st.set_page_config(page_title="Chatbot Profesor — Multi‑cursos + Web", page_icon="🧠")
st.title("🧠 Chatbot Profesor — Multi‑cursos + Web (local)")

# Auto‑actualización al arrancar (sin botones) para Streamlit Cloud
if AUTO_REFRESH and "boot_refreshed" not in st.session_state:
    try:
        # 1) Descargar transcripciones desde Drive (si hay manifest)
        download_all_from_manifest()
        # 2) Actualizar webs oficiales desde manifest
        webmf = load_web_manifest()
        names = [c.get("name") for c in webmf.get("courses", []) if c.get("name")]
        if names:
            update_web_from_manifest(names)
        # 3) Recrear índice
        build_index()
        st.session_state["boot_refreshed"] = True
        st.success("Contenido actualizado automáticamente (Drive + Web + Índice).")
    except Exception as e:
        st.warning(f"Auto‑actualización fallida: {e}")

if not HIDE_ADMIN:
    with st.expander("⚙️ Datos y estructura", expanded=True):
        colA, colB, colC, colD = st.columns(4)
        with colA:
            if st.button("📥 Descargar cursos (Drive/manifest)"):
                try:
                    download_all_from_manifest()
                except Exception as e:
                    st.error(f"Error en descarga: {e}")
        with colB:
            if st.button("🌐 Actualizar desde webs oficiales (manifest)"):
                try:
                    all_courses = [c for c in list_local_courses()]
                    saved = update_web_from_manifest(all_courses)
                    st.success(f"Guardadas {saved} páginas en data_web/")
                except Exception as e:
                    st.error(f"Error al actualizar webs: {e}")
        with colC:
            if st.button("🧱 (Re)crear índice"):
                try:
                    build_index()
                except Exception as e:
                    st.error(f"Error creando índice: {e}")
        with colD:
            st.caption(
    f"📂 Datos: {DATA_DIR.resolve()}\\n"
    f"🌐 Web: {WEB_DIR.resolve()}\\n"
    f"🗂 Índice: {INDEX_PATH.resolve()}"
)


        st.toggle("🔍 Ver logs de crawling (VERBOSE)", value=VERBOSE_CRAWL, key="verbose_toggle")
        if st.session_state.get("verbose_toggle") and not VERBOSE_CRAWL:
            st.info("Activa VERBOSE_CRAWL=1 en tu .env para ver los logs permanentes.")

    # Utilidad de test rápido de una sola URL
    with st.expander("🧪 Probar una URL concreta (debug)"):
        test_url = st.text_input("URL a probar, p. ej. https://support.garmin.com/es-ES/", key="url_test")
        if st.button("Probar descarga de URL") and test_url:
            html = fetch_url(test_url)
            if not html:
                st.error("No se pudo obtener HTML (posible bloqueo o contenido dinámico). Prueba con otra URL.")
            else:
                text = extract_text_from_html(html)
                st.write(f"Longitud de texto extraído: {len(text)} caracteres (mínimo guardado = {LEN_MIN_TEXT})")
                st.code(text[:1200] + ("…" if len(text) > 1200 else ""))
else:
    st.caption("🔒 Modo alumno: mantenimiento oculto. (Configurable con HIDE_ADMIN=0)")

st.divider()

all_courses = list_local_courses()
selected = st.multiselect("Elige uno o varios cursos", options=all_courses, default=all_courses)
modo_profesor = st.toggle("Modo profesor (explicaciones pedagógicas)", value=True)

question = st.text_input("📩 Escribe tu pregunta:")

if st.button("🤖 Responder"):
    if not question.strip():
        st.warning("Escribe una pregunta.")
    else:
        try:
            with st.spinner("Buscando en transcripciones + webs oficiales…"):
                passages = search_similar(question, selected_courses=selected, top_k=TOP_K)
                if not passages:
                    st.info("No se han encontrado pasajes relevantes. ¿Has creado el índice?")
                else:
                    context_block = build_context_snippet(passages, max_chars=MAX_PROMPT_CHARS)
                    prompt = build_teacher_prompt(context_block, question, modo_profesor)
                    chat = client.chat.completions.create(
                        model=MODEL_CHAT,
                        messages=[{"role": "system", "content": "Eres un asistente útil y preciso."}, {"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    answer = chat.choices[0].message.content
                    st.markdown("### 📢 Respuesta")
                    st.info(answer)

                    with st.expander("🔎 Pasajes usados (auditoría)"):
                        for p in passages:
                            st.markdown(
                                f"**{p['course']}** / {p['source']} — {p['source_type']} — CHUNK {p['chunk_id']} — score {p['score']:.3f}\n\n" +
                                p['text'][:1200] + ("…" if len(p['text']) > 1200 else "")
                            )
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.divider()
st.caption("MVP multi‑cursos + webs oficiales. Respeta robots, usa fuentes permitidas, y prioriza transcripciones si difieren.")

# --------------------------------------------------
# Ejemplo de web_sources.yaml (colócalo junto al .py)
# --------------------------------------------------
# courses:
#   - name: Sadira
#     seeds:
#       - https://www.sadira.es/
#       - https://www.sadira.es/productos/
#     max_pages: 15
#   - name: Garmin
#     seeds:
#       - https://www.garmin.com/es-ES/
#     max_pages: 10
#   - name: Raymarine
#     seeds:
#       - https://www.raymarine.com/es-es
#     max_pages: 10







