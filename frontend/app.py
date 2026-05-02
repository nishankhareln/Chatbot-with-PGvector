import json
import time
import uuid
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Advanced RAG", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #135bec;
                   text-align: center; margin-bottom: 1rem; }
    .source-box { background: #f0f2f6; padding: 0.8rem 1rem; border-radius: 0.5rem;
                  margin: 0.4rem 0; border-left: 4px solid #135bec; font-size: 0.9rem; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
             background: #e8f0ff; color: #135bec; font-size: 0.75rem; margin-right: 4px; }
    .badge-warn { background: #fff5d6; color: #8a6d00; }
    .diag { color: #666; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# -------------------- session --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None


def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def upload_files(files):
    payload = [("files", (f.name, f, f.type)) for f in files]
    r = requests.post(f"{API_URL}/upload", files=payload)
    return r.json() if r.status_code == 200 else {"_error": r.json().get("detail", r.text)}


def stream_chat(question: str, doc_id, top_k: int):
    """Yield (kind, payload). kind in {'meta','token','done'}."""
    body = {
        "question": question,
        "document_id": doc_id,
        "top_k": top_k,
        "session_id": st.session_state.session_id,
    }
    with requests.post(f"{API_URL}/chat/stream", json=body, stream=True, timeout=180) as r:
        if r.status_code != 200:
            yield ("error", r.text)
            return
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:"):].strip()
            if not data:
                continue
            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                continue
            t = evt.get("type")
            if t == "meta":
                yield ("meta", evt)
            elif t == "token":
                yield ("token", evt.get("text", ""))
            elif t == "done":
                yield ("done", None)


def render_sources(sources):
    if not sources:
        return
    for i, s in enumerate(sources, 1):
        rerank = f" • rerank {s['rerank_score']:.2f}" if s.get("rerank_score") else ""
        section = f" • <i>{s['section']}</i>" if s.get("section") else ""
        st.markdown(f"""
        <div class="source-box">
            <span class="badge">[{i}]</span>
            <b>{s['filename']}</b>
            <span class="diag">sim {s['similarity']:.2f} • rrf {s['rrf_score']:.3f}{rerank}{section}</span><br>
            {s['text']}
        </div>
        """, unsafe_allow_html=True)


def render_diagnostics(diag):
    if not diag:
        return
    bits = []
    if diag.get("rewritten_query") and diag["rewritten_query"] != diag.get("original_query"):
        bits.append(f"<b>Rewritten →</b> <i>{diag['rewritten_query']}</i>")
    if diag.get("hyde"):
        hyde_short = diag["hyde"][:200] + ("…" if len(diag["hyde"]) > 200 else "")
        bits.append(f"<b>HyDE →</b> <i>{hyde_short}</i>")
    if diag.get("candidates"):
        bits.append(f"<span class='badge'>{diag['candidates']} candidates</span>")
    if diag.get("reranked"):
        bits.append("<span class='badge'>reranked</span>")
    if bits:
        st.markdown("<div class='diag'>" + " &nbsp; ".join(bits) + "</div>", unsafe_allow_html=True)


# -------------------- main --------------------
st.markdown('<div class="main-header">📚 Advanced RAG</div>', unsafe_allow_html=True)

health = check_health()
if not health:
    st.error("Backend unreachable. Make sure the FastAPI container/process is running on :8000.")
    st.stop()

cols = st.columns([2, 1, 1])
with cols[0]:
    st.success("Backend connected")
with cols[1]:
    st.info(f"Reranker: {'on' if health.get('reranker') else 'off'}")
with cols[2]:
    st.info(f"Contextual: {'on' if health.get('contextual_retrieval') else 'off'}")


# -------------------- sidebar --------------------
with st.sidebar:
    st.header("📤 Upload")
    uploaded = st.file_uploader(
        "Drop docs or code here",
        type=["pdf", "txt", "md", "markdown",
              "py", "js", "jsx", "ts", "tsx", "go", "java", "rs",
              "cpp", "c", "rb", "php", "cs", "kt", "swift"],
        accept_multiple_files=True,
        help="Docs use hierarchical chunking; code uses AST chunking.",
    )
    if uploaded and st.button("Process all", type="primary", use_container_width=True):
        with st.spinner(f"Processing {len(uploaded)} file(s)…"):
            res = upload_files(uploaded)
        if "_error" in res:
            st.error(res["_error"])
        else:
            st.success(res["message"])
            for d in res["documents"]:
                if d["status"] == "success":
                    st.write(f"✅ **{d['filename']}** — {d['chunks_count']} chunks")
                else:
                    st.write(f"❌ **{d['filename']}** — {d.get('error')}")
            st.session_state.messages = []
            time.sleep(0.5)
            st.rerun()

    st.divider()

    st.subheader("📄 Documents")
    docs = []
    try:
        r = requests.get(f"{API_URL}/documents", timeout=5)
        if r.status_code == 200:
            docs = r.json().get("documents", [])
    except Exception as e:
        st.error(f"List failed: {e}")

    if docs:
        for d in docs:
            with st.expander(f"📄 {d['filename']}"):
                st.caption(f"id={d['id']} • {d['file_type'].upper()} • "
                           f"{d['file_size']/1024:.1f} KB • {d['uploaded_at'][:19]}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"[Download]({API_URL}/document/{d['id']}/download)")
                with c2:
                    if st.button("Delete", key=f"del_{d['id']}", use_container_width=True):
                        requests.delete(f"{API_URL}/document/{d['id']}")
                        st.rerun()
    else:
        st.caption("No documents yet")

    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Top-K chunks", 1, 15, 5)

    st.subheader("🎯 Scope")
    scope = st.radio("Search in:", ["All documents", "Specific document"])
    selected_doc_id = None
    if scope == "Specific document" and docs:
        opts = {f"{d['filename']} (id={d['id']})": d["id"] for d in docs}
        sel = st.selectbox("Document:", list(opts.keys()))
        selected_doc_id = opts[sel]
    st.session_state.selected_doc_id = selected_doc_id

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# -------------------- chat --------------------
st.header("💬 Chat with your knowledge base")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("diagnostics"):
                with st.expander("🔍 Retrieval diagnostics"):
                    render_diagnostics(msg["diagnostics"])
            if msg.get("sources"):
                with st.expander(f"📎 Sources ({len(msg['sources'])})"):
                    render_sources(msg["sources"])

if prompt := st.chat_input("Ask a question…"):
    if not docs:
        st.warning("Upload a document first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        diag_box = st.empty()
        body = st.empty()
        sources_box = st.container()

        sources = []
        diagnostics = {}
        full = ""

        try:
            for kind, payload in stream_chat(prompt, st.session_state.selected_doc_id, top_k):
                if kind == "meta":
                    sources = payload.get("sources", [])
                    diagnostics = payload.get("diagnostics", {})
                    with diag_box.container():
                        render_diagnostics(diagnostics)
                elif kind == "token":
                    full += payload
                    body.markdown(full + "▌")
                elif kind == "error":
                    body.error(f"Backend error: {payload}")
                    full = "Error from backend."
                elif kind == "done":
                    body.markdown(full)
        except Exception as e:
            body.error(f"Stream failed: {e}")
            full = f"Error: {e}"

        if sources:
            with sources_box.expander(f"📎 Sources ({len(sources)})"):
                render_sources(sources)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full,
            "sources": sources,
            "diagnostics": diagnostics,
        })

st.divider()
st.markdown(
    "<div style='text-align:center;color:#666;font-size:0.85rem'>"
    "Hybrid search (pgvector + BM25 RRF) · BGE reranker · contextual retrieval · "
    "AST chunking · streaming · multi-turn"
    "</div>",
    unsafe_allow_html=True,
)
