import os
import re
import json
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Set, Optional, Tuple

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader


# ----------------------------
# Page Config (MUST be first st.* call)
# ----------------------------
st.set_page_config(
    page_title="Resume–Job Matcher (AI)",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume–Job Matcher (AI)")
st.caption(
    "Upload a resume PDF + paste a Job Description → match score, skill gap, improvement plan, bullets + sample resume."
)


# ----------------------------
# Environment helpers
# ----------------------------
def is_streamlit_cloud() -> bool:
    return os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud"


def get_backend_label() -> str:
    backend = os.getenv("EMBEDDINGS_BACKEND", "").strip().lower()
    if is_streamlit_cloud():
        return "Cloud (HF embeddings • no Ollama)"
    if backend == "ollama":
        return "Local (Ollama embeddings + Ollama LLM)"
    return "Local (HF embeddings + Ollama LLM)"


st.caption(
    f"Runtime env: {os.getenv('STREAMLIT_RUNTIME_ENV')} | "
    f"EMBEDDINGS_BACKEND: {os.getenv('EMBEDDINGS_BACKEND')} | "
    f"Mode: {get_backend_label()}"
)


# ----------------------------
# Helpers
# ----------------------------
def load_pdf(file) -> List:
    """Load uploaded PDF into LangChain Documents."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        path = tmp.name

    loader = PyPDFLoader(path)
    docs = loader.load()

    try:
        os.remove(path)
    except OSError:
        pass

    for d in docs:
        d.metadata["source_file"] = getattr(file, "name", "resume.pdf")

    return docs


def chunk_docs(docs: List, chunk_size: int = 800, chunk_overlap: int = 120) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


@st.cache_resource(show_spinner=False)
def _hf_embeddings_cached():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings():
    """
    Cloud-safe default: HuggingFace embeddings
    Optional local: Ollama embeddings (only if EMBEDDINGS_BACKEND=ollama)
    """
    backend = os.getenv("EMBEDDINGS_BACKEND", "").strip().lower()

    # Cloud OR default -> HuggingFace embeddings
    if is_streamlit_cloud() or backend in ("", "hf", "huggingface"):
        return _hf_embeddings_cached()

    # Local Ollama embeddings only if explicitly chosen
    if backend == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model="nomic-embed-text")
        except Exception:
            st.warning("Ollama embeddings selected but unavailable. Falling back to HuggingFace.")
            return _hf_embeddings_cached()

    # Fallback
    return _hf_embeddings_cached()


@st.cache_resource(show_spinner=False)
def _faiss_from_texts_cached(texts: Tuple[str, ...], metadatas: Tuple[Tuple[Tuple[str, Any], ...], ...]):
    """
    Cached FAISS builder using texts + metadatas (hashable) as key.
    """
    embeddings = get_embeddings()
    docs = []
    # Reconstruct minimal docs format expected by FAISS.from_documents
    # Use langchain_core.documents.Document without importing it at top (keeps imports lighter)
    from langchain_core.documents import Document
    for t, md_items in zip(texts, metadatas):
        docs.append(Document(page_content=t, metadata=dict(md_items)))
    return FAISS.from_documents(docs, embeddings)


def build_vectorstore(chunks):
    # Make chunks hashable for cache key
    texts = tuple([c.page_content for c in chunks])
    metadatas = tuple([tuple(sorted(c.metadata.items())) for c in chunks])
    return _faiss_from_texts_cached(texts, metadatas)


def extract_skills_from_text(text: str) -> Set[str]:
    """Simple offline keyword-based skill extraction. Expand anytime."""
    skills = [
        # core
        "python", "sql", "excel",
        "pandas", "numpy", "scikit-learn", "sklearn",
        "statistics", "probability", "data cleaning", "feature engineering",
        # ml/dl
        "pytorch", "tensorflow", "keras",
        "classification", "regression", "clustering", "xgboost", "lightgbm",
        # nlp/genai
        "nlp", "transformers", "bert", "llm", "rag", "langchain", "prompt engineering",
        "faiss", "chromadb", "vector database", "embeddings",
        # web/apps
        "streamlit", "fastapi", "flask",
        # engineering
        "git", "github", "docker", "linux", "rest api",
        # cloud/mlops
        "aws", "gcp", "azure",
        "mlops", "ci/cd", "airflow",
    ]

    t = text.lower()
    found: Set[str] = set()
    for s in skills:
        if " " in s or "/" in s:
            if s in t:
                found.add(s)
        else:
            if re.search(rf"\b{re.escape(s)}\b", t):
                found.add(s)
    return found


def safe_json_parse(text: str) -> Dict[str, Any]:
    """Parse JSON safely; extract first JSON object if extra text exists."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def build_skill_gap_rows(jd_skills: List[str], resume_skills: List[str]) -> List[Dict[str, str]]:
    rows = []
    resume_set = set(resume_skills)
    for skill in sorted(set(jd_skills)):
        present = skill in resume_set
        rows.append(
            {
                "Skill": skill,
                "In JD": "✅",
                "In Resume": "✅" if present else "❌",
                "Status": "OK" if present else "Missing",
            }
        )
    return rows


def compute_basic_match_score(jd_skills: List[str], resume_skills: List[str]) -> int:
    """Fallback score without LLM."""
    jd_set = set(jd_skills)
    if not jd_set:
        return 50
    overlap = len(jd_set.intersection(set(resume_skills)))
    score = int(round((overlap / max(1, len(jd_set))) * 100))
    return max(0, min(100, score))


def get_llm() -> Optional[object]:
    """
    Local Ollama LLM if available.
    Cloud: returns None.
    """
    if is_streamlit_cloud():
        return None

    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2", temperature=0)
    except Exception:
        return None


def retrieve_best_context(vectordb: FAISS, job_text: str, top_k: int) -> Tuple[List, str]:
    """
    Faster + better RAG:
    - Do MMR retrieval + similarity retrieval
    - Merge unique chunks
    """
    focused_query = "Find resume evidence matching these job requirements:\n" + job_text[:2000]

    retriever_mmr = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(15, top_k * 3), "lambda_mult": 0.5},
    )
    retriever_sim = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    chunks_mmr = retriever_mmr.invoke(focused_query)
    chunks_sim = retriever_sim.invoke(focused_query)

    seen = set()
    relevant = []
    for d in (chunks_mmr + chunks_sim):
        key = d.page_content.strip()[:200]
        if key not in seen:
            seen.add(key)
            relevant.append(d)

    relevant = relevant[: max(top_k, 6)]
    context = "\n\n".join([doc.page_content for doc in relevant])
    return relevant, context


# ----------------------------
# Prompts
# IMPORTANT: Escape { } with {{ }} so LangChain doesn't treat JSON as variables
# ----------------------------
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI recruiter assistant.\n"
            "Return ONLY valid JSON. No extra text.\n"
            "Output must be STRICT JSON (no markdown, no commentary).\n"
            "Use double quotes for all keys and strings. No trailing commas.\n"
            "Strictly follow this schema:\n"
            "{{\n"
            '  "match_score": number,\n'
            '  "strengths": [string, ...],\n'
            '  "missing_or_weak_skills": [string, ...],\n'
            '  "improvement_plan": [string, ...],\n'
            '  "ats_keywords_to_add": [string, ...],\n'
            '  "final_recommendation": string,\n'
            '  "evidence": [{{"claim": string, "resume_evidence": string}}]\n'
            "}}\n"
            "Rules:\n"
            "- Base evaluation ONLY on provided Resume Context + Job Description.\n"
            "- match_score MUST reflect skill overlap + evidence. If little evidence, keep score low.\n"
            "- Evidence must be copied/shortened from Resume Context. Do NOT invent.\n"
            "- Be conservative: if not clearly supported in resume, list it as missing.\n"
            "- Keep items short, actionable, recruiter-style.\n",
        ),
        (
            "human",
            "Resume Context (most relevant parts):\n{resume}\n\n"
            "Job Description:\n{job}\n\n"
            "Return JSON now:",
        ),
    ]
)

BULLETS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior recruiter and resume writer.\n"
            "Write 6 strong, ATS-friendly resume bullet points tailored to the job description.\n"
            "Rules:\n"
            "- Use action verbs.\n"
            "- Include tools/skills from JD.\n"
            "- Keep each bullet 1 line.\n"
            "- Do not invent company names or years.\n"
            "Return bullets ONLY.",
        ),
        (
            "human",
            "Resume Context:\n{resume}\n\n"
            "Job Description:\n{job}\n\n"
            "Write 6 improved resume bullets:",
        ),
    ]
)

RESUME_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert resume writer.\n"
            "Generate a SAMPLE ATS-optimized resume tailored to the job description.\n"
            "Assume candidate is a fresher/junior.\n"
            "Rules:\n"
            "- Do NOT invent company names, exact years, phone numbers, addresses.\n"
            "- Use placeholders where needed (e.g., [Your Name], [Email]).\n"
            "- Include: Summary, Skills, Projects (2-3), Education.\n"
            "- Make projects relevant to the job description.\n"
            "Return the resume in clean Markdown.",
        ),
        (
            "human",
            "Job Description:\n{job}\n\n"
            "Missing Skills to Improve:\n{missing}\n\n"
            "Generate the SAMPLE resume now:",
        ),
    ]
)


# ----------------------------
# UI State (MUST be above widgets)
# ----------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
.card {border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 16px 18px; background: rgba(255,255,255,0.02);}
.small {opacity: 0.8; font-size: 0.9rem;}
.badge {display: inline-block; padding: 6px 10px; border-radius: 999px; font-size: 0.82rem; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.04); margin-right: 8px;}
.kpi {font-size: 1.6rem; font-weight: 700; line-height: 1;}
.kpi-label {opacity: 0.75; font-size: 0.85rem;}
.stButton>button {border-radius: 12px; padding: 0.65rem 1.1rem; font-weight: 600;}
div[data-baseweb="tab-list"] button {border-radius: 12px;}
</style>
""",
    unsafe_allow_html=True,
)

# Header
top_left, top_right = st.columns([3, 2], vertical_alignment="center")
with top_left:
    st.markdown("## 📄 Resume–Job Matcher")
    st.markdown(f'<span class="badge">Mode: {get_backend_label()}</span>', unsafe_allow_html=True)
    if is_streamlit_cloud():
        st.warning(
            "Cloud mode: Ollama is not available here. The app runs in **basic mode** (no LLM). "
            "For full AI features, run locally."
        )
    st.markdown(
        '<div class="small">Upload a resume PDF + paste a Job Description → get match score, skill gap, bullets, and a sample resume.</div>',
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown(
        """
<div class="card">
  <div class="kpi">Fast + Practical</div>
  <div class="kpi-label">RAG retrieval • Skill gap • Reports</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.caption("Tune retrieval and chunking for best results.")
    chunk_size = st.slider("Chunk size", 300, 1200, 800, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 120, 10)
    top_k = st.slider("Top-k evidence", 3, 10, 6, 1)
    st.divider()
    st.markdown("### ✅ Tips")
    st.markdown(
        "- Use **PDF resume** (not image-only)\n"
        "- Paste full JD including **requirements**\n"
        "- If results feel off, increase **Top-k**"
    )

# Input Area
tab_input, tab_about = st.tabs(["🧾 Input", "ℹ️ About"])

with tab_input:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📄 Resume")
        resume_file = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"],
            label_visibility="collapsed",
            key=f"resume_uploader_{st.session_state.uploader_key}",
        )
        st.caption("Best: clean text-based PDF (ATS friendly).")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Job Description")
        job_text = st.text_area(
            "Paste job description",
            height=260,
            placeholder="Paste the job description here (requirements + responsibilities + tech stack)...",
            label_visibility="collapsed",
            key=f"job_text_{st.session_state.uploader_key}",
        )
        st.caption("Include must-have skills, responsibilities, and tools.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    a1, a2, a3 = st.columns([1.2, 1, 1], vertical_alignment="center")
    with a1:
        analyze_btn = st.button("🔍 Analyze Match", use_container_width=True)
    with a2:
        clear_btn = st.button("🧹 Clear Inputs", use_container_width=True)
    with a3:
        st.markdown(
            '<div class="small">Output includes: score • skill gap (table) • bullets • sample resume</div>',
            unsafe_allow_html=True,
        )

    if clear_btn:
        st.session_state.uploader_key += 1
        st.rerun()

with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### What this tool does")
    st.markdown(
        "- Builds a resume index with embeddings + FAISS\n"
        "- Retrieves best evidence using **MMR + Similarity** (better + faster answers)\n"
        "- Shows a **Skill Gap Table** (JD vs Resume)\n"
        "- Local mode: uses Ollama LLM for AI scoring + bullets + sample resume\n"
    )
    st.markdown("### Privacy")
    st.markdown("- Local mode: data stays on your machine.")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# Output placeholders
st.markdown("## 📊 Results")
st.caption("After you click **Analyze Match**, results will appear below.")

res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(
    ["✅ Summary", "📊 Skill Gap (Table)", "✍️ Bullets", "🧾 Sample Resume"]
)


# ----------------------------
# Logic
# ----------------------------
if analyze_btn:
    if not resume_file or not str(job_text).strip():
        st.warning("Please upload a resume PDF and paste a Job Description.")
    else:
        with st.spinner("Building resume index + retrieving relevant evidence..."):
            # Load + chunk
            resume_docs = load_pdf(resume_file)
            resume_chunks = chunk_docs(resume_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Build FAISS (cached)
            vectordb = build_vectorstore(resume_chunks)

            # Better retrieval (MMR + similarity)
            relevant_chunks, resume_context = retrieve_best_context(vectordb, str(job_text), top_k=top_k)

            # Add extra context (helps accuracy without huge slowdown)
            extra_resume = "\n\n".join([d.page_content for d in resume_chunks[:30]])
            resume_context = resume_context + "\n\n---\n\nAdditional Resume Context:\n" + extra_resume[:6000]

            # Skill signals
            jd_skills = sorted(list(extract_skills_from_text(str(job_text))))
            resume_skills = sorted(list(extract_skills_from_text(resume_context)))
            auto_missing = sorted(list(set(jd_skills) - set(resume_skills)))
            skill_gap_rows = build_skill_gap_rows(jd_skills, resume_skills)

        llm = get_llm()

        # Show mode explicitly (prevents confusion)
        st.caption("LLM mode: " + ("✅ Ollama" if llm is not None else "⚠️ Basic (no LLM)"))

        if llm is None:
            # --------- BASIC MODE (no LLM) ----------
            score = compute_basic_match_score(jd_skills, resume_skills)
            strengths = [f"Keyword match: {s}" for s in sorted(set(jd_skills).intersection(set(resume_skills)))][:12]
            missing = auto_missing[:12]
            plan = [f"Add evidence/projects showing: {m}" for m in auto_missing[:10]]
            keywords = jd_skills[:30]
            recommendation = "Basic mode (no LLM). Install/enable Ollama locally for AI scoring + rewriting."
            evidence = []

            improved_bullets = "\n".join(
                [
                    "- Built an NLP-based resume-to-JD matcher to compute similarity and identify missing skills.",
                    "- Implemented PDF parsing and chunking to improve retrieval quality from resumes.",
                    "- Used embeddings + FAISS vector search to retrieve the most relevant resume evidence for a JD.",
                    "- Designed a Streamlit UI to upload resumes, paste JDs, and export ATS-ready reports.",
                    "- Added skill-gap analysis and keyword overlap checks to improve JD alignment.",
                    "- Packaged the project with reproducible requirements and GitHub deployment workflow.",
                ]
            )

            sample_resume = (
                "# [Your Name]\n"
                "**Email:** [Email]  | **LinkedIn:** [Link] | **GitHub:** [Link]\n\n"
                "## Summary\n"
                "Junior AI/ML candidate with hands-on NLP projects, resume parsing, and similarity scoring.\n\n"
                "## Skills\n"
                f"{', '.join(sorted(set(jd_skills + resume_skills))[:25])}\n\n"
                "## Projects\n"
                "### Resume–Job Matcher (NLP)\n"
                "- Built a resume-to-JD matching app using embeddings and vector search.\n"
                "- Extracted resume text from PDFs, chunked documents, and retrieved relevant evidence.\n"
                "- Added skill-gap analysis and ATS keyword overlap reporting.\n\n"
                "## Education\n"
                "- [Your Degree] — [College/University]\n"
            )
        else:
            # --------- FULL MODE (Local Ollama) ----------
            chain = PROMPT | llm | StrOutputParser()

            with st.spinner("Scoring match + generating improvement plan..."):
                result = chain.invoke(
                    {
                        "resume": resume_context + "\n\nDetected Resume Skills: " + ", ".join(resume_skills),
                        "job": str(job_text) + "\n\nDetected JD Skills: " + ", ".join(jd_skills),
                    }
                )

            try:
                data = safe_json_parse(result)
            except Exception:
                st.error("The model returned invalid JSON. Try again (or shorten the JD).")
                st.code(result)
                st.stop()

            score = int(max(0, min(100, data.get("match_score", 0))))
            strengths = data.get("strengths", []) or []
            missing = data.get("missing_or_weak_skills", []) or []
            plan = data.get("improvement_plan", []) or []
            keywords = data.get("ats_keywords_to_add", []) or []
            recommendation = data.get("final_recommendation", "") or ""
            evidence = data.get("evidence", []) or []

            bullets_chain = BULLETS_PROMPT | llm | StrOutputParser()
            with st.spinner("Generating improved resume bullets..."):
                improved_bullets = bullets_chain.invoke({"resume": resume_context, "job": str(job_text)})

            resume_chain = RESUME_PROMPT | llm | StrOutputParser()
            with st.spinner("Generating a sample resume aligned to the JD..."):
                sample_resume = resume_chain.invoke(
                    {
                        "job": str(job_text),
                        "missing": ", ".join(auto_missing) if auto_missing else "None",
                    }
                )

        # ----------------------------
        # Render Results
        # ----------------------------
        with res_tab1:
            st.subheader("✅ Match Analysis")
            st.metric("Match Score", f"{score}/100")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ✅ Strengths")
                if strengths:
                    for s in strengths[:12]:
                        st.write(f"- {s}")
                else:
                    st.write("- (No strengths returned)")

            with c2:
                st.markdown("### ❌ Missing / Weak Skills")
                if missing:
                    for m in missing[:12]:
                        st.write(f"- {m}")
                else:
                    st.write("- (No missing skills returned)")

            st.markdown("### 🧩 Missing Skills (Auto-check from JD vs Resume)")
            st.write(
                ", ".join(auto_missing)
                if auto_missing
                else "Looks good — no obvious missing keywords from the built-in list."
            )

            st.markdown("### 🛠️ Improvement Plan")
            if plan:
                for p in plan[:12]:
                    st.write(f"- {p}")
            else:
                st.write("- (No improvement plan returned)")

            st.markdown("### 🧾 ATS Keywords to Add")
            st.write(", ".join(keywords[:30]) if keywords else "- (No keywords returned)")

            st.markdown("### 🎯 Final Recommendation")
            st.write(recommendation if recommendation else "(No recommendation returned)")

            if evidence:
                with st.expander("✅ Evidence (claims backed by resume context)"):
                    for item in evidence[:12]:
                        claim = (item.get("claim", "") or "").strip()
                        ev = (item.get("resume_evidence", "") or "").strip()
                        if claim:
                            st.write(f"• {claim}")
                        if ev:
                            st.caption(ev)
                        st.divider()

            with st.expander("🔎 Resume Evidence Used (Top Retrieved Chunks)"):
                for i, doc in enumerate(relevant_chunks, start=1):
                    src = doc.metadata.get("source_file", "resume.pdf")
                    page = doc.metadata.get("page", None)
                    page_str = f"{page + 1}" if isinstance(page, int) else "?"
                    st.markdown(f"**Chunk {i} — {src} (page {page_str})**")
                    st.write(doc.page_content.strip())
                    st.divider()

        with res_tab2:
            st.markdown("### 📊 Skill Gap Table (JD vs Resume)")
            if skill_gap_rows:
                st.dataframe(skill_gap_rows, use_container_width=True, hide_index=True)

                st.markdown("### ❌ Missing Skills (Only)")
                missing_only = [r["Skill"] for r in skill_gap_rows if r["Status"] == "Missing"]
                st.write(", ".join(missing_only) if missing_only else "None (from built-in skills list).")
            else:
                st.write("No skills detected from the built-in list. You can expand the skills list in code.")

        with res_tab3:
            st.markdown("### ✍️ JD-Optimized Resume Bullets (Suggestions)")
            st.write(improved_bullets)

        with res_tab4:
            st.markdown("## 🧾 Sample Resume (JD-Aligned Reference)")
            st.text_area("Sample Resume (Copy and edit with your real details)", sample_resume, height=420)
            st.caption(
                "⚠️ This is a generated SAMPLE resume for reference. Replace placeholders with your real information before applying."
            )
            st.download_button(
                label="⬇️ Download Sample Resume (MD)",
                data=sample_resume,
                file_name="sample_resume.md",
                mime="text/markdown",
            )

        # ----------------------------
        # Downloadable report (TXT)
        # ----------------------------
        report_lines = []
        report_lines.append("Resume–Job Match Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append(f"Match Score: {score}/100")
        report_lines.append("")
        report_lines.append("Strengths:")
        report_lines.extend([f"- {x}" for x in strengths])
        report_lines.append("")
        report_lines.append("Missing/Weak Skills:")
        report_lines.extend([f"- {x}" for x in missing])
        report_lines.append("")
        report_lines.append("Missing Skills (Auto-check):")
        report_lines.append(", ".join(auto_missing) if auto_missing else "None from built-in list.")
        report_lines.append("")
        report_lines.append("Skill Gap Table (JD vs Resume):")
        for row in skill_gap_rows:
            report_lines.append(f"- {row['Skill']}: {row['Status']}")
        report_lines.append("")
        report_lines.append("Improvement Plan:")
        report_lines.extend([f"- {x}" for x in plan])
        report_lines.append("")
        report_lines.append("ATS Keywords to Add:")
        report_lines.append(", ".join(keywords) if keywords else "None returned.")
        report_lines.append("")
        report_lines.append("Final Recommendation:")
        report_lines.append(recommendation if recommendation else "None returned.")
        report_lines.append("")
        report_lines.append("Improved Resume Bullets (Suggestions):")
        report_lines.append(str(improved_bullets).strip())
        report_lines.append("")
        report_lines.append("Evidence:")
        if evidence:
            for item in evidence:
                report_lines.append(f"- {item.get('claim','')}")
                ev = item.get("resume_evidence", "")
                if ev:
                    report_lines.append(f"  Evidence: {ev}")
        else:
            report_lines.append("None returned.")

        st.download_button(
            label="⬇️ Download Report (TXT)",
            data="\n".join(report_lines),
            file_name="resume_job_match_report.txt",
            mime="text/plain",
        )
