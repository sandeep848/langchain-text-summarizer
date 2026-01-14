import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- env ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- ui + utils ---
import streamlit as st
import validators
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# --- llm ---
from langchain_groq import ChatGroq

try:
    from langchain.prompts import PromptTemplate
except Exception:
    from langchain_core.prompts import PromptTemplate

# --- youtube ---
YoutubeLoader = None
try:
    from langchain_community.document_loaders import YoutubeLoader
except Exception:
    try:
        from langchain.document_loaders import YoutubeLoader
    except Exception:
        YoutubeLoader = None

try:
    from yt_dlp import YoutubeDL
except Exception:
    YoutubeDL = None


# =========================================================
# Config
# =========================================================
APP_TITLE = "Advanced Summarizer (YouTube • Web • PDF)"
DEFAULT_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "llama-3.2-90b-vision-preview",
    # Optional: keep your requested ids selectable (availability depends on your Groq account)
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct-0905",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-prompt-guard-2-86m",
    "llama-3.3-70b-versatile",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

MAX_PDF_MB = 25
MAX_TOTAL_CHUNKS = 14          # hard cap to keep costs stable
CHUNK_TARGET_CHARS = 8500      # chunk size target (text-aware splitting)
REQUEST_TIMEOUT = 25


# =========================================================
# Helpers
# =========================================================
def is_youtube(u: str) -> bool:
    u = (u or "").lower()
    return "youtube.com" in u or "youtu.be" in u


def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def soft_normalize(text: str) -> str:
    """
    Keep paragraph breaks but normalize whitespace inside paragraphs.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # collapse crazy spacing
    text = re.sub(r"[ \t]+", " ", text)
    # collapse many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    text = soft_normalize(text)
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras


def chunk_by_paragraphs(text: str, target_chars: int = CHUNK_TARGET_CHARS) -> List[str]:
    """
    Better than naive slicing: builds chunks from paragraphs.
    """
    paras = split_into_paragraphs(text)
    if not paras:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    size = 0

    for p in paras:
        p_len = len(p) + 2
        if size + p_len <= target_chars or not buf:
            buf.append(p)
            size += p_len
        else:
            chunks.append("\n\n".join(buf))
            buf = [p]
            size = p_len

    if buf:
        chunks.append("\n\n".join(buf))

    # enforce safety cap
    if len(chunks) > MAX_TOTAL_CHUNKS:
        chunks = chunks[:MAX_TOTAL_CHUNKS]

    return chunks


def build_llm(api_key: str, model_id: str, temperature: float) -> ChatGroq:
    return ChatGroq(groq_api_key=api_key, model=model_id, temperature=temperature)


def llm_call_text(llm: ChatGroq, prompt: str) -> str:
    """
    Supports both invoke() and predict().
    """
    try:
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp)).strip()
    except Exception:
        return llm.predict(prompt).strip()


@dataclass
class ExtractResult:
    text: str
    info: Optional[str] = None
    source_title: Optional[str] = None


# =========================================================
# Extraction: YouTube
# =========================================================
@st.cache_data(show_spinner=False, ttl=60 * 30)
def youtube_metadata(url: str) -> Tuple[str, str]:
    if YoutubeDL is None:
        return "", ""
    with YoutubeDL({"quiet": True, "skip_download": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    title = info.get("title", "") or ""
    desc = info.get("description", "") or ""
    return title, desc


def extract_youtube(url: str) -> ExtractResult:
    title, desc = youtube_metadata(url)
    info_lines = []
    if title:
        info_lines.append(f"**Video title:** {title}")
    info_lines.append("**Transcript first**, fallback to description.")

    # transcript
    if YoutubeLoader is not None:
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
            docs = loader.load()
            transcript = "\n\n".join([d.page_content for d in docs if getattr(d, "page_content", "")]).strip()
            if transcript:
                return ExtractResult(text=transcript, info="\n\n".join(info_lines), source_title=title)
        except Exception:
            pass

    # fallback description
    if desc.strip():
        info_lines.append("**Transcript unavailable; using description.**")
        return ExtractResult(text=desc.strip(), info="\n\n".join(info_lines), source_title=title)

    info_lines.append("**Could not extract transcript/description.**")
    return ExtractResult(text="", info="\n\n".join(info_lines), source_title=title)


# =========================================================
# Extraction: Website (requests + bs4)
# =========================================================
@st.cache_data(show_spinner=False, ttl=60 * 15)
def fetch_url_html(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text


def extract_main_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # remove junk
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "form", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # try to focus "main" content if present
    main = soup.find("main")
    container = main if main else soup.body if soup.body else soup

    text = container.get_text(separator="\n")
    text = soft_normalize(text)

    # heuristic: drop very short lines
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    kept = [ln for ln in lines if len(ln) >= 35 or ln.endswith((".", "!", "?", ":"))]
    text2 = "\n".join(kept)
    text2 = soft_normalize(text2)
    return text2


def extract_website(url: str) -> ExtractResult:
    try:
        html = fetch_url_html(url)
        text = extract_main_text_from_html(html)
        if not text:
            return ExtractResult(text="", info="Website loaded, but no readable text was extracted.")
        return ExtractResult(text=text, info=None)
    except Exception as e:
        return ExtractResult(text="", info=f"Failed to load website content: {e}")


# =========================================================
# Extraction: PDF
# =========================================================
def extract_pdf(file_bytes: bytes) -> ExtractResult:
    try:
        reader = PdfReader(file_bytes)
        pages = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            t = t.strip()
            if t:
                pages.append(t)
        text = "\n\n".join(pages).strip()
        if not text:
            return ExtractResult(
                text="",
                info=(
                    "This PDF has **no extractable text** (likely scanned images). "
                    "If you want, I can add OCR support."
                ),
            )
        return ExtractResult(text=text, info=None)
    except Exception as e:
        return ExtractResult(text="", info=f"Failed to read PDF: {e}")


# =========================================================
# Summarization: Advanced Map → Reduce → Final
# =========================================================
def summarize_advanced(
    llm: ChatGroq,
    text: str,
    target_words: int,
    output_style: str,
) -> str:
    if not (text or "").strip():
        return "No text could be extracted."

    chunks = chunk_by_paragraphs(text)

    map_prompt = PromptTemplate(
        template=(
            "You are a precise summarizer.\n"
            "Summarize the chunk below into crisp notes.\n"
            "Rules:\n"
            "- Keep important facts, numbers, names.\n"
            "- Remove fluff.\n"
            "- Use short bullet-like lines.\n\n"
            "CHUNK:\n{text}\n"
        ),
        input_variables=["text"],
    )

    reduce_prompt = PromptTemplate(
        template=(
            "You are combining multiple chunk notes.\n"
            "Merge them into a clean, deduplicated set of notes.\n"
            "Keep only high-signal points.\n\n"
            "NOTES:\n{notes}\n"
        ),
        input_variables=["notes"],
    )

    if output_style == "Clean Summary":
        final_template = (
            "Write a final summary in about {target_words} words.\n"
            "- Structured with short sections.\n"
            "- No filler.\n"
            "- Preserve key facts.\n\n"
            "MERGED NOTES:\n{notes}\n"
        )
    elif output_style == "Key Takeaways":
        final_template = (
            "Produce:\n"
            "1) 8-12 key takeaways (bullets)\n"
            "2) 3 important quotes/claims (as paraphrases)\n"
            "3) 5 follow-up questions the reader should ask\n\n"
            "MERGED NOTES:\n{notes}\n"
        )
    elif output_style == "Action Plan":
        final_template = (
            "Create an action plan:\n"
            "- Goals\n"
            "- Steps\n"
            "- Risks\n"
            "- Timeline (high-level)\n"
            "- Success metrics\n\n"
            "MERGED NOTES:\n{notes}\n"
        )
    else:  # "Q&A"
        final_template = (
            "Generate a Q&A:\n"
            "- 10 questions a user might ask\n"
            "- Clear answers grounded in the content\n\n"
            "MERGED NOTES:\n{notes}\n"
        )

    final_prompt = PromptTemplate(
        template=final_template,
        input_variables=["notes", "target_words"] if "{target_words}" in final_template else ["notes"],
    )

    # MAP
    notes_list: List[str] = []
    for c in chunks:
        notes_list.append(llm_call_text(llm, map_prompt.format(text=c)))

    merged_notes = "\n\n".join(notes_list).strip()

    # REDUCE (can reduce twice if needed)
    merged_notes = llm_call_text(llm, reduce_prompt.format(notes=merged_notes))
    if len(merged_notes) > 14000:
        merged_notes = llm_call_text(llm, reduce_prompt.format(notes=merged_notes))

    # FINAL
    if "target_words" in final_prompt.input_variables:
        return llm_call_text(llm, final_prompt.format(notes=merged_notes, target_words=target_words))
    return llm_call_text(llm, final_prompt.format(notes=merged_notes))


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="centered")
st.title(APP_TITLE)
st.caption("Fast, reliable summarization with a Map→Reduce pipeline. Works best with accessible text sources.")

with st.sidebar:
    st.header("Model & Output")

    groq_api_key = st.text_input(
        "GROQ API KEY",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Put GROQ_API_KEY in .env or paste it here.",
    )

    model_choice = st.selectbox("Model", DEFAULT_MODELS + ["(custom)"], index=0)
    custom_model = ""
    if model_choice == "(custom)":
        custom_model = st.text_input("Custom model id", value="llama-3.1-8b-instant")

    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    target_words = st.slider("Target length (words)", 80, 1200, 280, 20)

    output_style = st.selectbox(
        "Output format",
        ["Clean Summary", "Key Takeaways", "Action Plan", "Q&A"],
        index=0,
    )

    show_extracted = st.checkbox("Show extracted text (debug)", value=False)

tabs = st.tabs(["URL (YouTube / Website)", "Upload PDF", "Paste Text"])

# --- URL tab ---
with tabs[0]:
    url = st.text_input("Enter a URL:", placeholder="https://...", label_visibility="collapsed")
    run_url = st.button("Summarize URL", type="primary")

# --- PDF tab ---
with tabs[1]:
    pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    run_pdf = st.button("Summarize PDF", type="primary")

# --- Text tab ---
with tabs[2]:
    pasted = st.text_area("Paste text here:", height=220)
    run_text = st.button("Summarize Text", type="primary")


def require_llm_or_stop() -> ChatGroq:
    if not groq_api_key.strip():
        st.error("Please enter your GROQ API KEY in the sidebar (or set GROQ_API_KEY in .env).")
        st.stop()
    model_id = custom_model.strip() if model_choice == "(custom)" else model_choice
    try:
        return build_llm(groq_api_key, model_id, temperature)
    except Exception as e:
        st.error("Failed to initialize the Groq model. Check your API key and model id.")
        st.exception(e)
        st.stop()


def render_downloads(summary: str):
    st.download_button("Download .txt", data=summary, file_name="summary.txt", mime="text/plain")
    st.download_button("Download .md", data=summary, file_name="summary.md", mime="text/markdown")


def run_pipeline(extracted: ExtractResult):
    if extracted.info:
        st.info(extracted.info)

    if show_extracted and extracted.text:
        with st.expander("Extracted text"):
            st.write(extracted.text)

    llm = require_llm_or_stop()

    with st.spinner("Summarizing (Map → Reduce → Final)..."):
        start = time.time()
        summary = summarize_advanced(llm, extracted.text, target_words, output_style)
        elapsed = time.time() - start

    st.success(f"Done in {elapsed:.1f}s")
    st.write(summary)
    render_downloads(summary)


# --- Run actions ---
if run_url:
    if not url.strip():
        st.error("Please enter a URL.")
        st.stop()
    if not validators.url(url):
        st.error("That doesn't look like a valid URL. Please paste a full URL starting with http(s).")
        st.stop()

    with st.spinner("Extracting content..."):
        extracted = extract_youtube(url) if is_youtube(url) else extract_website(url)
    run_pipeline(extracted)

if run_pdf:
    if pdf is None:
        st.error("Please upload a PDF first.")
        st.stop()

    # file size guard
    pdf_bytes = pdf.getvalue()
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_MB:
        st.error(f"PDF is too large ({size_mb:.1f} MB). Max allowed is {MAX_PDF_MB} MB.")
        st.stop()

    with st.spinner("Reading PDF..."):
        extracted = extract_pdf(pdf_bytes)
    run_pipeline(extracted)

if run_text:
    if not (pasted or "").strip():
        st.error("Please paste some text first.")
        st.stop()
    run_pipeline(ExtractResult(text=pasted.strip(), info=None))
