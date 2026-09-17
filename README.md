# Multi-Source Map-Reduce Summarizer

A long-form summarization pipeline for PDFs, web pages and YouTube transcripts. The project focuses on **content compression**, not conversational question answering.

## Distinct use case

The application converts long source material into one of several structured outputs:

- executive brief
- detailed study notes
- key claims and evidence
- action items
- short abstract

Large inputs are divided on paragraph boundaries. Each chunk is summarized independently and the intermediate summaries are reduced into a final document, keeping request size and cost bounded.

## Pipeline

~~~mermaid
flowchart LR
    A["PDF, webpage or video"] --> B["Source extraction"]
    B --> C["Paragraph-aware chunks"]
    C --> D["Map summaries"]
    D --> E["Reduce and deduplicate"]
    E --> F["Structured final summary"]
~~~

## Run

~~~bash
git clone https://github.com/sandeep848/langchain-text-summarizer.git
cd langchain-text-summarizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
~~~

## What makes the logic different

- multiple source extractors rather than a chat interface
- paragraph-aware batching instead of vector retrieval
- hard limits on file size, chunks and request time
- map-reduce synthesis for long contexts
- output-format selection based on the reader's goal

## Limitations

Web extraction depends on page structure, and YouTube summarization depends on transcript availability. Summaries can omit nuance and should not replace the original source for high-stakes interpretation.
