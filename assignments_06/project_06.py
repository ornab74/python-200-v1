"""Week 6 project: Groundwork Coffee Co. semantic RAG assistant."""

from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- Step 1: Setup ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "lessons/06_AI_augmentation/resources/groundwork_docs"
assert DOCS_DIR.exists(), f"Document directory not found: {DOCS_DIR}"


def print_top_source(response):
    """Print the highest-ranked source node for a response."""
    if not response.source_nodes:
        print("Top source: No source nodes returned.")
        return
    source = response.source_nodes[0]
    file_name = source.node.metadata.get("file_name", "Unknown file")
    score = "None" if source.score is None else f"{source.score:.4f}"
    snippet = source.node.get_content().replace("\n", " ")[:200]
    print("Top source document:", file_name)
    print("Top source similarity score:", score)
    print("Top source text:", snippet)


def print_all_sources(response):
    """Print all retrieved source nodes for the failure analysis."""
    for number, source in enumerate(response.source_nodes, start=1):
        file_name = source.node.metadata.get("file_name", "Unknown file")
        score = "None" if source.score is None else f"{source.score:.4f}"
        snippet = source.node.get_content().replace("\n", " ")[:200]
        print(f"Source {number} document:", file_name)
        print(f"Source {number} similarity score:", score)
        print(f"Source {number} text:", snippet)


def main():
    # --- Step 2: Load the Documents ---
    documents = SimpleDirectoryReader(str(DOCS_DIR)).load_data()
    print("Documents loaded:", len(documents))
    for document in documents:
        print("Loaded file:", document.metadata.get("file_name", "Unknown file"))

    # --- Step 3: Build the Index and Query Engine ---
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)
    print("Index built successfully. Ready to answer questions.")

    # --- Step 4: Query the Assistant ---
    questions = [
        "What are Groundwork's hours on weekends?",
        "Do you offer any dairy-free milk options?",
        "How does the loyalty program work?",
        "How did Groundwork Coffee get started?",
        "Do you offer catering or wholesale orders?",
    ]
    for question in questions:
        response = query_engine.query(question)
        print(f"\nQuestion: {question}")
        print("Answer:", response)
        print_top_source(response)
    # Expected observation to verify after running: the five answers should be specific
    # and grounded in faq.txt, menu.txt, our_story.txt, and wholesale_catering.txt. The
    # dairy-free question may retrieve seasonal_specials.txt as a lower-ranked source,
    # but menu.txt should contain the complete everyday milk options.

    # --- Step 5: Find a Failure ---
    failure_question = "Which Groundwork location is the quietest place to study on Tuesday?"
    failure_response = query_engine.query(failure_question)
    print(f"\nFailure-test question: {failure_question}")
    print("Failure-test answer:", failure_response)
    print_all_sources(failure_response)
    # I chose this question because the documents list locations and Wi-Fi but provide no
    # noise levels or location-specific atmosphere. Retrieval may find faq.txt because it
    # contains locations, but that evidence cannot establish which shop is quietest. If
    # the model still selects one confidently, it is guessing despite successful-looking
    # retrieval. I would add an instruction to say when the documents lack evidence and
    # use a minimum similarity threshold before sending context to the model.


# --- Step 6: Reflection ---
# The core LlamaIndex setup takes five lines: load the documents, set the embedding
# model, set the language model, build the VectorStoreIndex, and create a query engine.
# The keyword retriever required manual token cleanup, stopword removal, scoring, sorting,
# and result selection. Implementing chunking, embeddings, storage, and retrieval manually
# would take substantially more code. A framework reduces that plumbing and lets a developer
# spend more time testing retrieval quality and application behavior.
#
# A useful non-coffee-shop application would be an employee assistant grounded in a
# company's benefits, leave, security, and onboarding policies. Employees could find a
# cited answer without searching several long documents, while HR could update the source
# material without retraining a model.
#
# RAG cannot fully prevent a model from misreading relevant context or combining correct
# facts into an unsupported conclusion. Good retrieval improves the available evidence,
# but the generator can still hallucinate, omit qualifications, or answer with too much
# confidence. Important answers still need clear sources and human review.


if __name__ == "__main__":
    main()
