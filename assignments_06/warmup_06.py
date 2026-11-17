"""Week 6 warmups: RAG concepts, retrieval, and LlamaIndex."""

from pathlib import Path
import string

from dotenv import load_dotenv


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---
# Concepts Q1
# Scenario A: RAG is the best choice. The policy PDFs are private source material and
# change every quarter, so retrieval can ground answers in the newest documents without
# retraining the model whenever a policy changes.
#
# Scenario B: Fine-tuning is the best choice. The startup has thousands of examples of
# the exact writing behavior it wants, so those examples can teach the model a consistent
# brand voice that is difficult to describe completely in one prompt.
#
# Scenario C: Prompt engineering is the best choice. A two-page report can fit directly
# in the prompt, and the analyst only needs to use it once, so building or training a
# larger system would add unnecessary work.

# Concepts Q2
# A confidently wrong answer is more harmful because its certain tone makes people less
# likely to verify it. For example, a confident hallucination about a medication dosage
# could cause someone to take an unsafe amount, while "I am not sure" signals that the
# answer needs confirmation from a qualified source.

# Concepts Q3
# Correct RAG pipeline order:
# 1. Extract text from source documents — read usable text from PDFs or other files.
# 2. Split text into chunks — divide long text into focused, retrievable sections.
# 3. Convert text chunks into embeddings — represent each chunk's meaning as a vector.
# 4. Receive the user's query — accept the question the system needs to answer.
# 5. Embed the user's query — represent the question in the same vector space.
# 6. Retrieve the most relevant chunks — find chunks closest to the query embedding.
# 7. Inject retrieved chunks into the prompt — provide the selected evidence as context.
# 8. Generate a response from the LLM — answer using the question and retrieved context.


# --- Keyword RAG ---
def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


DOCUMENTS = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


def run_keyword_questions():
    """Run all three local keyword-retrieval questions."""
    # Keyword Q1
    query = "What are your hours on weekends?"
    result = simple_keyword_retrieval(query, DOCUMENTS, verbose=True)
    print("Keyword Q1 selected document:", result[0][0])
    # The function unexpectedly selects loyalty.txt. hours.txt matches "weekends," but
    # hiring.txt and loyalty.txt both match the weak word "your." All three score 1, so
    # reverse tuple sorting breaks the tie by filename and puts loyalty.txt first. This
    # shows how exact overlap and poor tie-breaking can select the wrong document.

    # Keyword Q2
    query = "Do you have anything without caffeine?"
    result = simple_keyword_retrieval(query, DOCUMENTS, verbose=True)
    print("Keyword Q2 selected document:", result[0][0])
    # No document is selected because neither "without" nor "caffeine" appears in the
    # documents. This misses the useful menu content, so keyword RAG gets it wrong.
    # Semantic retrieval would do better because it can connect caffeine with drinks
    # such as espresso, lattes, and cold brew even without exact word overlap.

    # Keyword Q3
    # Prediction before running: loyalty.txt will be selected because rewards and a
    # loyalty program mean nearly the same thing.
    query = "How do I sign up for rewards?"
    result = simple_keyword_retrieval(query, DOCUMENTS, verbose=True)
    print("Keyword Q3 selected document:", result[0][0])
    # My prediction was wrong. No document was selected because loyalty.txt describes
    # earning and redeeming points but never uses "sign," "up," or "rewards." Semantic
    # retrieval could connect those related ideas even without exact matching words.


# --- Semantic RAG Concepts ---
# Semantic Q1
# A vector embedding is a numerical representation of a text's meaning. Similar ideas
# are placed near one another in the vector space even when their wording differs.
#
# The chunk with a cosine similarity of 0.85 is more relevant than the chunk scoring
# 0.30. The higher score means its embedding points in a much more similar direction to
# the query embedding, suggesting a stronger semantic relationship.
#
# Semantic search compares learned meaning rather than exact vocabulary. It can connect
# related concepts and synonyms, such as "rewards" and "loyalty points," even when the
# query and chunk share no words.

# Semantic Q2
# | Feature                 | Keyword RAG                    | Semantic RAG                 |
# |-------------------------|--------------------------------|------------------------------|
# | What is compared?       | Exact word overlap             | Embedding vectors            |
# | What is retrieved?      | Full document                  | Relevant text chunks         |
# | Can it handle synonyms? | No                             | Yes                          |
# | Storage format          | Plain text dictionary          | Vector store / index         |
# | Relevance score         | Overlapping keyword count      | Cosine similarity            |


# --- LlamaIndex ---
REPO_ROOT = Path(__file__).resolve().parents[1]
BRIGHTLEAF_DIR = REPO_ROOT / "lessons/06_AI_augmentation/resources/brightleaf_pdfs"


def print_sources(response, snippet_length=150):
    """Print source-node similarity scores and snippets."""
    for number, source in enumerate(response.source_nodes, start=1):
        score = "None" if source.score is None else f"{source.score:.4f}"
        snippet = source.node.get_content().replace("\n", " ")[:snippet_length]
        print(f"Source {number} similarity score: {score}")
        print(f"Source {number} text: {snippet}")


def run_llamaindex_questions():
    """Run the model-backed LlamaIndex questions after user approval."""
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    assert BRIGHTLEAF_DIR.exists(), f"Brightleaf directory not found: {BRIGHTLEAF_DIR}"
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

    documents = SimpleDirectoryReader(str(BRIGHTLEAF_DIR)).load_data()
    index = VectorStoreIndex.from_documents(documents)

    # LlamaIndex Q1
    query_engine = index.as_query_engine(similarity_top_k=3)
    questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?",
    ]
    for question in questions:
        response = query_engine.query(question)
        print(f"\nLlamaIndex Q1 question: {question}")
        print("LlamaIndex Q1 answer:", response)
        print_sources(response)
    # Expected observation to verify after running: benefits should retrieve the
    # employee-benefits PDF, while security should retrieve the security-policy PDF.
    # Strong matches should produce direct, specific answers. Any lower-ranked chunks
    # from unrelated PDFs should be noted as unexpected retrievals.

    # LlamaIndex Q2
    comparison_question = questions[0]
    for top_k in (1, 5):
        engine = index.as_query_engine(similarity_top_k=top_k)
        response = engine.query(comparison_question)
        print(f"\nLlamaIndex Q2 top_k={top_k} response:", response)
        print_sources(response)
    # Expected observation to verify after running: top_k=1 should be focused but may
    # omit details split across chunks. top_k=5 provides more context, but extra chunks
    # can add noise; more retrieved context is not automatically better.

    # LlamaIndex Q3
    difficult_question = "What is BrightLeaf's policy for employees working from another country?"
    difficult_response = query_engine.query(difficult_question)
    print(f"\nLlamaIndex Q3 question: {difficult_question}")
    print("LlamaIndex Q3 answer:", difficult_response)
    print_sources(difficult_response)
    # Expected observation to verify after running: the documents may not contain a
    # cross-border remote-work policy. A better system would set a similarity threshold,
    # state when evidence is missing, and direct the user to HR instead of guessing.

    # LlamaIndex Q4
    judge_llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    faithfulness = FaithfulnessEvaluator(llm=judge_llm)
    relevancy = RelevancyEvaluator(llm=judge_llm)
    evaluation_queries = [
        "What employee benefits does BrightLeaf offer?",
        "What is BrightLeaf's company policy on caring for pet dragons?",
    ]
    for query in evaluation_queries:
        response = query_engine.query(query)
        faithfulness_result = faithfulness.evaluate_response(query=query, response=response)
        relevancy_result = relevancy.evaluate_response(query=query, response=response)
        print(f"\nLlamaIndex Q4 query: {query}")
        print("Faithfulness score:", faithfulness_result.score)
        print("Relevancy score:", relevancy_result.score)
    # A faithfulness score of 1.0 means the response's claims are supported by the
    # retrieved context; 0.0 means at least one claim is not supported. Relevancy asks
    # whether the response and context address the query, while faithfulness asks whether
    # the answer stays grounded in the context. The unsupported-query scores may fall
    # because the documents do not contain a useful answer. LLM-as-a-judge uses another
    # model to assess natural-language quality because there may be many valid phrasings,
    # making exact-match accuracy too rigid for RAG responses.


def main():
    run_keyword_questions()
    run_llamaindex_questions()


if __name__ == "__main__":
    main()
