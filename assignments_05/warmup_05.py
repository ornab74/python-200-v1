"""Week 5 warmups: Chat Completions and prompt engineering."""

import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"


def completion(messages, **kwargs):
    """Return a Chat Completions response using the course model."""
    return client.chat.completions.create(model=MODEL, messages=messages, **kwargs)


def ask(prompt, **kwargs):
    """Return only the text for a one-message completion."""
    response = completion([{"role": "user", "content": prompt}], **kwargs)
    return response.choices[0].message.content


def run_api_questions():
    # --- Completions API ---
    # API Q1
    response = completion(
        [{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
    )
    print("API Q1 response:", response.choices[0].message.content)
    print("API Q1 model:", response.model)
    print("API Q1 total tokens:", response.usage.total_tokens)

    # API Q2
    prompt = "Suggest a creative name for a data engineering consultancy."
    temperatures = [0, 0.7, 1.5]
    for temperature in temperatures:
        print(f"API Q2 (temperature={temperature}):", ask(prompt, temperature=temperature))
    # Higher temperatures generally produce more varied wording and ideas. I would use
    # temperature 0 when consistency and reproducibility matter most.

    # API Q3
    response = completion(
        [{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
        n=3,
        temperature=1.0,
    )
    for number, choice in enumerate(response.choices, start=1):
        print(f"API Q3 completion {number}:", choice.message.content)

    # API Q4
    print("API Q4:", ask("Explain how neural networks work.", max_tokens=15))
    # The answer stops after a small number of generated tokens, possibly mid-thought.
    # A token cap controls latency, cost, and unexpectedly long output in production.


def run_system_questions():
    # --- System Messages and Personas ---
    # System Q1
    question = "I don't understand what a list comprehension is."
    tutor_messages = [
        {
            "role": "system",
            "content": "You are a patient, encouraging Python tutor. Explain simply and end with encouragement.",
        },
        {"role": "user", "content": question},
    ]
    detective_messages = [
        {
            "role": "system",
            "content": "You are a concise noir detective. Explain Python using mystery metaphors in under 90 words.",
        },
        {"role": "user", "content": question},
    ]
    print("System Q1 - tutor:", completion(tutor_messages).choices[0].message.content)
    print("System Q1 - detective:", completion(detective_messages).choices[0].message.content)
    # The same question should produce different tone, imagery, length, and closing style
    # because each system message establishes different behavioral constraints.

    # System Q2
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Jordan and I'm learning Python."},
        {
            "role": "assistant",
            "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?",
        },
        {"role": "user", "content": "Can you remind me what my name is?"},
    ]
    print("System Q2:", completion(messages).choices[0].message.content)
    # The API is stateless, but Jordan's name is present in the history supplied with
    # this request, so the model can use it as context.


def sentiment_prompt(reviews, examples=()):
    example_text = "\n".join(
        f'Example:\nReview: "{review}"\nSentiment: {sentiment}'
        for review, sentiment in examples
    )
    review_text = "\n".join(f'{i}. "{review}"' for i, review in enumerate(reviews, 1))
    return f"""
Classify each review as positive, negative, or mixed.
Return exactly one line per review in the format: Review NUMBER: SENTIMENT
{example_text}

Reviews:
{review_text}
"""


def run_prompt_questions():
    # --- Prompt Engineering ---
    reviews = [
        "The onboarding process was smooth and the team was welcoming.",
        "The software crashes constantly and support never responds.",
        "Great price, but the documentation is nearly impossible to follow.",
    ]

    # Prompt Q1 - Zero-Shot
    print("Prompt Q1 - zero-shot:\n", ask(sentiment_prompt(reviews), temperature=0))

    # Prompt Q2 - One-Shot
    one_shot = [("Fast shipping but the item arrived damaged.", "mixed")]
    print("Prompt Q2 - one-shot:\n", ask(sentiment_prompt(reviews, one_shot), temperature=0))
    # The example makes the expected labels and line format more explicit, so the
    # response should be more consistent than the zero-shot version.

    # Prompt Q3 - Few-Shot
    few_shot = [
        ("Setup was quick and everything worked perfectly.", "positive"),
        ("The service failed and erased my saved work.", "negative"),
        ("The features are useful, but the interface is confusing.", "mixed"),
    ]
    print("Prompt Q3 - few-shot:\n", ask(sentiment_prompt(reviews, few_shot), temperature=0))
    # Zero-shot is efficient for familiar, simple tasks. One-shot is useful when one
    # example can establish the format. Few-shot is best when the task has edge cases
    # or when several examples are needed to demonstrate the decision boundary.

    # Prompt Q4 - Chain of Thought
    math_prompt = """
Solve this problem. Show the calculation step by step, then put the result on a final
line beginning with "Final answer:".

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""
    print("Prompt Q4:\n", ask(math_prompt, temperature=0))
    # Explicit intermediate calculations can reduce skipped operations and make the
    # answer easier for a person to inspect. The correct final salary is $102,700.

    # Prompt Q5 - Structured Output
    review = (
        "I've been using this tool for three months. It handles large datasets well, "
        "but the UI is clunky and the export options are limited."
    )
    structured_prompt = f"""
Analyze the review between triple backticks. Return ONLY valid JSON with exactly these
keys: "sentiment", "confidence" (a number from 0 to 1), and "reason" (one sentence).
```{review}```
"""
    raw_response = ask(structured_prompt, temperature=0)
    print("Prompt Q5 raw response:", raw_response)
    try:
        analysis = json.loads(raw_response)
        print("Prompt Q5 sentiment:", analysis["sentiment"])
        print("Prompt Q5 confidence:", analysis["confidence"])
        print("Prompt Q5 reason:", analysis["reason"])
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        print("Prompt Q5 JSON parsing failed:", error)
        print("Prompt Q5 raw response for debugging:", raw_response)

    # Prompt Q6 - Delimiters
    user_text = (
        "First boil a pot of water. Once boiling, add a handful of salt and the pasta. "
        "Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
    )
    prose = "The neighborhood library is quiet in the morning. Sunlight falls across the reading tables."
    delimiter_template = """
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{text}```
"""
    print("Prompt Q6 - instructions:\n", ask(delimiter_template.format(text=user_text), temperature=0))
    print("Prompt Q6 - prose:", ask(delimiter_template.format(text=prose), temperature=0))
    # Delimiters distinguish untrusted user content from instructions, reducing
    # ambiguity and helping limit prompt-injection or accidental instruction mixing.


def run_local_model_question():
    # --- Local Models with Ollama ---
    # Ollama Q1
    # Output captured from Qwen3 0.6B
    #
    # Reasoning trace:
    # """
    # Okay, the user wants me to explain a large language model in two sentences. Let
    # me start by recalling what I know about them. A large language model is a type of
    # machine learning model that can understand and generate human language. So, first
    # sentence: Introduce the key points, like understanding and generating language.
    # Second sentence: Mention the use cases, maybe like answering questions or creating
    # content. Wait, need to make sure it's concise. Let me check if I'm covering both
    # main aspects. Also, use simple language. Avoid jargon. Alright, that should work.
    # """
    #
    # Final reply:
    # """
    # A large language model is a machine learning model that can understand and generate
    # human language, enabling it to answer questions, create content, or perform tasks
    # based on text.
    # """
    prompt = "Explain what a large language model is in two sentences."
    print("Ollama Q1 - OpenAI gpt-4o-mini response:", ask(prompt, temperature=0))
    # Qwen used simpler language and exposed a long reasoning trace before giving a
    # one-sentence answer. GPT-4o-mini followed the requested two-sentence format and
    # added technical context about training data, deep learning, and next-word
    # prediction. One advantage of a local model is that prompts can remain on the
    # user's machine; one disadvantage is that local inference consumes storage,
    # memory, and processing power and may be less capable on limited hardware.


def main():
    run_api_questions()
    run_system_questions()
    run_prompt_questions()
    run_local_model_question()


if __name__ == "__main__":
    main()
