

from openai import OpenAI
import json
from sklearn.metrics.pairwise import cosine_similarity
import torch


MODEL_NAME = "gpt-4o-mini"
client = OpenAI()


with open("../artifacts/ocr/context.json", "r", encoding="utf-8") as f:
    context_data = json.load(f)

context_text = "\n".join([c["text"] for c in context_data])


question = "Quels sont les droits des parents mentionnés dans ce document ?"

prompt = f"""
Tu es un assistant qui répond UNIQUEMENT à partir du contexte ci-dessous.

CONTEXTE:
{context_text}

QUESTION:
{question}

Réponds uniquement avec les informations présentes dans le contexte.
"""


response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

answer = response.choices[0].message.content

print(" RÉPONSE LLM:")
print(answer)


def faithfulness_score(answer, context):
    return sum(1 for word in answer.split() if word in context) / max(len(answer.split()), 1)

faithfulness = faithfulness_score(answer, context_text)

print("\n FAITHFULNESS:", round(faithfulness, 3))
