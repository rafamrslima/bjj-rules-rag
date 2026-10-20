import json
import asyncio
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from bjj_rules_rag import user_query
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

JUDGE_MODEL = "gpt-4o"
JUDGE_EMBEDDING_MODEL = "text-embedding-3-small"

# 1. Define evaluation questions and ground truth answers here
print("Loading test dataset from eval_questions.json...")
with open("eval_questions.json", "r", encoding="utf-8") as f:
    qa_bank = json.load(f)

async def main():
    # 2. Build the Ragas Evaluation Dataset by routing questions through your RAG
    load_dotenv()
    samples = []
    for item in qa_bank:
        # Run your actual system using the evaluation question
        rag_output = user_query(item["question"])

        # Context metrics score each chunk separately; one joined string would count as a single chunk.
        contexts = rag_output["retrieved_contexts"]
        if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
            raise TypeError(
                f"retrieved_contexts must be list[str], got {type(contexts).__name__} "
                f"for question: {item['question']!r}"
            )

        # Structure it into the Ragas layout
        sample = SingleTurnSample(
            user_input=item["question"],
            response=rag_output["response"],
            retrieved_contexts=rag_output["retrieved_contexts"],
            reference=item["reference"]
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)

    # 3. Configure Ragas to use OpenAI as the Judge
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    )

    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=JUDGE_EMBEDDING_MODEL)
    )

    # 4. Run evaluation with the OpenAI API
    print(f"Evaluating your RAG using OpenAI ({JUDGE_MODEL}) as the judge...")
    run_config = RunConfig(
        timeout=120,
        max_workers=3,    # 8 exceeded the 30k tokens/min limit for gpt-4o and lost jobs to 429 errors
        max_retries=5,
        max_wait=60,
    )
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config
    )

    df = results.to_pandas()
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"openai_{JUDGE_MODEL}_{timestamp}.csv"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved per-question results to {csv_path}")

    print("\n=== OpenAI Evaluation Results ===")
    print(results)
    print(
    df.sort_values("faithfulness")[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
        .head(5)
        .to_string(index=False)
    )

if __name__ == "__main__":
    # Ragas evaluations are asynchronous under the hood
    asyncio.run(main())
