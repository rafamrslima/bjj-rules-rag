import json
import asyncio
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import faithfulness, answer_relevancy
from bjj_rules_rag import user_query 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datetime import datetime
from pathlib import Path

# 1. Define evaluation questions and ground truth answers here
print("Loading test dataset from eval_questions.json...")
with open("eval_questions.json", "r", encoding="utf-8") as f:
    qa_bank = json.load(f)

async def main():
    # 2. Build the Ragas Evaluation Dataset by routing questions through your RAG
    samples = []
    for item in qa_bank:
        # Run your actual system using the evaluation question
        rag_output = user_query(item["question"])
        
        # Structure it into the Ragas layout
        sample = SingleTurnSample(
            user_input=item["question"],
            response=rag_output["response"],
            retrieved_contexts=rag_output["retrieved_contexts"],
            reference=item["reference"]
        )
        samples.append(sample)
    
    dataset = EvaluationDataset(samples=samples)

    # 3. Configure Ragas to use your local Ollama instance as the Judge
    OLLAMA_URL = "http://localhost:11434/v1"

    local_llm = LangchainLLMWrapper(
        ChatOpenAI(model="llama3", base_url=OLLAMA_URL, api_key="ollama", temperature=0)
    )
    
    local_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL,
        api_key="ollama",
        check_embedding_ctx_length=False,
    )
)

    # 4. Run evaluation completely local
    print("Evaluating your RAG using local Ollama models...")
    run_config = RunConfig(
        timeout=600,      # seconds per job, including time waiting in Ollama's queue
        max_workers=1,    # concurrent jobs; keep close to what Ollama processes in parallel
        max_retries=3,    # default 10 retries x long waits can take hours
        max_wait=30,      # max backoff between retries
    )
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=local_llm,
        embeddings=local_embeddings,
        run_config=run_config
    )

    df = results.to_pandas()
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"ollama_llama3_{timestamp}.csv"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved per-question results to {csv_path}")
    
    print("\n=== Local Evaluation Results ===")
    print(results)
    print(
    df.sort_values("faithfulness")[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
        .head(5)
        .to_string(index=False)
    )

if __name__ == "__main__":
    # Ragas evaluations are asynchronous under the hood
    asyncio.run(main())