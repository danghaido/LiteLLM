from LiteLLM.lite import LiteLLMClient
from LiteLLM.Response import ResponseInput

if __name__ == "__main__":
    queries_text = [
        "How many continents are there in the world?",
        "How many r's are in the word 'strawberry'?",
        "What is the capital of Japan?",
        "Explain what overfitting is in one sentence.",
        "2 + 2 * 5 = ?",
        "Is math or literature harder for high school student?",
    ]

    queries = [ResponseInput(q) for q in queries_text]
    client =  LiteLLMClient()

    responses = client.batch_complete(queries)
    for i, (q, resp) in enumerate(zip(queries_text, responses), 1):
        print(f"\n=== Q{i}: {q}")
        print("Answer:", resp.transform())
        print("Usage:", resp.usage())