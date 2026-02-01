#!/usr/bin/env python3
"""RAG (Retrieval-Augmented Generation) example.

This example demonstrates how to use Neural Search as the retrieval
component in a RAG pipeline. It requires an OpenAI API key.
"""

import os

import httpx

# Configuration
NEURAL_SEARCH_URL = "http://localhost:8000/api/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class NeuralSearchClient:
    """Simple client for Neural Search API."""

    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url, timeout=60.0)

    def create_collection(self, name: str) -> bool:
        """Create a collection."""
        response = self.client.post("/collections", json={"name": name})
        return response.status_code in (201, 409)

    def index_documents(self, collection: str, documents: list[dict]) -> dict:
        """Index documents."""
        response = self.client.post(
            "/documents",
            json={"collection": collection, "documents": documents},
        )
        return response.json()

    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        hybrid: bool = True,
        rerank: bool = False,
    ) -> list[dict]:
        """Search for relevant documents."""
        response = self.client.post(
            "/search",
            json={
                "query": query,
                "collection": collection,
                "top_k": top_k,
                "hybrid": hybrid,
                "rerank": rerank,
            },
        )
        return response.json()["results"]


def generate_with_openai(prompt: str, context: str) -> str:
    """Generate response using OpenAI API."""
    if not OPENAI_API_KEY:
        # Fallback for demo without API key
        return f"[Demo Mode - Would generate response based on context]\n\nContext provided:\n{context[:500]}..."

    client = httpx.Client(timeout=60.0)
    response = client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Answer the user's question based on the following context. If the context doesn't contain relevant information, say so.\n\nContext:\n{context}",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.7,
            "max_tokens": 500,
        },
    )

    return response.json()["choices"][0]["message"]["content"]


def rag_query(
    search_client: NeuralSearchClient,
    collection: str,
    question: str,
    top_k: int = 5,
) -> str:
    """Perform RAG query: retrieve relevant context and generate answer."""
    print(f"\nQuestion: {question}")

    # Step 1: Retrieve relevant documents
    print("Retrieving relevant documents...")
    results = search_client.search(
        query=question,
        collection=collection,
        top_k=top_k,
        hybrid=True,
    )

    if not results:
        return "No relevant documents found."

    # Show retrieved documents
    print(f"Found {len(results)} relevant documents:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.3f}] {r['content'][:100]}...")

    # Step 2: Build context from retrieved documents
    context = "\n\n---\n\n".join([
        f"Document {i+1}:\n{r['content']}"
        for i, r in enumerate(results)
    ])

    # Step 3: Generate answer
    print("\nGenerating answer...")
    answer = generate_with_openai(question, context)

    return answer


def main():
    """Run the RAG example."""
    search_client = NeuralSearchClient(NEURAL_SEARCH_URL)

    # Create collection and index sample documents
    print("Setting up knowledge base...")
    collection = "rag-demo"
    search_client.create_collection(collection)

    # Sample knowledge base about a fictional company
    documents = [
        {
            "id": "about-company",
            "content": "TechCorp is a technology company founded in 2020. We specialize in AI-powered solutions for enterprise customers. Our headquarters is in San Francisco, with offices in London and Singapore.",
            "metadata": {"category": "company"},
        },
        {
            "id": "products",
            "content": "TechCorp offers three main products: DataFlow (data pipeline management), InsightAI (business intelligence), and SecureShield (cybersecurity). All products are available as cloud services or on-premise installations.",
            "metadata": {"category": "products"},
        },
        {
            "id": "pricing",
            "content": "TechCorp pricing is based on usage tiers. The Starter plan costs $99/month for up to 10 users. The Business plan costs $499/month for up to 50 users. Enterprise pricing is custom and includes dedicated support.",
            "metadata": {"category": "pricing"},
        },
        {
            "id": "support",
            "content": "TechCorp provides 24/7 support for Business and Enterprise customers. Starter customers have access to email support during business hours. All customers have access to our comprehensive documentation and community forums.",
            "metadata": {"category": "support"},
        },
        {
            "id": "integrations",
            "content": "TechCorp products integrate with major platforms including Salesforce, HubSpot, Slack, Microsoft Teams, AWS, Azure, and Google Cloud. Custom integrations are available through our REST API and webhooks.",
            "metadata": {"category": "integrations"},
        },
    ]

    search_client.index_documents(collection, documents)
    print(f"Indexed {len(documents)} documents")

    # Example RAG queries
    questions = [
        "What products does TechCorp offer?",
        "How much does the Business plan cost?",
        "Does TechCorp integrate with Slack?",
        "Where is TechCorp headquartered?",
    ]

    print("\n" + "=" * 60)
    print("RAG Q&A Demo")
    print("=" * 60)

    for question in questions:
        answer = rag_query(search_client, collection, question)
        print(f"\nAnswer:\n{answer}")
        print("\n" + "-" * 60)

    print("\nDone!")


if __name__ == "__main__":
    main()
