from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.core import PromptTemplate
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from src.db.read_db import SemanticBM25Retriever

class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


template = (
    "The below provided is the context from a bunch of cook books \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question like a cheif in bullet points and elaborate: {query_str}\n"
)

qa_template = PromptTemplate(template)

class RAGWorkflow(Workflow):
    
    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:

        collection_name = ev.get("collection_name")
        if not collection_name:
            return None

        retriever = SemanticBM25Retriever(collection_name=collection_name)

        return StopEvent(result=retriever)

    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> RetrieverEvent | None:
        
        query = ev.get("query")
        retriever = ev.get("retriever")

        if not query:
            return None

        print(f"Query the database with: {query}")

        await ctx.set("query", query)

        if retriever is None:
            print("Index is empty, load some documents before querying!")
            return None

        nodes = await retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")

        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:

        llm = Ollama(model="gemma2:2b", request_timeout=60.0)
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True, text_qa_template=qa_template)
        query = await ctx.get("query", default=None)

        response = await summarizer.asynthesize(query, nodes=ev.nodes)

        return StopEvent(result=response)
    
