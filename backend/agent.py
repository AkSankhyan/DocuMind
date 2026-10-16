import os
import re
from typing import Dict, Any, List, Literal, Generator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from .schema import AgentState, DocumentChunk, Citation
from .vector_store import VectorStoreManager
from . import config

# Phrases that are purely conversational — no RAG needed
_CONVERSATIONAL_PATTERNS = [
    r"^(hi|hello|hey|howdy|hiya|greetings)\.?[!?]?$",
    r"^(thanks|thank you|thx|ty|cheers)\.?[!?]?$",
    r"^(bye|goodbye|see you|cya|good night|good morning|good afternoon)\.?[!?]?$",
    r"^(ok|okay|alright|sure|got it|sounds good|perfect|great|nice)\.?[!?]?$",
    r"^(how are you|how do you do|what's up|what up|wassup)\??$",
    r"^(who are you|what are you|what can you do)\??$",
]
_CONVERSATIONAL_RE = re.compile("|".join(_CONVERSATIONAL_PATTERNS), re.IGNORECASE)

_CONVERSATIONAL_RESPONSES = {
    "greeting": "Hello! 👋 I'm your Document Intelligence Assistant. Ask me anything about the document you've uploaded and I'll retrieve and cite the most relevant information for you.",
    "thanks": "You're welcome! Let me know if you have more questions about the document. 😊",
    "farewell": "Goodbye! Feel free to come back anytime with more document questions. 👋",
    "ok": "Got it! Feel free to ask me anything about your document.",
    "howami": "I'm doing great and ready to help! What would you like to know from the document?",
    "whoareyou": "I'm an AI-powered Document Intelligence Assistant built on a Corrective RAG pipeline with LangGraph, Groq LLMs, Qdrant vector search, and FlashRank reranking. Ask me anything about your document!",
}

def _get_conversational_response(text: str) -> str | None:
    """Returns a canned response if input is purely conversational, else None."""
    t = text.strip().lower()
    if re.match(r"^(hi|hello|hey|howdy|hiya|greetings)\.?[!?]?$", t, re.I):
        return _CONVERSATIONAL_RESPONSES["greeting"]
    if re.match(r"^(thanks|thank you|thx|ty|cheers)\.?[!?]?$", t, re.I):
        return _CONVERSATIONAL_RESPONSES["thanks"]
    if re.match(r"^(bye|goodbye|see you|cya|good night|good morning|good afternoon)\.?[!?]?$", t, re.I):
        return _CONVERSATIONAL_RESPONSES["farewell"]
    if re.match(r"^(ok|okay|alright|sure|got it|sounds good|perfect|great|nice)\.?[!?]?$", t, re.I):
        return _CONVERSATIONAL_RESPONSES["ok"]
    if re.match(r"^(how are you|how do you do|what's up|what up|wassup)\??$", t, re.I):
        return _CONVERSATIONAL_RESPONSES["howami"]
    if re.match(r"^(who are you|what are you|what can you do)\??$", t, re.I):
        return _CONVERSATIONAL_RESPONSES["whoareyou"]
    return None


class AgenticRAG:
    """
    State-of-the-art Corrective RAG (CRAG) Agent using LangGraph and Groq.
    Features:
    - Multi-stage retrieval + FlashRank cross-encoder reranking
    - Document relevance grading
    - Query rewriting feedback loop for ambiguous questions
    - Strict citation grounding to eliminate hallucinations
    """

    def __init__(self, vector_store: VectorStoreManager = None):
        self.vector_store = vector_store or VectorStoreManager()
        
        # Primary LLM for generation & reasoning
        self.llm = ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.1
        )
        
        # Fast LLM for grading & query transformations
        self.fast_llm = ChatGroq(
            model=config.GROQ_FALLBACK_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.0
        )
        
        self.fallback_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ]
        
        self.graph = self._build_graph()

    def _invoke_with_fallback(self, prompt, input_dict, primary_llm, temperature=0.1):
        """Invokes chain with automatic model fallback if a model is unavailable or 404s."""
        try:
            chain = prompt | primary_llm
            return chain.invoke(input_dict)
        except Exception as e:
            err_str = str(e).lower()
            if "not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                for alt in self.fallback_models:
                    if alt == getattr(primary_llm, "model_name", ""):
                        continue
                    try:
                        print(f"[Model Fallback] Trying alternate model '{alt}'...")
                        alt_llm = ChatGroq(model=alt, api_key=config.GROQ_API_KEY, temperature=temperature)
                        return (prompt | alt_llm).invoke(input_dict)
                    except Exception:
                        continue
            raise e

    def _build_graph(self):
        """Constructs the LangGraph state graph with corrective loops."""
        workflow = StateGraph(AgentState)

        # Register nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)
        workflow.add_node("generate_answer", self._generate_answer_node)

        # Edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "grade_documents")

        # Conditional branching from grader
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_next_step,
            {
                "generate": "generate_answer",
                "rewrite": "rewrite_query"
            }
        )

        # Query rewrite loops back to retrieve
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    # Node 1: Retrieve candidate chunks
    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        question = state.get("question", "")
        collection_id = state.get("collection_id", "")
        
        print(f"[Node: Retrieve] Searching collection '{collection_id}' for query: '{question}'")
        raw_chunks = self.vector_store.search(
            collection_id=collection_id,
            query=question,
            top_k=config.RETRIEVAL_TOP_K
        )

        chunks_dict = [chunk.model_dump() for chunk in raw_chunks]
        return {"retrieved_chunks": chunks_dict}

    # Node 2: FlashRank Cross-Encoder Reranking
    def _rerank_node(self, state: AgentState) -> Dict[str, Any]:
        question = state.get("question", "")
        retrieved = state.get("retrieved_chunks", [])
        
        if not retrieved:
            return {"reranked_chunks": []}

        chunks = [DocumentChunk(**c) for c in retrieved]
        print(f"[Node: Rerank] Reranking {len(chunks)} candidate chunks with FlashRank...")
        
        reranked = self.vector_store.rerank(
            query=question,
            chunks=chunks,
            top_k=config.RERANK_TOP_K
        )

        reranked_dict = [chunk.model_dump() for chunk in reranked]
        return {"reranked_chunks": reranked_dict}

    # Node 3: Document Relevance Grader
    def _grade_documents_node(self, state: AgentState) -> Dict[str, Any]:
        question = state.get("question", "")
        chunks = state.get("reranked_chunks", [])
        
        if not chunks:
            print("[Node: Grader] No chunks retrieved. Marking irrelevant.")
            return {"is_relevant": False}

        # Build context for evaluation
        context_preview = "\n---\n".join([c["content"][:400] for c in chunks[:3]])
        
        grader_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an evaluator assessing whether the retrieved context contains information relevant to the user question. "
                       "Respond ONLY with 'yes' if relevant, or 'no' if irrelevant. Do not provide explanation."),
            ("human", f"Context:\n{context_preview}\n\nQuestion: {question}\n\nIs this context relevant to answering the question? (yes/no):")
        ])

        try:
            response = self._invoke_with_fallback(grader_prompt, {}, self.fast_llm, temperature=0.0)
            grade = response.content.strip().lower()
            is_relevant = "yes" in grade
            print(f"[Node: Grader] Evaluated relevance: {is_relevant} (Grade: '{grade}')")
            return {"is_relevant": is_relevant}
        except Exception as e:
            print(f"[Node: Grader] Grading fallback due to error: {e}")
            return {"is_relevant": True}

    # Node 4: Query Rewriting Node
    def _rewrite_query_node(self, state: AgentState) -> Dict[str, Any]:
        original_q = state.get("original_question", state.get("question", ""))
        retry_count = state.get("retry_count", 0) + 1
        
        print(f"[Node: Rewrite] Rewriting query attempt #{retry_count} for: '{original_q}'")
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert search query optimizer. Given a user query that failed to retrieve relevant documents, "
                       "reformulate it to improve semantic vector retrieval. Focus on technical keywords, domain terminology, and clarity. "
                       "Output ONLY the improved search query, nothing else."),
            ("human", f"Initial Query: {original_q}\n\nOptimized Search Query:")
        ])

        try:
            response = self._invoke_with_fallback(rewrite_prompt, {}, self.fast_llm, temperature=0.0)
            new_query = response.content.strip().replace('"', '')
            print(f"[Node: Rewrite] New formulated query: '{new_query}'")
            return {"question": new_query, "retry_count": retry_count}
        except Exception as e:
            print(f"[Node: Rewrite] Failed: {e}")
            return {"retry_count": retry_count}

    # Node 5: Answer Generation with Strict Grounding & Citations
    def _generate_answer_node(self, state: AgentState) -> Dict[str, Any]:
        question = state.get("original_question", state.get("question", ""))
        chunks = state.get("reranked_chunks", [])

        if not chunks:
            return {
                "answer": "I could not find any relevant information in the uploaded document to answer your question.",
                "citations": []
            }

        # Build numbered context passages with source and page
        context_blocks = []
        citations_data = []

        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.get("metadata", {}).get("source", "Document")
            page = chunk.get("metadata", {}).get("page", 1)
            content = chunk.get("content", "")
            score = chunk.get("score")
            
            context_blocks.append(f"[[Passage {idx} | Source: {source}, Page {page}]]\n{content}")
            citations_data.append({
                "source": source,
                "page": page,
                "snippet": content[:250] + "..." if len(content) > 250 else content,
                "score": score
            })

        formatted_context = "\n\n".join(context_blocks)

        # Inject today's date so LLM can calculate durations accurately
        from datetime import date
        today_str = date.today().strftime("%d %B %Y")  # e.g. "16 October 2026"

        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an advanced Document Intelligence Assistant.
Today's date is {today_str}. Use this to calculate durations, ages, or time elapsed when relevant.
Answer the user's question with high precision based on the provided Context Passages.

Rules:
1. Base ALL factual claims on the Context Passages. Do NOT hallucinate facts not present in the document.
2. REASONING IS ALLOWED: You MAY perform arithmetic, date calculations, duration math, and logical inference on top of the document facts. Clearly distinguish calculated results from directly stated facts.
3. CRITICAL: When listing items (projects, skills, experiences, certifications, etc.), enumerate ALL items found across ALL passages — never truncate. Use a numbered or bulleted list.
4. If a question cannot be answered from the passages at all, say: "Based on the document, I could not find information regarding..." and summarize what IS present.
5. Present tabular or numeric data in clean markdown tables or bullets.
6. Include inline citations for directly quoted facts, e.g.: [Source: report.pdf, Page 3].
7. Read EVERY passage carefully before concluding something is absent from the document."""),
            ("human", "Context Passages:\n\n{context}\n\nQuestion: {question}\n\nDetailed Answer:")
        ])

        try:
            response = self._invoke_with_fallback(
                gen_prompt,
                {"context": formatted_context, "question": question},
                self.llm,
                temperature=0.1
            )
            answer_text = response.content.strip()
            return {"answer": answer_text, "citations": citations_data}
        except Exception as e:
            print(f"[Node: Generator] Error: {e}")
            return {
                "answer": f"An error occurred while generating the response: {str(e)}",
                "citations": citations_data
            }

    # Conditional router
    def _decide_next_step(self, state: AgentState) -> Literal["generate", "rewrite"]:
        is_relevant = state.get("is_relevant", False)
        retry_count = state.get("retry_count", 0)

        # If relevant or retry limit reached (max 1 rewrite), generate answer
        if is_relevant or retry_count >= 1:
            return "generate"
        
        # Otherwise, attempt to rewrite query once
        return "rewrite"

    def run(self, question: str, collection_id: str) -> Dict[str, Any]:
        """Runs the LangGraph workflow synchronously."""
        # Short-circuit for purely conversational inputs
        convo_resp = _get_conversational_response(question)
        if convo_resp:
            return {"answer": convo_resp, "citations": [], "collection_id": collection_id}

        initial_state: AgentState = {
            "question": question,
            "original_question": question,
            "collection_id": collection_id,
            "retry_count": 0,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "citations": []
        }

        result = self.graph.invoke(initial_state)
        return {
            "answer": result.get("answer", "No response could be generated."),
            "citations": result.get("citations", []),
            "collection_id": collection_id
        }

    def stream_answer(self, question: str, collection_id: str) -> Generator[str, None, None]:
        """
        Streams the answer token-by-token using LangChain streaming.
        Yields raw text chunks for SSE. Emits a final JSON citations block.
        """
        import json

        # Short-circuit conversational inputs — stream the canned reply char-by-char
        convo_resp = _get_conversational_response(question)
        if convo_resp:
            for char in convo_resp:
                yield f"data: {json.dumps({'token': char})}\n\n"
            yield f"data: {json.dumps({'citations': [], 'done': True})}\n\n"
            return

        # Run non-streaming RAG graph to get context + citations
        initial_state: AgentState = {
            "question": question,
            "original_question": question,
            "collection_id": collection_id,
            "retry_count": 0,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "citations": []
        }

        # Run up to (but not including) generate_answer to get the reranked chunks
        # We replay generation with streaming enabled
        result = self.graph.invoke(initial_state)
        chunks = result.get("reranked_chunks", [])
        citations_data = result.get("citations", [])
        original_q = question

        if not chunks:
            msg = "I could not find any relevant information in the uploaded document to answer your question."
            for char in msg:
                yield f"data: {json.dumps({'token': char})}\n\n"
            yield f"data: {json.dumps({'citations': [], 'done': True})}\n\n"
            return

        # Re-build context blocks for streaming generation
        context_blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.get("metadata", {}).get("source", "Document")
            page = chunk.get("metadata", {}).get("page", 1)
            content = chunk.get("content", "")
            context_blocks.append(f"[[Passage {idx} | Source: {source}, Page {page}]]\n{content}")

        formatted_context = "\n\n".join(context_blocks)

        # Inject today's date so LLM can calculate durations accurately
        from datetime import date
        today_str = date.today().strftime("%d %B %Y")

        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an advanced Document Intelligence Assistant.
Today's date is {today_str}. Use this to calculate durations, ages, or time elapsed when relevant.
Answer the user's question with high precision based on the provided Context Passages.

Rules:
1. Base ALL factual claims on the Context Passages. Do NOT hallucinate facts not present in the document.
2. REASONING IS ALLOWED: You MAY perform arithmetic, date calculations, duration math, and logical inference on top of the document facts. Clearly distinguish calculated results from directly stated facts.
3. CRITICAL: When listing items (projects, skills, experiences, certifications, etc.), enumerate ALL items found across ALL passages — never truncate. Use a numbered or bulleted list.
4. If a question cannot be answered from the passages at all, say: "Based on the document, I could not find information regarding..." and summarize what IS present.
5. Present tabular or numeric data in clean markdown tables or bullets.
6. Include inline citations for directly quoted facts, e.g.: [Source: report.pdf, Page 3].
7. Read EVERY passage carefully before concluding something is absent from the document."""),
            ("human", "Context Passages:\n\n{context}\n\nQuestion: {question}\n\nDetailed Answer:")
        ])

        streaming_llm = ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.1,
            streaming=True
        )

        try:
            chain = gen_prompt | streaming_llm
            for chunk in chain.stream({"context": formatted_context, "question": original_q}):
                token = chunk.content
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            err_msg = f"\n\n⚠️ Stream error: {str(e)}"
            yield f"data: {json.dumps({'token': err_msg})}\n\n"

        yield f"data: {json.dumps({'citations': citations_data, 'done': True})}\n\n"