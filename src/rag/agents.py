import logging

logger = logging.getLogger(__name__)

async def grade_context(llm, question: str, chunks: list[str]) -> bool:
    """
    Evaluates if the retrieved chunks contain relevant information to answer the question.
    Returns True if relevant, False if irrelevant.
    """
    if not chunks:
        return False
        
    context_text = "\n\n".join(chunks)
    prompt = (
        "You are an objective grader evaluating the relevance of retrieved documents to a user question.\n"
        "Here is the retrieved context:\n"
        "------------------\n"
        f"{context_text}\n"
        "------------------\n"
        f"Here is the user question: {question}\n\n"
        "Does this context contain information directly relevant to the user's question? "
        "Reply exactly with 'YES' or 'NO' and nothing else."
    )
    
    try:
        response = await llm.ainvoke(prompt)
        content = response.content.strip().upper()
        # Clean up the output to handle LLM variations safely
        if "YES" in content:
            return True
        elif "NO" in content:
            return False
        return True  # Fallback
    except Exception as e:
        logger.error(f"[Agents] Context grading failed: {e}")
        return True # Fallback to true if LLM fails

async def rewrite_query(llm, original_question: str) -> str:
    """
    Rewrites a failed search query to be highly optimized for a vector database.
    """
    prompt = (
        "You are an expert search query optimizer. The user's original query failed to return relevant results.\n"
        f"Original query: {original_question}\n\n"
        "Rewrite this query into a highly specific, keyword-rich search string optimized for a vector database (Semantic & BM25 search). "
        "Extract the core entities and intent. "
        "Do NOT include any pleasantries, conversational text, or punctuation. Output ONLY the rewritten search string."
    )
    
    try:
        response = await llm.ainvoke(prompt)
        rewritten = response.content.strip().replace('"', '')
        return rewritten
    except Exception as e:
        logger.error(f"[Agents] Query rewriting failed: {e}")
        return original_question

async def grade_generation_vs_documents(llm, question: str, chunks: list[str], answer: str) -> bool:
    """
    Checks if the generated answer hallucinated facts not present in the chunks.
    """
    context_text = "\n\n".join(chunks)
    prompt = (
        "You are a hallucination checker. Your job is to verify if an AI-generated answer is entirely grounded "
        "in the provided context, or if it hallucinates information.\n"
        "Here is the provided context:\n"
        "------------------\n"
        f"{context_text}\n"
        "------------------\n"
        f"User question: {question}\n"
        f"AI's Answer: {answer}\n\n"
        "Does the AI's Answer rely ONLY on the provided context, without making up facts? "
        "Reply exactly with 'YES' or 'NO' and nothing else."
    )
    
    try:
        response = await llm.ainvoke(prompt)
        content = response.content.strip().upper()
        if "YES" in content:
            return True
        elif "NO" in content:
            return False
        return True
    except Exception as e:
        logger.error(f"[Agents] Hallucination grading failed: {e}")
        return True
