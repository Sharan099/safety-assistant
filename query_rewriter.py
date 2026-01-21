"""
Query Rewriter for Better Retrieval
Rewrites user questions into keyword-heavy legal queries and semantic paraphrases
"""
import re
from typing import Dict, List
from config import LLM_PROVIDER, LLM_MODEL, ANTHROPIC_API_KEY, OPENAI_API_KEY

# Initialize LLM clients
claude_client = None
openai_client = None

if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
    from anthropic import Anthropic
    claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
elif LLM_PROVIDER == "openai" and OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

def rewrite_query(question: str) -> Dict[str, str]:
    """
    Rewrite user question into:
    1. Keyword-heavy legal query (preserves regulation terms, clause hints, limits)
    2. Semantic paraphrase (for dense retrieval)
    
    Returns:
        {
            "keyword_query": "...",
            "semantic_query": "..."
        }
    """
    prompt = f"""You are a query rewriting expert for safety regulation search.

**User Question:** {question}

**Task:**
Rewrite this question into TWO versions:

1. **Keyword-Heavy Legal Query**: 
   - Preserve exact regulation terms (e.g., "UNECE R94", "HIC", "FMVSS 208")
   - Preserve clause hints (e.g., "clause 5.2.1", "section 3")
   - Preserve exact limits and thresholds (e.g., "≤ 1000", "shall not exceed")
   - Use legal terminology
   - Make it keyword-rich for exact matching

2. **Semantic Paraphrase**:
   - Rewrite in natural language
   - Expand abbreviations
   - Use synonyms where appropriate
   - Focus on meaning and intent
   - Good for semantic similarity search

**Output Format (JSON):**
{{
    "keyword_query": "<keyword-heavy legal query>",
    "semantic_query": "<semantic paraphrase>"
}}

**Example:**
Input: "What is the HIC limit for frontal collision?"

Output:
{{
    "keyword_query": "UNECE R94 HIC Head Injury Criterion limit threshold frontal collision test requirement clause",
    "semantic_query": "What is the maximum allowed Head Injury Criterion value for frontal impact crash tests according to UNECE regulation R94?"
}}

**Output (JSON only):**"""

    try:
        if claude_client:
            response = claude_client.messages.create(
                model=LLM_MODEL if LLM_MODEL.startswith("claude") else "claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = response.content[0].text.strip()
        elif openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            result_text = response.choices[0].message.content.strip()
        else:
            # Fallback: simple extraction
            return {
                "keyword_query": question,
                "semantic_query": question
            }
        
        # Extract JSON from response
        import json
        # Try to find JSON in response
        json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "keyword_query": result.get("keyword_query", question),
                "semantic_query": result.get("semantic_query", question)
            }
        else:
            # Fallback
            return {
                "keyword_query": question,
                "semantic_query": question
            }
    except Exception as e:
        print(f"⚠️ Query rewriting failed: {e}, using original question")
        return {
            "keyword_query": question,
            "semantic_query": question
        }


