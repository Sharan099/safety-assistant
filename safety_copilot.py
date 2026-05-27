"""
Safety Copilot Agent with LangGraph Orchestration
Handles question answering with source attribution and confidence scoring
"""
from typing import TypedDict, List, Dict, Optional
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    # Fallback for older versions
    from langgraph import StateGraph, END
import json
import re
from config import (
    LLM_PROVIDER, LLM_MODEL, ANTHROPIC_API_KEY, OPENAI_API_KEY,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW,
    REFUSE_KEYWORDS
)
from vector_store import SafetyVectorStore
from document_processor import DocumentChunk
from domain_classifier import DomainClassifier

# Initialize LLM client
claude_client = None
openai_client = None

if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
    try:
        from anthropic import Anthropic
        claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        print("✅ Anthropic Claude client initialized")
    except Exception as e:
        print(f"⚠️  Failed to initialize Anthropic: {e}")

if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"⚠️  Failed to initialize OpenAI: {e}")

class SafetyCopilotState(TypedDict):
    """State for Safety Copilot workflow"""
    user_question: str
    retrieved_chunks: List[tuple]  # (chunk, similarity_score)
    answer: str
    sources: List[Dict]
    confidence_score: float
    confidence_level: str
    should_refuse: bool
    refusal_reason: str
    workflow_stage: str
    needs_synthesis: bool
    needs_scenario_reasoning: bool
    synthesis_result: Optional[Dict]
    conversation_history: List[Dict]

class SafetyGuardrails:
    """Safety guardrails to prevent inappropriate responses"""
    
    @staticmethod
    def check_should_refuse(question: str) -> tuple:
        """Check if question should be refused"""
        question_lower = question.lower()
        
        for keyword in REFUSE_KEYWORDS:
            if keyword in question_lower:
                return True, f"Request contains '{keyword}' - cannot provide legal interpretations or approvals"
        
        # Check for approval/certification requests
        approval_patterns = [
            r"approve.*design",
            r"certify.*safe",
            r"guarantee.*compliance",
            r"legal.*interpretation"
        ]
        
        for pattern in approval_patterns:
            if re.search(pattern, question_lower):
                return True, "Cannot provide approvals, certifications, or legal interpretations"
        
        return False, ""
    
    @staticmethod
    def add_disclaimer(answer: str) -> str:
        """Add safety disclaimer to answer"""
        disclaimer = "\n\n⚠️ **Disclaimer**: This information is for decision support only. Always consult qualified safety engineers and follow your organization's safety processes."
        return answer + disclaimer

class ConfidenceScorer:
    """Calculate confidence scores for answers"""
    
    @staticmethod
    def calculate_confidence(
        retrieval_scores: List[float],
        answer_consistency: float = 1.0
    ) -> tuple[float, str]:
        """
        Calculate confidence score based on retrieval similarity and answer consistency
        Returns (score, level) where level is "high", "medium", or "low"
        """
        if not retrieval_scores:
            return 0.0, "low"
        
        # Average retrieval similarity
        avg_similarity = sum(retrieval_scores) / len(retrieval_scores)
        
        # Top result similarity (most important)
        top_similarity = max(retrieval_scores)
        
        # Combined score (weighted average)
        confidence = (top_similarity * 0.6 + avg_similarity * 0.4) * answer_consistency
        
        # Determine level
        if confidence >= CONFIDENCE_HIGH:
            level = "high"
        elif confidence >= CONFIDENCE_MEDIUM:
            level = "medium"
        else:
            level = "low"
        
        return confidence, level

def safety_guardrail_agent(state: SafetyCopilotState) -> SafetyCopilotState:
    """Check if question should be refused and detect synthesis/scenario needs"""
    question = state.get("user_question", "")
    should_refuse, reason = SafetyGuardrails.check_should_refuse(question)
    
    # Check if synthesis is needed
    needs_synthesis = DomainClassifier.needs_synthesis(question)
    
    # Check if scenario reasoning is needed
    needs_scenario_reasoning = DomainClassifier.needs_scenario_reasoning(question)
    
    return {
        **state,
        "should_refuse": should_refuse,
        "refusal_reason": reason,
        "needs_synthesis": needs_synthesis,
        "needs_scenario_reasoning": needs_scenario_reasoning,
        "workflow_stage": "refused" if should_refuse else ("synthesis" if needs_synthesis else ("scenario" if needs_scenario_reasoning else "retrieval"))
    }

def retrieval_agent(state: SafetyCopilotState, vector_store: SafetyVectorStore, 
                    use_hybrid: bool = True) -> SafetyCopilotState:
    """Retrieve relevant document chunks with improved search"""
    question = state.get("user_question", "")
    needs_synthesis = state.get("needs_synthesis", False)
    
    # Detect domain for better filtering
    detected_domain = DomainClassifier.get_primary_domain(question)
    
    # For synthesis, retrieve more chunks from different documents
    top_k = 12 if needs_synthesis else 8
    
    # Search with domain-aware filtering
    results = vector_store.search(
        question, 
        top_k=top_k, 
        similarity_threshold=0.3,  # Lower threshold for better recall
        domain_filter=detected_domain if detected_domain != "General Safety" else None
    )
    
    # If no results or too few, try without domain filter
    if len(results) < 3:
        results = vector_store.search(question, top_k=top_k, similarity_threshold=0.25, domain_filter=None)
    
    # If synthesis needed, ensure diversity across documents
    if needs_synthesis and results:
        # Group by document
        doc_groups = {}
        for chunk, similarity in results:
            doc_name = chunk.document_name
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append((chunk, similarity))
        
        # If all from same document, search for more diverse sources
        if len(doc_groups) == 1 and len(results) < 10:
            # Get additional results with even lower threshold
            additional = vector_store.search(question, top_k=20, similarity_threshold=0.2)
            # Add unique chunks from different documents
            existing_docs = {chunk.document_name for chunk, _ in results}
            for chunk, sim in additional:
                if chunk.document_name not in existing_docs:
                    results.append((chunk, sim))
                    existing_docs.add(chunk.document_name)
                    if len(results) >= 12:
                        break
    
    # Re-sort by similarity after adding additional results
    results.sort(key=lambda x: x[1], reverse=True)
    
    return {
        **state,
        "retrieved_chunks": results,
        "workflow_stage": "synthesis" if needs_synthesis else "generation"
    }

def synthesis_agent_wrapper(state: SafetyCopilotState) -> SafetyCopilotState:
    """Wrapper for synthesis agent"""
    from synthesis_agent import synthesis_agent
    
    question = state.get("user_question", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    conversation_history = state.get("conversation_history", [])
    
    # Run synthesis agent
    synthesis_result = synthesis_agent(question, retrieved_chunks, conversation_history)
    
    # Build sources from synthesis
    sources = []
    for chunk, similarity in retrieved_chunks:
        from pdf_linker import find_pdf_path, get_pdf_relative_path
        pdf_path = find_pdf_path(chunk.document_name)
        pdf_relative_path = get_pdf_relative_path(pdf_path) if pdf_path else None
        
        source_info = {
            "document_name": chunk.document_name,
            "page_number": chunk.page_number,
            "section_number": chunk.section_number,
            "similarity_score": round(similarity, 3),
            "chunk_id": chunk.chunk_id,
            "origin": chunk.origin,
            "domain": chunk.domain,
            "strictness": chunk.strictness,
            "method": chunk.method,
            "year": chunk.year,
            "source_type": chunk.source_type,
            "test_type": chunk.test_type,
            "metric": chunk.metric,
            "dummy_type": chunk.dummy_type,
            "pdf_path": str(pdf_path) if pdf_path else None,
            "pdf_relative_path": pdf_relative_path
        }
        sources.append(source_info)
    
    # Calculate confidence based on multiple sources
    if retrieved_chunks:
        similarity_scores = [sim for _, sim in retrieved_chunks]
        confidence_score, confidence_level = ConfidenceScorer.calculate_confidence(similarity_scores)
        # Boost confidence if multiple sources agree
        if len(retrieved_chunks) >= 3:
            confidence_score = min(1.0, confidence_score * 1.1)
    else:
        confidence_score = 0.0
        confidence_level = "low"
    
    # Format synthesis answer
    answer = synthesis_result.get("synthesis", "")
    
    # Add table interpretations if found
    if synthesis_result.get("tables"):
        answer += "\n\n**📊 Technical Tables Interpreted:**\n"
        for i, table in enumerate(synthesis_result["tables"], 1):
            answer += f"\n**Table {i}** from {table['document']} (Page {table['page']}):\n"
            if table['table_data'].get('headers'):
                answer += f"Columns: {', '.join(table['table_data']['headers'])}\n"
    
    # Add conflict information
    if synthesis_result.get("conflicts"):
        answer += "\n\n**⚠️ Standard Conflicts Detected:**\n"
        for conflict in synthesis_result["conflicts"]:
            answer += f"\n- **{conflict['standard']}**: Multiple interpretations found\n"
            for source in conflict['sources'][:3]:
                answer += f"  - {source.get('origin', 'Unknown')} ({source.get('method', 'N/A')}): Page {source.get('page', 'N/A')}\n"
    
    # Add disclaimer
    answer = SafetyGuardrails.add_disclaimer(answer)
    
    # Clean up synthesis answer - remove garbled text
    import re
    answer = re.sub(r'[^\w\s\-.,;:()\[\]{}%°±×÷≤≥≠≈∞∑∏∫√αβγδεθλμπστφω]', ' ', answer)
    answer = re.sub(r'\s+', ' ', answer)
    answer = re.sub(r'([a-z])([A-Z])', r'\1 \2', answer)
    
    # Filter sources by quality - only show high-quality sources (similarity >= 0.65)
    high_quality_sources = [s for s in sources if s.get("similarity_score", 0.0) >= 0.65]
    
    # Extract only sources actually cited in synthesis answer
    from source_extractor import extract_cited_sources
    cited_sources = extract_cited_sources(answer, high_quality_sources)
    
    # If no sources were cited, use top high-quality sources by similarity
    if not cited_sources:
        cited_sources = sorted(high_quality_sources, key=lambda x: x.get("similarity_score", 0.0), reverse=True)[:min(5, len(high_quality_sources))]
    
    return {
        **state,
        "answer": answer,
        "sources": cited_sources,  # Only return sources actually used
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "synthesis_result": synthesis_result,
        "workflow_stage": "complete"
    }

def answer_generation_agent(state: SafetyCopilotState) -> SafetyCopilotState:
    """Generate answer from retrieved chunks"""
    question = state.get("user_question", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    
    if not retrieved_chunks:
        return {
            **state,
            "answer": f"I couldn't find relevant information in the available safety documents for your question: '{question}'. The documents I have access to don't appear to contain this information. Please try rephrasing your question or check if the specific document you're looking for is available in the library.",
            "sources": [],
            "confidence_score": 0.0,
            "confidence_level": "low",
            "workflow_stage": "complete"
        }
    
    # Prepare context from retrieved chunks
    context_parts = []
    sources = []
    
    # Classify domain for better context
    detected_domain = DomainClassifier.get_primary_domain(question)
    
    for chunk, similarity in retrieved_chunks:
        # Find PDF path for linking
        from pdf_linker import find_pdf_path, get_pdf_relative_path
        pdf_path = find_pdf_path(chunk.document_name)
        pdf_relative_path = get_pdf_relative_path(pdf_path) if pdf_path else None
        
        # Rich metadata for source attribution
        source_info = {
            "document_name": chunk.document_name,
            "page_number": chunk.page_number,
            "section_number": chunk.section_number,
            "similarity_score": round(similarity, 3),
            "chunk_id": chunk.chunk_id,
            "origin": chunk.origin,
            "domain": chunk.domain,
            "strictness": chunk.strictness,
            "method": chunk.method,
            "year": chunk.year,
            "source_type": chunk.source_type,
            "test_type": chunk.test_type,
            "metric": chunk.metric,
            "dummy_type": chunk.dummy_type,
            "pdf_path": str(pdf_path) if pdf_path else None,
            "pdf_relative_path": pdf_relative_path
        }
        sources.append(source_info)
        
        # Enhanced context with metadata tags (including passive safety)
        context_text = f"[Document: {chunk.document_name}"
        if chunk.origin:
            context_text += f", Origin: {chunk.origin}"
        if chunk.domain:
            context_text += f", Domain: {chunk.domain}"
        if chunk.method:
            context_text += f", Method: {chunk.method}"
        if chunk.test_type:
            context_text += f", Test_Type: {chunk.test_type}"
        if chunk.metric:
            context_text += f", Metric: {chunk.metric}"
        if chunk.dummy_type:
            context_text += f", Dummy_Type: {chunk.dummy_type}"
        context_text += f", Page: {chunk.page_number}"
        if chunk.section_number:
            context_text += f", Section: {chunk.section_number}"
        if chunk.year:
            context_text += f", Year: {chunk.year}"
        context_text += f"]\n{chunk.text}\n"
        context_parts.append(context_text)
    
    context = "\n---\n".join(context_parts)
    
    # Clean context - remove garbled text patterns
    import re
    # Remove common PDF extraction artifacts
    cleaned_context = re.sub(r'[^\w\s\-.,;:()\[\]{}%°±×÷≤≥≠≈∞∑∏∫√αβγδεθλμπστφω]', ' ', context)
    cleaned_context = re.sub(r'\s+', ' ', cleaned_context)  # Multiple spaces to single
    cleaned_context = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_context)  # Fix merged words
    
    # Get conversation history for context
    conversation_history = state.get("conversation_history", [])
    
    # Build conversation context
    history_context = ""
    if conversation_history:
        recent_context = []
        for msg in conversation_history[-6:]:  # Last 6 messages (3 exchanges)
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                recent_context.append(f"**Previous Question:** {content}")
            elif role == "assistant":
                recent_context.append(f"**Previous Answer:** {content[:200]}...")  # Truncate long answers
        if recent_context:
            history_context = "\n\n**Conversation History (for context):**\n" + "\n".join(recent_context)
    
    # Check if scenario reasoning is needed
    needs_scenario = state.get("needs_scenario_reasoning", False)
    
    # Generate answer using LLM with human-friendly format following The Three R's
    if needs_scenario:
        # Scenario-based reasoning prompt
        prompt = f"""You are an intelligent AI Safety Assistant. Your role is to provide thoughtful, well-formatted, scenario-based answers by combining information from verified safety documents with logical reasoning and clear summarization.

**User Question (Scenario-Based):** {question}
{history_context}

**Relevant Context from Safety Documents:**
{cleaned_context}

**THE THREE R's - CRITICAL REQUIREMENTS FOR SCENARIO REASONING:**

**1. RETRIEVAL (Did you find the right information?):**
- Use information from the provided context above as your foundation
- Identify relevant standards, requirements, or procedures from the documents
- If the context doesn't contain relevant information, you MUST say so explicitly
- Do NOT use information from your training data - only from the context provided

**2. REASONING (How do you apply the documents to this scenario?):**
- Think through the scenario step-by-step using the document information
- Apply the relevant standards, requirements, or procedures to the specific situation
- Explain your reasoning process clearly (e.g., "Based on the standard's requirements for..., in this scenario...")
- Use logical deduction to connect document requirements to the scenario
- Show how the document information applies to the specific case
- Use natural language, not technical citations

**3. REFUSAL (Admit when data is missing):**
- If the context doesn't contain relevant information for the scenario, say so clearly
- Example: "I found information about UNECE R94 requirements, but I don't see details about [specific aspect] needed to fully answer this scenario"
- NEVER make up numbers, values, or requirements
- If information is incomplete, state what you can determine and what's missing
- It's better to say "I can partially answer based on..." than to guess

**SCENARIO REASONING INSTRUCTIONS:**
1. **Think Logically**: Analyze the scenario step-by-step, applying document requirements to the specific situation
2. **Summarize Key Points**: Provide a clear summary of relevant requirements before applying them
3. **Reason Through**: Explain your thought process - how document requirements apply to this scenario
4. **Stay Grounded**: All conclusions must be traceable to the provided context - NO hallucination
5. **Format Beautifully**: Use markdown formatting (headers, bullet points, bold text) for clarity
6. **Be Conversational**: Write naturally, as if explaining to a colleague
7. **Remember Context**: Reference previous conversation if relevant, but stay focused on current question
8. **DO NOT mention document names, page numbers, or file paths in your answer text**

**VERY IMPORTANT OUTPUT RULES (CHATGPT-STYLE READABILITY):**

- Do NOT write long paragraphs
- Write short lines (1–2 sentences max per line)
- Use clear headings
- Use bullet points where possible
- Leave a blank line between sections
- Prefer step-by-step formatting
- If explaining, explain one idea per line
- If listing values, put each value on a new line
- Keep the tone clear and readable like ChatGPT
- If regulations are cited, put each reference on a separate line
- Do not embed references inside paragraphs
- Your goal is maximum readability, not verbosity

**OUTPUT FORMAT (MANDATORY - YOU MUST FOLLOW THIS EXACT STRUCTURE):**

You MUST respond in this exact format. Do not deviate:

### ✅ Simple Answer

- <one short sentence>
- <one short sentence>

### 📘 Regulation Requirement

- Keyword: <exact wording>
- Limit: <value>
- Condition: <if any>

### 🧮 Analysis / Calculation (if applicable)

Step 1:
- ...

Step 2:
- ...

### 🔗 References

- Regulation: <name>
- Clause: <number>
- Page: <page>

**CRITICAL FORMATTING RULES:**
- Start with "### ✅ Simple Answer" (exact text with emoji)
- Then "### 📘 Regulation Requirement" (exact text with emoji)
- Use bullet points (-) for ALL lists
- Put each bullet point on a NEW line
- Use blank lines between sections
- Use **bold** for regulation names, limits, and key terms
- Keep each idea on a separate line
- Do NOT write paragraphs - use bullet points instead

**Example Good Scenario Answer:**
"Based on the UNECE R94 requirements for frontal collision protection, in a scenario where a vehicle undergoes a 40% offset deformable barrier test, the HIC value must not exceed 1000 for a 50th percentile male dummy. The standard specifies that this measurement is taken during the test procedure, and if the value exceeds the threshold, the vehicle would not meet the regulatory requirement. The test protocol requires specific dummy positioning and instrumentation setup as defined in the regulation."

**Example Good Refusal:**
"I found information about UNECE R94 frontal collision requirements, but to fully answer this scenario, I would need details about [specific aspect] which I don't see in the available documents. Based on what I have, I can tell you that..."

**CRITICAL:**
- All reasoning must be grounded in the provided document context
- Do NOT hallucinate or make up requirements not in the documents
- If the context contains garbled text, skip it and use only clear information
- Think logically but stay within the bounds of the document information

**Answer:**"""
    else:
        # Standard factual retrieval prompt with enhanced formatting and reasoning
        prompt = f"""You are an intelligent AI Safety Assistant. Your role is to provide clear, well-formatted, logically reasoned answers based on verified safety documents. You can summarize, analyze, and think through problems while staying grounded in the provided documents.

**User Question:** {question}
{history_context}

**Relevant Context from Safety Documents:**
{cleaned_context}

**THE THREE R's - CRITICAL REQUIREMENTS:**

**1. RETRIEVAL (Did you find the right information?):**
- Only use information that is CLEARLY present in the provided context above
- If the context doesn't contain the answer, you MUST say so explicitly
- Do NOT use information from your training data - only from the context provided
- Verify that the context actually answers the question before responding

**2. REASONING (Think through and explain your answer):**
- **Summarize**: Provide a clear summary of the key information found
- **Analyze**: Think through what this information means in context
- **Explain**: Explain WHERE in the documents you found the information (e.g., "According to the test procedures section..." or "The table shows...")
- **Connect**: If citing a specific value, explain the context and why it matters (e.g., "The standard specifies... because...")
- **Reason**: Use logical thinking to connect document information to the question
- **Remember**: Reference previous conversation context if relevant
- Use natural, conversational language - not technical citations

**3. REFUSAL (Admit when data is missing):**
- If the context doesn't contain the answer, say so clearly
- Example: "I found information about UNECE R94, but I don't see details about [specific thing] in the available documents"
- NEVER make up numbers, values, or requirements
- If information is incomplete or unclear, state this explicitly
- It's better to say "I cannot find this information" than to guess

**ADDITIONAL INSTRUCTIONS:**
1. **Think First**: Before answering, think through the question logically
2. **Summarize**: Start with a brief summary of what you found
3. **Format Beautifully**: Use markdown (bold, bullet points, headers) for better readability
4. **Be Conversational**: Write naturally, as if explaining to a colleague
5. **Stay Grounded**: All information must come from the provided context - NO hallucination
6. **Remember Context**: Reference previous conversation if relevant
7. **Write Clearly**: Use simple, clear language that anyone can understand
8. **Be Concise**: Get to the point but provide enough detail
9. **DO NOT mention document names, page numbers, or file paths in your answer text**
10. If context contains garbled text, skip it and use only clear information

**VERY IMPORTANT OUTPUT RULES (CHATGPT-STYLE READABILITY):**

- Do NOT write long paragraphs
- Write short lines (1–2 sentences max per line)
- Use clear headings
- Use bullet points where possible
- Leave a blank line between sections
- Prefer step-by-step formatting
- If explaining, explain one idea per line
- If listing values, put each value on a new line
- Keep the tone clear and readable like ChatGPT
- If regulations are cited, put each reference on a separate line
- Do not embed references inside paragraphs
- Your goal is maximum readability, not verbosity

**OUTPUT FORMAT (MANDATORY - YOU MUST FOLLOW THIS EXACT STRUCTURE):**

You MUST respond in this exact format. Do not deviate:

### ✅ Simple Answer

- <one short sentence>
- <one short sentence>

### 📘 Regulation Requirement

- Keyword: <exact wording>
- Limit: <value>
- Condition: <if any>

### 🧮 Analysis / Calculation (if applicable)

Step 1:
- ...

Step 2:
- ...

### 🔗 References

- Regulation: <name>
- Clause: <number>
- Page: <page>

**CRITICAL FORMATTING RULES:**
- Start with "### ✅ Simple Answer" (exact text with emoji)
- Then "### 📘 Regulation Requirement" (exact text with emoji)
- Use bullet points (-) for ALL lists
- Put each bullet point on a NEW line
- Use blank lines between sections
- Use **bold** for regulation names, limits, and key terms
- Keep each idea on a separate line
- Do NOT write paragraphs - use bullet points instead

**Example Good Answer (with all 3 R's):**
"The maximum allowable HIC (Head Injury Criterion) for a 50th percentile male dummy in UNECE R94 is 1000. This value is specified in the performance criteria section of the regulation, which defines the injury thresholds for frontal collision tests. The test procedures require measuring HIC during the 40% offset deformable barrier test to ensure occupant safety."

**Example Good Refusal:**
"I found information about UNECE R94 frontal collision requirements, but I don't see specific details about the IIHS 2025 protocol in the available documents. The documents I have access to cover UNECE regulations and Euro NCAP protocols, but not the IIHS 2025 standard."

**Answer:**"""
    
    answer = ""
    llm_error = None
    
    # Try Anthropic Claude first with model fallback
    if claude_client:
        # List of models to try in order (newest to oldest)
        claude_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
        
        # Add user-specified model first if not already in list
        if LLM_MODEL not in claude_models:
            claude_models.insert(0, LLM_MODEL)
        else:
            # Move user model to front
            claude_models.remove(LLM_MODEL)
            claude_models.insert(0, LLM_MODEL)
        
        print(f"🔍 Attempting to use Anthropic Claude with models: {', '.join(claude_models[:3])}...")
        
        for model_name in claude_models:
            try:
                print(f"🔄 Trying model: {model_name}")
                
                # Build messages with conversation history for better context
                messages = []
                
                # Add conversation history (last 4 exchanges for context)
                if conversation_history:
                    for msg in conversation_history[-8:]:  # Last 8 messages (4 exchanges)
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role in ["user", "assistant"]:
                            messages.append({"role": role, "content": content})
                
                # Add current prompt
                messages.append({"role": "user", "content": prompt})
                
                response = claude_client.messages.create(
                    model=model_name,
                    max_tokens=3000,  # Increased for better formatted answers
                    temperature=0.2,  # Lower temperature for more focused, less hallucination
                    messages=messages
                )
                # Extract answer from response with validation
                if response.content and len(response.content) > 0:
                    answer = response.content[0].text.strip()
                    if answer:
                        print(f"✅ Successfully got answer from Claude model: {model_name} (length: {len(answer)} chars)")
                        break
                    else:
                        print(f"⚠️  Empty answer from {model_name}, trying next model...")
                        continue
                else:
                    print(f"⚠️  No content in response from {model_name}, trying next model...")
                    continue
            except Exception as e:
                error_msg = str(e)
                llm_error = error_msg
                # If it's a 404 (model not found), try next model
                if "404" in error_msg or "not_found" in error_msg.lower():
                    print(f"⚠️  Model {model_name} not found (404), trying next model...")
                    continue
                elif "401" in error_msg or "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                    print(f"❌ Authentication error with {model_name}: {error_msg}")
                    # Don't try more models if auth fails
                    break
                elif "429" in error_msg or "rate_limit" in error_msg.lower():
                    print(f"⚠️  Rate limit error with {model_name}: {error_msg}")
                    # Don't try more models if rate limited
                    break
                else:
                    print(f"⚠️  Claude error with {model_name}: {error_msg}")
                    # For other errors, try next model
                    continue
    else:
        print(f"⚠️  Claude client not initialized. LLM_PROVIDER={LLM_PROVIDER}, ANTHROPIC_API_KEY={'set' if ANTHROPIC_API_KEY else 'NOT SET'}")
        llm_error = "Claude client not initialized. Check ANTHROPIC_API_KEY in .env file."
    
    # Fallback to OpenAI with model fallback
    if not answer and openai_client:
        print(f"🔄 Falling back to OpenAI...")
        openai_models = [
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
        
        for model_name in openai_models:
            try:
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3
                )
                answer = response.choices[0].message.content.strip()
                print(f"✅ Used OpenAI model: {model_name}")
                break
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "not_found" in error_msg.lower():
                    print(f"⚠️  Model {model_name} not found, trying next...")
                    continue
                else:
                    print(f"⚠️  OpenAI error with {model_name}: {e}")
                    break
    
    # Fallback if no LLM available - Format in structured ChatGPT-style
    if not answer:
        print(f"⚠️  No LLM answer generated. Error details: {llm_error if llm_error else 'Unknown error'}")
        # Simple extraction-based answer with heavy cleaning and structured format
        if retrieved_chunks and len(retrieved_chunks) > 0:
            top_chunk, top_similarity = retrieved_chunks[0]
            # Aggressive text cleaning
            clean_text = top_chunk.text
            
            # Remove patterns like "F z F z" (single letters separated by spaces)
            clean_text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+\1\s+\2\b', '', clean_text)
            clean_text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b(?=\s+[A-Za-z])', r'\1\2', clean_text)  # Fix split words
            
            # Remove excessive special characters and garbled patterns
            clean_text = re.sub(r'[^\w\s\-.,;:()\[\]{}%°±×÷≤≥≠≈∞∑∏∫√αβγδεθλμπστφω]', ' ', clean_text)
            # Fix merged words
            clean_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean_text)
            # Remove excessive spaces
            clean_text = re.sub(r'\s+', ' ', clean_text)
            # Remove garbled patterns (repeated characters, weird spacing)
            clean_text = re.sub(r'([a-zA-Z])\1{2,}', r'\1', clean_text)  # Remove repeated letters
            clean_text = re.sub(r'([^\w\s])\1{2,}', ' ', clean_text)  # Remove repeated special chars
            
            # Remove single-letter "words" that are likely garbled (but keep "a", "I" in context)
            words = clean_text.split()
            clean_words = []
            for i, word in enumerate(words):
                # Skip isolated single letters (likely garbled)
                if len(word) == 1 and word.isalpha() and word.lower() not in ['a', 'i']:
                    # Check if it's between two words (likely garbled)
                    if i > 0 and i < len(words) - 1:
                        continue
                clean_words.append(word)
            clean_text = ' '.join(clean_words)
            
            # Extract meaningful sentences (at least 15 characters, mostly readable)
            sentences = re.split(r'[.!?]\s+', clean_text)
            meaningful_sentences = []
            for s in sentences:
                s = s.strip()
                if len(s) >= 15:
                    # Check readability (at least 60% alphanumeric)
                    alnum_ratio = sum(1 for c in s if c.isalnum() or c.isspace()) / len(s) if s else 0
                    # Check for too many single-letter words (garbled pattern)
                    words_in_s = s.split()
                    single_letter_words = sum(1 for w in words_in_s if len(w) == 1 and w.isalpha() and w.lower() not in ['a', 'i'])
                    if len(words_in_s) > 0 and single_letter_words / len(words_in_s) <= 0.2:  # Max 20% single letters
                        if alnum_ratio >= 0.6:
                            meaningful_sentences.append(s)
            
            if meaningful_sentences:
                # Format in structured ChatGPT-style format
                answer = "### ✅ Simple Answer\n\n"
                answer += f"- {meaningful_sentences[0]}\n"
                if len(meaningful_sentences) > 1:
                    answer += f"- {meaningful_sentences[1]}\n"
                
                answer += "\n### 📘 Regulation Requirement\n\n"
                # Try to extract key values/limits from the text
                for sentence in meaningful_sentences[2:5]:
                    # Look for numbers that might be limits
                    if any(char.isdigit() for char in sentence):
                        answer += f"- {sentence}\n"
                        break
                
                answer += "\n### 🔗 References\n\n"
                answer += f"- Regulation: {top_chunk.regulation or top_chunk.document_name}\n"
                if top_chunk.clause:
                    answer += f"- Clause: {top_chunk.clause}\n"
                answer += f"- Page: {top_chunk.page_number}\n"
            else:
                # If no clean sentences, provide a refusal message in structured format
                answer = "### ✅ Simple Answer\n\n"
                answer += "- I found information in the documents, but it appears to be unclear or garbled.\n"
                answer += "- Please try rephrasing your question or check if the documents contain clear information about this topic.\n"
        else:
            answer = "### ✅ Simple Answer\n\n"
            answer += "- I couldn't find relevant information in the available safety documents for your question.\n"
            answer += "- Please try rephrasing your question or check if the specific document you're looking for is available.\n"
    
    # CRITICAL: Check if answer has structured markdown format - if so, preserve it!
    import re
    has_markdown_structure = "###" in answer or "**" in answer or "##" in answer
    
    if not has_markdown_structure:
        # Only apply aggressive cleaning to non-structured answers
        # CRITICAL: Remove patterns like "F z F z" FIRST (before other cleaning)
        answer = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+\1\s+\2\b', '', answer)
        answer = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b(?=\s+[A-Za-z])', r'\1\2', answer)  # Fix split words like "F z" -> "Fz" if followed by letter
        
        # Remove isolated single letters (except a, I) BEFORE other cleaning
        words = answer.split()
        clean_words = []
        for i, word in enumerate(words):
            if len(word) == 1 and word.isalpha() and word.lower() not in ['a', 'i']:
                if i > 0 and i < len(words) - 1:
                    continue
            clean_words.append(word)
        answer = ' '.join(clean_words)
        
        # Remove common citation patterns from answer
        answer = re.sub(r'\[Document[^\]]+\]', '', answer)
        answer = re.sub(r'\(Document[^\)]+\)', '', answer)
        answer = re.sub(r'Page \d+', '', answer)
        answer = re.sub(r'Section [^\s]+', '', answer)
        answer = re.sub(r'\([^\)]*Origin[^\)]*\)', '', answer)
        answer = re.sub(r'\([^\)]*Method[^\)]*\)', '', answer)
        
        # Aggressive garbled text removal
        answer = re.sub(r'([a-zA-Z])\1{2,}', r'\1', answer)  # Remove repeated letters (aaa -> a)
        answer = re.sub(r'([^\w\s])\1{2,}', ' ', answer)  # Remove repeated special chars
        answer = re.sub(r'_{3,}', ' ', answer)  # Remove multiple underscores
        answer = re.sub(r'__+', ' ', answer)  # Remove double+ underscores
        answer = re.sub(r'\s+_\s+', ' ', answer)  # Remove standalone underscores
        answer = re.sub(r'_\d+', ' ', answer)  # Remove underscore followed by number
        
        # Fix merged words
        answer = re.sub(r'([a-z])([A-Z])', r'\1 \2', answer)
        
        # Remove isolated single letters that are likely garbled (but keep "a", "I")
        words = answer.split()
        clean_words = []
        for i, word in enumerate(words):
            # Skip isolated single letters (likely garbled) except common words
            if len(word) == 1 and word.isalpha() and word.lower() not in ['a', 'i']:
                # Check if it's between two words (likely garbled)
                if i > 0 and i < len(words) - 1:
                    continue
            # Skip words with too many special chars
            if word and sum(1 for c in word if not c.isalnum()) > len(word) * 0.5:
                continue
            clean_words.append(word)
        answer = ' '.join(clean_words)
    else:
        # For structured markdown, only do minimal cleaning - preserve structure!
        # Only remove obvious garbled patterns that won't break markdown
        answer = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+\1\s+\2\b', '', answer)  # Remove "F z F z" patterns
        # Don't remove other patterns that might be part of valid markdown
    
    # ONLY clean if answer doesn't contain structured format markers - preserve markdown structure
    has_markdown = "###" in answer or "**" in answer or "##" in answer
    
    if not has_markdown:
        # Only apply aggressive cleaning if answer doesn't have markdown structure
        # Clean up text (less aggressive for structured answers)
        answer = re.sub(r'[^\w\s\-.,;:()\[\]{}%°±×÷≤≥≠≈∞∑∏∫√αβγδεθλμπστφω]', ' ', answer)
        answer = re.sub(r'\s+', ' ', answer)
        
        # Remove sentences that are mostly garbled (less than 60% alphanumeric)
        sentences = re.split(r'[.!?]\s+', answer)
        clean_sentences = []
        for sentence in sentences:
            if sentence.strip() and len(sentence.strip()) >= 10:
                alnum_ratio = sum(1 for c in sentence if c.isalnum() or c.isspace()) / len(sentence) if sentence else 0
                # Check for too many single-letter "words" (garbled pattern)
                words_in_sentence = sentence.split()
                single_letter_words = sum(1 for w in words_in_sentence if len(w) == 1 and w.isalpha() and w.lower() not in ['a', 'i'])
                if len(words_in_sentence) > 0 and single_letter_words / len(words_in_sentence) > 0.3:
                    continue  # Skip sentences with >30% single-letter words
                if alnum_ratio >= 0.6:  # At least 60% readable
                    clean_sentences.append(sentence.strip())
        
        if clean_sentences:
            answer = '. '.join(clean_sentences) + '.'
        else:
            # If no clean sentences, try to extract meaningful phrases
            words = answer.split()
            phrases = []
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                if sum(1 for c in phrase if c.isalnum()) / len(phrase) >= 0.6:
                    phrases.append(phrase)
            if phrases:
                answer = '. '.join(phrases[:3]) + '.'
            else:
                answer = "I found some information in the documents, but it appears to be unclear or garbled. Please try rephrasing your question or check if the documents contain clear information about this topic."
    
    # Final cleanup - ensure no "F z" patterns remain and proper spacing
    answer = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+\1\s+\2\b', '', answer)
    answer = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', r'\1\2', answer)  # Merge remaining "F z" -> "Fz"
    answer = re.sub(r'\s+', ' ', answer)
    answer = answer.strip()
    
    # Ensure structured format if answer doesn't have it
    if answer and "### ✅" not in answer and "### Simple Answer" not in answer and "### 📘" not in answer:
        # Try to format as structured - split into sections
        lines = [l.strip() for l in answer.split('\n') if l.strip()]
        if lines:
            formatted = "### ✅ Simple Answer\n\n"
            # First 2-3 lines as simple answer
            for line in lines[:3]:
                if len(line) > 10:
                    formatted += f"- {line}\n"
            formatted += "\n### 📘 Regulation Requirement\n\n"
            # Next lines as regulation
            for line in lines[3:6]:
                if len(line) > 10:
                    formatted += f"- {line}\n"
            if len(lines) > 6:
                formatted += "\n### 🔗 References\n\n"
                formatted += "- See sources below\n"
            answer = formatted
    
    # Add disclaimer at the end (after formatting)
    answer = SafetyGuardrails.add_disclaimer(answer)
    
    # Filter sources by quality - only show high-quality sources (similarity >= 0.65)
    high_quality_sources = [s for s in sources if s.get("similarity_score", 0.0) >= 0.65]
    
    # Extract only sources actually cited in the answer
    from source_extractor import extract_cited_sources
    cited_sources = extract_cited_sources(answer, high_quality_sources)
    
    # If no sources were cited, use top high-quality sources by similarity
    if not cited_sources:
        cited_sources = sorted(high_quality_sources, key=lambda x: x.get("similarity_score", 0.0), reverse=True)[:min(3, len(high_quality_sources))]
    
    # Calculate confidence
    similarity_scores = [sim for _, sim in retrieved_chunks]
    confidence_score, confidence_level = ConfidenceScorer.calculate_confidence(similarity_scores)
    
    return {
        **state,
        "answer": answer,
        "sources": cited_sources,  # Only return sources actually used
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "workflow_stage": "complete"
    }

def create_safety_copilot_workflow(vector_store: SafetyVectorStore):
    """Create LangGraph workflow for safety copilot with synthesis support"""
    
    workflow = StateGraph(SafetyCopilotState)
    
    # Add nodes
    workflow.add_node("guardrail", safety_guardrail_agent)
    workflow.add_node("retrieval", lambda state: retrieval_agent(state, vector_store))
    workflow.add_node("synthesis", synthesis_agent_wrapper)
    workflow.add_node("generation", answer_generation_agent)
    
    # Define edges
    workflow.set_entry_point("guardrail")
    
    def refusal_response_agent(state: SafetyCopilotState) -> SafetyCopilotState:
        """Generate refusal response"""
        return {
            **state,
            "answer": f"⚠️ **Request Refused**\n\n{state.get('refusal_reason', 'This request cannot be processed.')}\n\n{SafetyGuardrails.add_disclaimer('')}",
            "sources": [],
            "confidence_score": 0.0,
            "confidence_level": "low",
            "workflow_stage": "complete"
        }
    
    workflow.add_node("refusal", refusal_response_agent)
    
    def route_after_guardrail(state: SafetyCopilotState) -> str:
        if state.get("should_refuse", False):
            return "refuse"
        # Both synthesis and scenario reasoning need retrieval first
        return "retrieval"
    
    def route_after_retrieval(state: SafetyCopilotState) -> str:
        """Route to synthesis or standard generation"""
        if state.get("needs_synthesis", False):
            return "synthesis"
        return "generation"
    
    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "refuse": "refusal",
            "retrieval": "retrieval"
        }
    )
    
    workflow.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "synthesis": "synthesis",
            "generation": "generation"
        }
    )
    
    workflow.add_edge("synthesis", END)
    workflow.add_edge("generation", END)
    workflow.add_edge("refusal", END)
    
    return workflow.compile()

