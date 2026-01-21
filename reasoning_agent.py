"""
Reasoning Agent for Question Type Identification and Calculations
Identifies question type and performs logical reasoning
"""
from typing import Dict, List, Optional, Tuple
from config import LLM_PROVIDER, LLM_MODEL, ANTHROPIC_API_KEY, OPENAI_API_KEY
import re

# Initialize LLM clients
claude_client = None
openai_client = None

if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
    from anthropic import Anthropic
    claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
elif LLM_PROVIDER == "openai" and OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

QUESTION_TYPES = [
    "Definition",  # "What is HIC?"
    "Requirement",  # "What is the HIC limit?"
    "Comparison",  # "Compare UNECE R94 and Euro NCAP"
    "Calculation",  # "If HIC is 1200, is it compliant?"
    "Compliance Check"  # "Does this meet the requirement?"
]

def identify_question_type(question: str) -> Dict[str, any]:
    """
    Identify question type and extract required information
    Returns:
        {
            "question_type": "Definition|Requirement|Comparison|Calculation|Compliance Check",
            "requires_calculation": bool,
            "regulation_mentioned": Optional[str],
            "values_mentioned": List[str],
            "needs_comparison": bool
        }
    """
    question_lower = question.lower()
    
    # Check for question type keywords
    is_definition = any(kw in question_lower for kw in ["what is", "define", "explain", "meaning of"])
    is_requirement = any(kw in question_lower for kw in ["limit", "threshold", "maximum", "minimum", "requirement", "standard"])
    is_comparison = any(kw in question_lower for kw in ["compare", "difference", "versus", "vs", "between"])
    is_calculation = any(kw in question_lower for kw in ["calculate", "compute", "if", "when", "given"])
    is_compliance = any(kw in question_lower for kw in ["compliant", "meet", "satisfy", "pass", "fail"])
    
    # Extract regulation mentions
    regulation = None
    regulation_patterns = [
        r'UNECE\s+R(\d+)',
        r'R(\d+)',
        r'FMVSS\s+(\d+)',
        r'Euro\s+NCAP',
        r'ISO\s+26262'
    ]
    for pattern in regulation_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            if "UNECE" in pattern or "R" in pattern:
                regulation = f"UNECE R{match.group(1)}"
            elif "FMVSS" in pattern:
                regulation = f"FMVSS {match.group(1)}"
            elif "Euro" in pattern:
                regulation = "Euro NCAP"
            elif "ISO" in pattern:
                regulation = "ISO 26262"
            break
    
    # Extract values/numbers mentioned
    values = re.findall(r'\d+(?:\.\d+)?', question)
    
    # Determine primary question type
    if is_compliance:
        question_type = "Compliance Check"
    elif is_calculation:
        question_type = "Calculation"
    elif is_comparison:
        question_type = "Comparison"
    elif is_requirement:
        question_type = "Requirement"
    elif is_definition:
        question_type = "Definition"
    else:
        question_type = "Requirement"  # Default
    
    return {
        "question_type": question_type,
        "requires_calculation": is_calculation or is_compliance,
        "regulation_mentioned": regulation,
        "values_mentioned": values,
        "needs_comparison": is_comparison
    }

def perform_calculation(question: str, context: str) -> Optional[str]:
    """
    Perform calculations if needed (e.g., compliance checks)
    Uses LLM to reason through calculations
    """
    if not claude_client and not openai_client:
        return None
    
    prompt = f"""You are a safety regulation calculation assistant.

**Question:** {question}

**Context from Regulations:**
{context}

**Task:**
If the question requires a calculation or compliance check, perform it step-by-step.

Examples:
- "If HIC is 1200, is it compliant?" → Check if 1200 ≤ limit (e.g., 1000)
- "What is the margin if HIC is 850?" → Calculate: limit - actual = margin

**Output Format:**
If calculation is needed, provide:
1. **Given Values**: Extract from question
2. **Regulation Limit**: Extract from context
3. **Calculation**: Step-by-step
4. **Result**: Answer

If no calculation needed, return "N/A"

**Calculation:**"""

    try:
        if claude_client:
            response = claude_client.messages.create(
                model=LLM_MODEL if LLM_MODEL.startswith("claude") else "claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text.strip()
        elif openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
        else:
            return None
        
        if result and result != "N/A":
            return result
        return None
    except Exception as e:
        print(f"⚠️ Calculation failed: {e}")
        return None

def extract_exact_limits(context: str) -> List[Dict[str, str]]:
    """
    Extract exact limits, thresholds, and conditions from context
    Returns list of {regulation, clause, limit, condition}
    """
    limits = []
    
    # Pattern for limits (e.g., "≤ 1000", "shall not exceed 1000", "maximum 1000")
    limit_patterns = [
        r'(?:shall\s+not\s+exceed|maximum|max|limit|threshold|≤|<=)\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:shall\s+not\s+be\s+exceeded|is\s+the\s+maximum)',
    ]
    
    for pattern in limit_patterns:
        matches = re.finditer(pattern, context, re.IGNORECASE)
        for match in matches:
            limit_value = match.group(1)
            # Try to extract regulation and clause from surrounding context
            regulation_match = re.search(r'(UNECE\s+R\d+|FMVSS\s+\d+|Euro\s+NCAP)', context[max(0, match.start()-100):match.start()], re.IGNORECASE)
            clause_match = re.search(r'(\d+\.\d+(?:\.\d+)*)', context[max(0, match.start()-50):match.start()])
            
            limits.append({
                "limit": limit_value,
                "regulation": regulation_match.group(1) if regulation_match else None,
                "clause": clause_match.group(1) if clause_match else None,
                "full_text": match.group(0)
            })
    
    return limits


