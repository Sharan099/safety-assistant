"""
Domain Classifier for Safety Questions
Classifies user questions into domains for better retrieval
"""
import re
from typing import List, Dict

class DomainClassifier:
    """Classifies safety questions into domains"""
    
    DOMAIN_KEYWORDS = {
        "Passive Safety": [
            "passive safety", "crash", "collision", "impact", "airbag", "seatbelt",
            "restraint", "dummy", "hict", "chest deflection", "tibia", "intrusion",
            "frontal", "side impact", "pole test", "pedestrian", "post-crash",
            "r94", "r137", "fmvss 208", "ncap", "euro ncap", "offset barrier",
            "deformable barrier", "rigid barrier", "mpdb", "worldsid", "hybrid-iii",
            "thor", "hic", "injury criteria", "occupant protection"
        ],
        "Functional Safety": [
            "functional safety", "iso 26262", "asil", "hara", "safety goal",
            "safety requirement", "safety concept", "safety lifecycle",
            "fusa", "safety integrity", "random failure", "systematic failure"
        ],
        "Cybersecurity": [
            "cybersecurity", "r155", "cyber security", "threat", "vulnerability",
            "attack", "security", "unauthorized access", "data protection"
        ],
        "ADAS": [
            "adas", "advanced driver assistance", "autonomous", "self-driving",
            "lane keeping", "adaptive cruise", "collision avoidance", "aeb"
        ],
        "Driver Monitoring": [
            "driver monitoring", "dms", "driver attention", "fatigue",
            "driver state", "driver distraction", "eye tracking"
        ],
        "Software Update": [
            "software update", "r156", "ota", "over the air", "firmware update",
            "software version", "update management"
        ],
        "Validation": [
            "validation", "testing", "test case", "verification", "test scenario",
            "test procedure", "test coverage"
        ]
    }
    
    @classmethod
    def classify_domain(cls, question: str) -> List[str]:
        """
        Classify question into one or more domains
        Returns list of domains ordered by relevance
        """
        question_lower = question.lower()
        domain_scores = {}
        
        for domain, keywords in cls.DOMAIN_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in question_lower:
                    score += 1
            if score > 0:
                domain_scores[domain] = score
        
        # Sort by score (descending)
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, _ in sorted_domains]
    
    @classmethod
    def get_primary_domain(cls, question: str) -> str:
        """Get the primary (most relevant) domain"""
        domains = cls.classify_domain(question)
        return domains[0] if domains else "General Safety"
    
    @classmethod
    def needs_synthesis(cls, question: str) -> bool:
        """Detect if question requires synthesis across multiple documents"""
        question_lower = question.lower()
        synthesis_keywords = [
            "compare", "difference", "conflict", "synthesize", "across",
            "multiple", "both", "versus", "vs", "versus", "between",
            "all", "together", "combine", "integrate", "unified",
            "table", "tabular", "data", "values", "threshold"
        ]
        return any(keyword in question_lower for keyword in synthesis_keywords)
    
    @classmethod
    def needs_scenario_reasoning(cls, question: str) -> bool:
        """
        Determine if the question requires scenario-based reasoning,
        analysis, or application of knowledge to a specific situation.
        """
        question_lower = question.lower()
        
        scenario_keywords = [
            "what if", "scenario", "situation", "case", "example",
            "how would", "what happens", "what should", "recommend",
            "analyze", "evaluate", "assess", "determine", "decide",
            "apply", "implement", "design", "plan", "strategy",
            "when", "if", "suppose", "imagine", "consider"
        ]
        
        # Check for scenario indicators
        if any(keyword in question_lower for keyword in scenario_keywords):
            return True
        
        # Check for conditional/hypothetical language
        conditional_patterns = [
            r"if\s+\w+\s+(happens|occurs|fails|works)",
            r"what\s+(should|would|could|might)",
            r"how\s+(should|would|could|might)",
            r"in\s+(case|situation|scenario|event)"
        ]
        
        import re
        for pattern in conditional_patterns:
            if re.search(pattern, question_lower):
                return True
        
        return False

