"""
File: agents/fact_verification_agent.py
Fact Verification Agent: Validates facts across multiple sources.
"""
import os
import sys
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.base_agent import BaseAgent
from graph.state import AgentState, add_error_to_state


class FactVerificationAgent(BaseAgent):
    """
    Fact Verification Agent: Validates facts across research and documents.
    
    Responsibilities:
    1. Detect contradictions between sources
    2. Calculate verification confidence
    3. Flag unreliable information
    4. Provide verification summary
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize FactVerificationAgent."""
        super().__init__(
            name="fact_verification_agent",
            description="Validates facts and detects contradictions",
            timeout_seconds=300,
            max_retries=2
        )
        
        self.config = config or {}
        
        # Contradiction detection thresholds
        self.contradiction_threshold = self.config.get("contradiction_threshold", 0.7)
        self.min_sources_for_verification = self.config.get("min_sources", 2)
        
        print(f"✅ FactVerificationAgent initialized")
    
    async def _execute_impl(self, state: AgentState) -> AgentState:
        """
        Execute fact verification on research findings and documents.
        """
        print(f"🔍 FactVerificationAgent validating facts...")
        
        try:
            # Extract information from state
            research_results = state.get("research_results", {})
            documents = state.get("retrieved_documents", [])
            
            if not research_results and not documents:
                print("⚠️  No information to verify")
                return state
            
            # Extract facts from research findings
            research_facts = self._extract_facts_from_research(research_results)
            
            # Extract facts from documents
            document_facts = self._extract_facts_from_documents(documents)
            
            # Combine all facts
            all_facts = research_facts + document_facts
            
            if not all_facts:
                print("⚠️  No facts extracted for verification")
                return state
            
            print(f"   Extracted {len(all_facts)} facts for verification")
            
            # Group similar facts
            fact_groups = self._group_similar_facts(all_facts)
            
            # Verify each fact group
            verification_results = []
            contradiction_report = []
            
            for group_id, facts in fact_groups.items():
                if len(facts) < 2:
                    # Single source facts - mark as unverified
                    verification_results.append({
                        "fact_id": f"single_{group_id}",
                        "content": facts[0]["content"],
                        "sources": [facts[0]["source_info"]], # FIX: Wrap single source info in list
                        "verification_status": "single_source",
                        "confidence": 0.3,
                        "notes": "Only one source found"
                    })
                    continue
                
                # Multiple sources - check for contradictions
                verification_result = self._verify_fact_group(facts)
                
                if verification_result["has_contradiction"]:
                    contradiction_report.append({
                        "fact_group": group_id,
                        "content": verification_result["representative_fact"],
                        "contradiction_details": verification_result["contradiction_details"],
                        "sources_involved": verification_result["sources_involved"]
                    })
                
                verification_results.append({
                    "fact_id": f"group_{group_id}",
                    "content": verification_result["representative_fact"],
                    "sources": verification_result["all_sources"],
                    "verification_status": verification_result["status"],
                    "confidence": verification_result["confidence"],
                    "contradictions_found": verification_result["has_contradiction"],
                    "source_count": len(facts)
                })
            
            # Calculate overall verification confidence
            overall_confidence = self._calculate_overall_confidence(verification_results)
            
            # Update state with verification results
            state["verified_facts"] = verification_results
            state["contradiction_report"] = contradiction_report
            state["verification_confidence"] = overall_confidence
            
            # Log results
            print(f"✅ Verification complete:")
            print(f"   Verified facts: {len(verification_results)}")
            print(f"   Contradictions found: {len(contradiction_report)}")
            print(f"   Overall confidence: {overall_confidence:.2f}")
            
            if contradiction_report:
                print(f"   ⚠️  Contradictions detected in {len(contradiction_report)} fact groups")
                for i, contradiction in enumerate(contradiction_report[:2]):  # Show first 2
                    print(f"     {i+1}. {contradiction['content'][:80]}...")
            
            return state
            
        except Exception as e:
            error_msg = f"Fact verification failed: {str(e)}"
            print(f"❌ {error_msg}")
            # Import traceback to see full error details if needed
            import traceback
            traceback.print_exc()
            state = add_error_to_state(state, self.name, error_msg)
            return state
    
    def _extract_facts_from_research(self, research_results: Dict) -> List[Dict]:
        """Extract facts from research findings."""
        facts = []
        if not research_results:
            return facts
        
        findings = research_results.get("findings", [])
        
        for i, finding in enumerate(findings):
            fact = {
                "id": f"research_{i}",
                "content": finding.get("content", ""),
                "source_type": finding.get("source_type", "unknown"),
                "source_url": finding.get("source_url", ""),
                "credibility": finding.get("credibility", 0.5),
                "recency": finding.get("recency", "unknown"),
                "key_metrics": finding.get("key_metrics", []),
                "extraction_method": "research_finding"
            }
            
            # Extract numeric claims
            numeric_claims = self._extract_numeric_claims(fact["content"])
            if numeric_claims:
                fact["numeric_claims"] = numeric_claims
            
            facts.append(fact)
        return facts
    
    def _extract_facts_from_documents(self, documents: List) -> List[Dict]:
        """Extract facts from retrieved documents."""
        facts = []
        if not documents:
            return facts
        
        for i, doc in enumerate(documents):
            # Extract key sentences (simple approach)
            content = doc.page_content
            sentences = content.split('. ')
            
            # Take first 3 sentences as key facts (simplified)
            key_sentences = sentences[:3]
            
            for j, sentence in enumerate(key_sentences):
                if len(sentence.strip()) > 20:  # Minimum length
                    fact = {
                        "id": f"doc_{i}_{j}",
                        "content": sentence.strip() + ".",
                        "source_type": doc.metadata.get("source_type", "document") if hasattr(doc, 'metadata') else "document",
                        "source_url": doc.metadata.get("source", "") if hasattr(doc, 'metadata') else "",
                        "credibility": 0.6,  # Default for documents
                        "recency": doc.metadata.get("date", "unknown") if hasattr(doc, 'metadata') else "unknown",
                        "extraction_method": "document_sentence"
                    }
                    
                    # Extract numeric claims
                    numeric_claims = self._extract_numeric_claims(fact["content"])
                    if numeric_claims:
                        fact["numeric_claims"] = numeric_claims
                    
                    facts.append(fact)
        return facts
    
    def _extract_numeric_claims(self, text: str) -> List[Dict]:
        """Extract numeric claims from text (dates, percentages, amounts)."""
        numeric_claims = []
        patterns = [
            (r'(\d+(?:\.\d+)?)%', 'percentage'),
            (r'\b(19|20)\d{2}\b', 'year'),
            (r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', 'currency'),
            (r'(\d+(?:\.\d+)?)\s*(GB|TB|GHz|MHz|TFLOPS|cores|threads)', 'unit'),
            (r'\b(\d{2,})\b', 'number'),
        ]
        
        for pattern, claim_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                numeric_claims.append({
                    "value": match.group(1),
                    "type": claim_type,
                    "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                })
        return numeric_claims
    
    def _group_similar_facts(self, facts: List[Dict]) -> Dict[str, List[Dict]]:
        """Group similar facts together for comparison."""
        groups = defaultdict(list)
        
        for fact in facts:
            group_key = self._create_fact_group_key(fact["content"])
            
            # Add source information to fact
            fact_with_source = fact.copy()
            fact_with_source["source_info"] = {
                "type": fact["source_type"],
                "credibility": fact["credibility"],
                "url": fact.get("source_url", "")
            }
            groups[group_key].append(fact_with_source)
        
        filtered_groups = {}
        for key, group_facts in groups.items():
            if len(group_facts) >= 2:
                filtered_groups[key] = group_facts
            elif any(f["credibility"] > 0.8 for f in group_facts):
                filtered_groups[key] = group_facts
        return filtered_groups
    
    def _create_fact_group_key(self, content: str) -> str:
        """Create a key to group similar facts."""
        content_lower = content.lower()
        key_terms = [
            "nvidia", "amd", "intel", "google", "microsoft", "apple",
            "h200", "mi300x", "tpu", "gpu", "cpu", "ai", "performance",
            "price", "cost", "speed", "memory", "bandwidth", "tfops"
        ]
        found_terms = [term for term in key_terms if term in content_lower]
        if found_terms:
            return "_".join(sorted(found_terms)[:3])
        words = content_lower.split()[:4]
        return "_".join(words)
    
    def _verify_fact_group(self, facts: List[Dict]) -> Dict[str, Any]:
        """Verify a group of similar facts."""
        if len(facts) < 2:
            return {
                "status": "single_source",
                "confidence": 0.3,
                "has_contradiction": False,
                "representative_fact": facts[0]["content"] if facts else "",
                "all_sources": [f["source_info"] for f in facts],
                "contradiction_details": None,
                "sources_involved": []
            }
        
        numeric_contradictions = self._check_numeric_contradictions(facts)
        semantic_contradictions = self._check_semantic_contradictions(facts)
        has_contradiction = numeric_contradictions["has_contradiction"] or semantic_contradictions["has_contradiction"]
        confidence = self._calculate_fact_confidence(facts, has_contradiction)
        
        status = "contradiction_found" if has_contradiction else ("verified" if confidence > 0.7 else "partially_verified")
        most_credible_fact = max(facts, key=lambda x: x["credibility"])
        
        return {
            "status": status,
            "confidence": confidence,
            "has_contradiction": has_contradiction,
            "representative_fact": most_credible_fact["content"],
            "all_sources": [f["source_info"] for f in facts],
            "contradiction_details": {
                "numeric": numeric_contradictions if numeric_contradictions["has_contradiction"] else None,
                "semantic": semantic_contradictions if semantic_contradictions["has_contradiction"] else None
            },
            "sources_involved": list(set(f["source_info"]["type"] for f in facts))
        }
    
    def _check_numeric_contradictions(self, facts: List[Dict]) -> Dict[str, Any]:
        """Check for contradictions in numeric claims."""
        contradictions = []
        all_numeric_claims = []
        for fact in facts:
            if "numeric_claims" in fact:
                for claim in fact["numeric_claims"]:
                    all_numeric_claims.append({
                        **claim,
                        "source_credibility": fact["credibility"],
                        "source_type": fact["source_type"]
                    })
        
        if not all_numeric_claims:
            return {"has_contradiction": False, "details": []}
        
        claims_by_type = defaultdict(list)
        for claim in all_numeric_claims:
            key = f"{claim['type']}_{claim['context'][:30]}"
            claims_by_type[key].append(claim)
        
        for claim_group in claims_by_type.values():
            if len(claim_group) < 2: continue
            try:
                numeric_values = []
                for claim in claim_group:
                    value_str = str(claim["value"]).replace(',', '').replace('%', '')
                    try:
                        numeric_value = float(value_str)
                        numeric_values.append({
                            "value": numeric_value,
                            "source": claim["source_type"],
                            "credibility": claim["source_credibility"]
                        })
                    except ValueError: continue
                
                if len(numeric_values) < 2: continue
                
                values = [v["value"] for v in numeric_values]
                min_val, max_val = min(values), max(values)
                avg_val = sum(values) / len(values)
                
                if avg_val != 0:
                    range_ratio = (max_val - min_val) / avg_val
                    if range_ratio > 0.5:
                        contradictions.append({
                            "type": "numeric",
                            "values": numeric_values,
                            "range_ratio": range_ratio,
                            "description": f"Values range from {min_val} to {max_val} ({range_ratio:.1%} difference)"
                        })
            except: continue
        
        return {"has_contradiction": len(contradictions) > 0, "details": contradictions}
    
    def _check_semantic_contradictions(self, facts: List[Dict]) -> Dict[str, Any]:
        """Check for semantic contradictions (opposite meanings)."""
        contradictions = []
        contradiction_pairs = [
            ("increase", "decrease"), ("high", "low"), ("fast", "slow"),
            ("expensive", "cheap"), ("better", "worse"), ("support", "oppose"),
            ("agree", "disagree"), ("true", "false"), ("available", "unavailable")
        ]
        
        fact_contents = [f["content"].lower() for f in facts]
        for word1, word2 in contradiction_pairs:
            has_word1 = any(word1 in content for content in fact_contents)
            has_word2 = any(word2 in content for content in fact_contents)
            
            if has_word1 and has_word2:
                contradictions.append({
                    "type": "semantic",
                    "contradictory_terms": (word1, word2),
                    "description": f"Found '{word1}' and '{word2}' in different sources"
                })
        
        return {"has_contradiction": len(contradictions) > 0, "details": contradictions}
    
    def _calculate_fact_confidence(self, facts: List[Dict], has_contradiction: bool) -> float:
        """Calculate confidence score for a fact group."""
        if not facts: return 0.0
        
        credibility_scores = [f["credibility"] for f in facts]
        avg_credibility = sum(credibility_scores) / len(credibility_scores)
        source_bonus = min(0.3, len(facts) * 0.1)
        contradiction_penalty = 0.4 if has_contradiction else 0.0
        
        confidence = avg_credibility + source_bonus - contradiction_penalty
        return max(0.0, min(1.0, confidence))
    
    def _calculate_overall_confidence(self, verification_results: List[Dict]) -> float:
        """Calculate overall verification confidence."""
        if not verification_results: return 0.5
        
        confidences = [r.get("confidence", 0.5) for r in verification_results]
        weights = [r.get("source_count", 1) for r in verification_results]
        
        total_weight = sum(weights)
        if total_weight == 0: return sum(confidences) / len(confidences)
        return sum(c * w for c, w in zip(confidences, weights)) / total_weight
    
    def get_verification_summary(self, state: AgentState) -> Dict[str, Any]:
        """Get a summary of verification results."""
        if "verified_facts" not in state:
            return {"status": "not_verified"}
        
        verified_facts = state.get("verified_facts", [])
        contradiction_report = state.get("contradiction_report", [])
        verified_count = len([f for f in verified_facts if f.get("confidence", 0) > 0.7])
        
        return {
            "status": "verified" if verified_facts else "no_facts",
            "total_facts_checked": len(verified_facts),
            "verified_facts": verified_count,
            "contradictions_found": len(contradiction_report),
            "overall_confidence": state.get("verification_confidence", 0.5),
            "has_issues": len(contradiction_report) > 0
        }


# Test function
async def test_fact_verification_agent():
    """Test the FactVerificationAgent with contradictory information."""
    print("🧪 Testing FactVerificationAgent...")
    
    # Initialize agent
    agent = FactVerificationAgent()
    
    # Create test state with contradictory information
    state = {
        "query": "Compare NVIDIA H200 and AMD MI300X",
        "research_results": {
            "findings": [
                {
                    "content": "NVIDIA H200 delivers 1,979 TFLOPS FP8 performance.",
                    "source_type": "technical_spec",
                    "source_url": "https://nvidia.com/h200",
                    "credibility": 0.95,
                    "recency": "2024",
                    "key_metrics": ["1,979 TFLOPS"]
                },
                {
                    "content": "AMD claims MI300X offers 2,100 TFLOPS, higher than H200.",
                    "source_type": "company_announcement",
                    "source_url": "https://amd.com/mi300x",
                    "credibility": 0.85,
                    "recency": "2024",
                    "key_metrics": ["2,100 TFLOPS"]
                },
                {
                    "content": "Independent tests show H200 at 1,950 TFLOPS and MI300X at 1,800 TFLOPS.",
                    "source_type": "independent_test",
                    "source_url": "https://tests.com/ai-chips",
                    "credibility": 0.90,
                    "recency": "2024",
                    "key_metrics": ["1,950 TFLOPS", "1,800 TFLOPS"]
                }
            ]
        },
        "retrieved_documents": [
            type('Document', (), {
                'page_content': 'H200 is priced at $98.32 per hour on AWS. MI300X costs $89.75 on Azure.',
                'metadata': {'source_type': 'cloud_pricing', 'date': '2024-01-15'}
            })(),
            type('Document', (), {
                'page_content': 'Some reports suggest H200 pricing starts at $85 per hour for bulk purchases.',
                'metadata': {'source_type': 'industry_report', 'date': '2024-01-10'}
            })()
        ],
        "errors": [],
        "agent_timestamps": {},
        "execution_path": [],
        # Explicitly initialize to None
        "verified_facts": None,
        "contradiction_report": None,
        "verification_confidence": None
    }
    
    # Execute agent
    result = await agent.execute(state)
    
    # Print results - Safe Access for None values
    confidence = result.get('verification_confidence')
    print(f"\n📊 Verification Results:")
    print(f"   Overall confidence: {(confidence if confidence is not None else 0.0):.2f}")
    
    if result.get("verified_facts"):
        verified_facts = result["verified_facts"]
        print(f"   Facts checked: {len(verified_facts)}")
        
        for i, fact in enumerate(verified_facts[:3]):
            print(f"\n   Fact {i+1}:")
            print(f"     Content: {fact['content'][:80]}...")
            print(f"     Status: {fact['verification_status']}")
            print(f"     Confidence: {fact['confidence']:.2f}")
            print(f"     Sources: {len(fact['sources'])}")
            if fact.get('contradictions_found'):
                print(f"     ⚠️  Contradictions detected")
    
    if result.get("contradiction_report"):
        contradictions = result["contradiction_report"]
        print(f"\n⚠️  Contradictions Found ({len(contradictions)}):")
        
        for i, contradiction in enumerate(contradictions[:2]):
            print(f"\n   Contradiction {i+1}:")
            print(f"     Fact: {contradiction['content'][:80]}...")
            if contradiction.get('contradiction_details'):
                details = contradiction['contradiction_details']
                if details.get('numeric'):
                    print(f"     Type: Numeric contradiction")
                    for detail in details['numeric']['details']:
                        print(f"       {detail['description']}")
    
    # Get summary
    summary = agent.get_verification_summary(result)
    print(f"\n📈 Verification Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_fact_verification_agent())