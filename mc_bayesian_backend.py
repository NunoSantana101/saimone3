# mc_bayesian_backend.py
"""
Monte Carlo and Bayesian Inference Backend Module
Integrates with sAImona backend stack to provide statistical analysis capabilities.
Designed to be called from assistant.py via function calls and return structured results.
"""

import json
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from scipy.stats import norm
import logging

# Import your existing MC and Bayesian modules
from mc_simulation_bayesian import mc_simulate_bayesian, sample_metric_and_prob, sample_stakeholder_and_prob
from bayesian_inference import compute_posterior, recommend_decision, classify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedAffairsMCBayesianEngine:
    """
    Medical Affairs Monte Carlo and Bayesian Analysis Engine
    Provides statistical analysis capabilities for pharmaceutical strategic planning
    """
    
    def __init__(self):
        self.session_cache = {}
        self.analysis_history = []
        
    def load_pharma_scenarios(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate pharmaceutical-specific scenarios based on context
        """
        therapy_area = context.get("therapy_area", "general")
        region = context.get("region", "US")
        phase = context.get("lifecycle_phase", "launch")
        
        # Base pharmaceutical metrics template
        base_metrics = {
            "regulatory_approval_probability": {
                "mean": 0.75, "stddev": 0.15, "min": 0.3, "max": 0.95,
                "therapy_area_overrides": {
                    "oncology": {"mean": 0.85, "stddev": 0.12},
                    "neurology": {"mean": 0.70, "stddev": 0.18},
                    "cardiology": {"mean": 0.80, "stddev": 0.14}
                }
            },
            "market_access_success": {
                "mean": 0.65, "stddev": 0.20, "min": 0.2, "max": 0.9,
                "phase_overrides": {
                    "pre_launch": {"mean": 0.60, "stddev": 0.25},
                    "launch": {"mean": 0.70, "stddev": 0.18},
                    "post_launch": {"mean": 0.75, "stddev": 0.15}
                }
            },
            "hcp_adoption_rate": {
                "mean": 0.55, "stddev": 0.25, "min": 0.1, "max": 0.85,
                "region_overrides": {
                    "US": {"mean": 0.60, "stddev": 0.22},
                    "EU": {"mean": 0.52, "stddev": 0.28},
                    "APAC": {"mean": 0.48, "stddev": 0.30}
                }
            },
            "payer_coverage_probability": {
                "mean": 0.70, "stddev": 0.18, "min": 0.3, "max": 0.95
            },
            "competitive_pressure": {
                "mean": 0.45, "stddev": 0.20, "min": 0.1, "max": 0.8
            },
            "safety_profile_score": {
                "mean": 0.80, "stddev": 0.12, "min": 0.5, "max": 0.95
            }
        }
        
        # Base stakeholder template
        base_stakeholders = [
            {
                "name": "Key Opinion Leaders",
                "base_weight": 0.85,
                "weight_distribution": {"mean": 0.85, "stddev": 0.10, "min": 0.6, "max": 1.0}
            },
            {
                "name": "Regulatory Authorities",
                "base_weight": 0.90,
                "weight_distribution": {"mean": 0.90, "stddev": 0.08, "min": 0.7, "max": 1.0}
            },
            {
                "name": "Payers",
                "base_weight": 0.80,
                "weight_distribution": {"mean": 0.80, "stddev": 0.15, "min": 0.5, "max": 0.95}
            },
            {
                "name": "Healthcare Providers",
                "base_weight": 0.75,
                "weight_distribution": {"mean": 0.75, "stddev": 0.12, "min": 0.5, "max": 0.9}
            },
            {
                "name": "Patient Advocacy Groups",
                "base_weight": 0.65,
                "weight_distribution": {"mean": 0.65, "stddev": 0.18, "min": 0.3, "max": 0.85}
            }
        ]
        
        # Generate 3 scenarios: Conservative, Balanced, Aggressive
        scenarios = []
        
        # Conservative Scenario
        conservative_metrics = {}
        for metric_name, metric_def in base_metrics.items():
            conservative_metrics[metric_name] = metric_def.copy()
            # Reduce means by 15% for conservative scenario
            if "mean" in conservative_metrics[metric_name]:
                conservative_metrics[metric_name]["mean"] *= 0.85
        
        scenarios.append({
            "name": "Conservative",
            "description": "Risk-averse approach with proven strategies",
            "metrics": conservative_metrics,
            "stakeholders": base_stakeholders
        })
        
        # Balanced Scenario
        scenarios.append({
            "name": "Balanced", 
            "description": "Optimal risk-return profile with mixed innovation",
            "metrics": base_metrics,
            "stakeholders": base_stakeholders
        })
        
        # Aggressive Scenario
        aggressive_metrics = {}
        for metric_name, metric_def in base_metrics.items():
            aggressive_metrics[metric_name] = metric_def.copy()
            # Increase means by 20% but also increase uncertainty
            if "mean" in aggressive_metrics[metric_name]:
                aggressive_metrics[metric_name]["mean"] *= 1.20
                aggressive_metrics[metric_name]["stddev"] *= 1.15
        
        scenarios.append({
            "name": "Aggressive",
            "description": "High-innovation approach with higher risk/reward",
            "metrics": aggressive_metrics,
            "stakeholders": base_stakeholders
        })
        
        return scenarios
    
    def run_mc_simulation(
        self,
        query_context: Dict[str, Any],
        n_iterations: int = 200,
        custom_scenarios: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for pharmaceutical strategic analysis
        """
        try:
            start_time = time.time()
            
            # Extract context
            therapy_area = query_context.get("therapy_area", "general")
            region = query_context.get("region", "US")
            phase = query_context.get("lifecycle_phase", "launch")
            
            # Create context for simulation
            context = {
                "therapy_area": therapy_area,
                "region": region,
                "lifecycle_phase": phase,
                "modifiers": {
                    "therapy_area_modifier": {
                        "oncology": 1.15,
                        "neurology": 0.95,
                        "cardiology": 1.05,
                        "general": 1.0
                    },
                    "region_modifier": {
                        "US": 1.1,
                        "EU": 1.0,
                        "APAC": 0.9
                    },
                    "regulatory_complexity_modifier": {
                        "high": 0.85,
                        "medium": 1.0,
                        "low": 1.15
                    },
                    "lifecycle_stage_modifier": {
                        "pre_launch": 0.9,
                        "launch": 1.0,
                        "post_launch": 1.1
                    }
                }
            }
            
            # Use custom scenarios or generate pharmaceutical-specific ones
            scenarios = custom_scenarios or self.load_pharma_scenarios(context)
            
            # Validate scenarios
            if not scenarios or len(scenarios) < 2:
                raise ValueError("At least 2 scenarios required for meaningful analysis")
            
            # Limit iterations for performance
            n_iterations = min(n_iterations, 5000)
            
            logger.info(f"Running MC simulation with {len(scenarios)} scenarios, {n_iterations} iterations")
            
            # Run the simulation using your existing code
            results = mc_simulate_bayesian(scenarios, n_iterations, context)
            
            # Post-process results for pharmaceutical context
            execution_time = time.time() - start_time
            
            # Add pharmaceutical-specific insights
            pharma_insights = self._generate_pharma_insights(results, context)
            
            # Prepare final output
            output = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": round(execution_time, 2),
                "simulation_parameters": {
                    "scenarios_analyzed": len(scenarios),
                    "iterations": n_iterations,
                    "therapy_area": therapy_area,
                    "region": region,
                    "lifecycle_phase": phase
                },
                "results": results,
                "pharmaceutical_insights": pharma_insights,
                "recommendations": self._generate_recommendations(results, context)
            }
            
            # Cache results
            session_id = query_context.get("session_id", "default")
            self.session_cache[session_id] = output
            self.analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "analysis_type": "monte_carlo",
                "parameters": context
            })
            
            return output
            
        except Exception as e:
            logger.error(f"MC simulation failed: {str(e)}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
                "error_type": type(e).__name__
            }
    
    def run_bayesian_analysis(
        self,
        query_context: Dict[str, Any],
        evidence: Dict[str, Any],
        prior_beliefs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run Bayesian inference analysis for pharmaceutical decision making
        """
        try:
            start_time = time.time()
            
            # Default pharmaceutical priors if none provided
            if not prior_beliefs:
                prior_beliefs = {
                    "Regulatory_Approval": 0.4,
                    "Market_Access_Success": 0.3, 
                    "Commercial_Failure": 0.3
                }
            
            # Pharmaceutical-specific likelihood models
            likelihood_models = {
                "Regulatory_Approval": {
                    "clinical_efficacy": {"high": 0.9, "medium": 0.6, "low": 0.2},
                    "safety_profile": {"excellent": 0.95, "good": 0.7, "concerning": 0.3},
                    "unmet_need": {"critical": 0.8, "moderate": 0.6, "low": 0.4}
                },
                "Market_Access_Success": {
                    "cost_effectiveness": {"superior": 0.85, "comparable": 0.6, "inferior": 0.3},
                    "payer_relationships": {"strong": 0.8, "moderate": 0.5, "weak": 0.2},
                    "competitive_landscape": {"favorable": 0.7, "neutral": 0.5, "challenging": 0.3}
                },
                "Commercial_Failure": {
                    "market_saturation": {"low": 0.2, "medium": 0.5, "high": 0.8},
                    "regulatory_risk": {"low": 0.1, "medium": 0.4, "high": 0.7},
                    "competitive_threat": {"low": 0.2, "medium": 0.5, "high": 0.8}
                }
            }
            
            # Prepare input for classification
            classification_input = {
                "prior": prior_beliefs,
                "likelihoods": likelihood_models,
                "evidence": evidence
            }
            
            # Run Bayesian classification
            result = classify(classification_input)
            
            execution_time = time.time() - start_time
            
            # Generate pharmaceutical-specific recommendations
            pharma_recommendations = self._generate_bayesian_recommendations(
                result, evidence, query_context
            )
            
            output = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": round(execution_time, 2),
                "analysis_parameters": {
                    "prior_beliefs": prior_beliefs,
                    "evidence_factors": list(evidence.keys()),
                    "therapy_area": query_context.get("therapy_area", "general")
                },
                "bayesian_results": result,
                "pharmaceutical_recommendations": pharma_recommendations,
                "confidence_metrics": self._calculate_confidence_metrics(result)
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Bayesian analysis failed: {str(e)}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
                "error_type": type(e).__name__
            }
    
    def _generate_pharma_insights(self, results: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate pharmaceutical-specific insights from MC results"""
        insights = []
        
        summary = results.get("summary", {})
        recommendation = results.get("recommendation", "")
        
        # Analyze scenario probabilities
        scenario_probs = {name: data.get("mean_posterior", 0) 
                         for name, data in summary.items()}
        
        best_scenario = max(scenario_probs, key=scenario_probs.get)
        best_prob = scenario_probs[best_scenario]
        
        insights.append(f"Recommended strategy: {best_scenario} approach with {best_prob:.1%} confidence")
        
        # Risk assessment
        if best_prob < 0.6:
            insights.append("⚠️ High uncertainty detected - consider additional evidence generation")
        
        # Therapy area specific insights
        therapy_area = context.get("therapy_area", "general")
        if therapy_area == "oncology":
            insights.append("🎯 Oncology focus: Prioritize KOL engagement and biomarker strategy")
        elif therapy_area == "neurology":
            insights.append("🧠 Neurology focus: Emphasize patient journey mapping and caregiver education")
        
        # Regional considerations
        region = context.get("region", "US")
        if region == "EU":
            insights.append("🇪🇺 EU market: Consider HTA requirements and value demonstration")
        elif region == "APAC":
            insights.append("🌏 APAC market: Focus on local partnership and regulatory pathway optimization")
        
        return insights
    
    def _generate_recommendations(self, results: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable pharmaceutical recommendations"""
        recommendations = []
        
        recommendation = results.get("recommendation", "")
        
        if "Conservative" in recommendation:
            recommendations.extend([
                {
                    "priority": "HIGH",
                    "action": "Risk Mitigation",
                    "description": "Implement robust pharmacovigilance and safety monitoring protocols",
                    "timeline": "Immediate"
                },
                {
                    "priority": "MEDIUM", 
                    "action": "Evidence Generation",
                    "description": "Initiate additional real-world evidence studies to reduce uncertainty",
                    "timeline": "3-6 months"
                }
            ])
        
        elif "Aggressive" in recommendation:
            recommendations.extend([
                {
                    "priority": "HIGH",
                    "action": "Innovation Investment", 
                    "description": "Accelerate digital health initiatives and precision medicine approaches",
                    "timeline": "Immediate"
                },
                {
                    "priority": "HIGH",
                    "action": "Market Access",
                    "description": "Develop premium pricing strategy with strong value proposition",
                    "timeline": "Pre-launch"
                }
            ])
        
        else:  # Balanced
            recommendations.extend([
                {
                    "priority": "HIGH",
                    "action": "Strategic Optimization",
                    "description": "Balance innovation with proven medical affairs best practices", 
                    "timeline": "Ongoing"
                },
                {
                    "priority": "MEDIUM",
                    "action": "Stakeholder Engagement",
                    "description": "Implement comprehensive omnichannel engagement strategy",
                    "timeline": "Launch preparation"
                }
            ])
        
        return recommendations
    
    def _generate_bayesian_recommendations(
        self, 
        result: Dict[str, Any], 
        evidence: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate Bayesian-specific pharmaceutical recommendations"""
        recommendations = []
        
        posterior = result.get("posterior", {})
        recommendation = result.get("recommendation", "")
        
        # High confidence recommendations
        max_prob = max(posterior.values()) if posterior else 0
        
        if max_prob > 0.8:
            recommendations.append({
                "confidence": "HIGH",
                "action": f"Proceed with {recommendation} strategy",
                "rationale": f"Strong evidence support ({max_prob:.1%} confidence)",
                "next_steps": "Implement decision with standard monitoring"
            })
        elif max_prob > 0.6:
            recommendations.append({
                "confidence": "MEDIUM", 
                "action": f"Conditional proceed with {recommendation}",
                "rationale": f"Moderate evidence support ({max_prob:.1%} confidence)",
                "next_steps": "Gather additional evidence before full commitment"
            })
        else:
            recommendations.append({
                "confidence": "LOW",
                "action": "Evidence generation required",
                "rationale": f"Insufficient evidence ({max_prob:.1%} confidence)",
                "next_steps": "Conduct additional research before decision"
            })
        
        return recommendations
    
    def _calculate_confidence_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence metrics for Bayesian analysis"""
        posterior = result.get("posterior", {})
        
        if not posterior:
            return {"overall_confidence": 0.0, "decision_clarity": 0.0}
        
        # Overall confidence = highest posterior probability
        overall_confidence = max(posterior.values())
        
        # Decision clarity = difference between top two probabilities
        sorted_probs = sorted(posterior.values(), reverse=True)
        decision_clarity = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        
        return {
            "overall_confidence": round(overall_confidence, 3),
            "decision_clarity": round(decision_clarity, 3),
            "entropy": round(-sum(p * np.log(p) for p in posterior.values() if p > 0), 3)
        }


# Initialize the engine
mc_bayesian_engine = MedAffairsMCBayesianEngine()

# Main function to be called from your assistant backend
def run_statistical_analysis(
    analysis_type: str,
    parameters: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Main entry point for statistical analysis from the assistant backend
    
    Args:
        analysis_type: "monte_carlo" or "bayesian_inference"  
        parameters: Analysis parameters and context
        **kwargs: Additional arguments
    
    Returns:
        Structured results dictionary
    """
    try:
        if analysis_type == "monte_carlo":
            return mc_bayesian_engine.run_mc_simulation(
                query_context=parameters,
                n_iterations=parameters.get("iterations", 200),
                custom_scenarios=parameters.get("scenarios")
            )
        
        elif analysis_type == "bayesian_inference":
            return mc_bayesian_engine.run_bayesian_analysis(
                query_context=parameters,
                evidence=parameters.get("evidence", {}),
                prior_beliefs=parameters.get("priors")
            )
        
        else:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": f"Unknown analysis type: {analysis_type}",
                "supported_types": ["monte_carlo", "bayesian_inference"]
            }
            
    except Exception as e:
        logger.error(f"Statistical analysis failed: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error_message": str(e),
            "error_type": type(e).__name__
        }

# Function to be integrated into your assistant.py function call handler
def handle_statistical_analysis_function_call(args: Dict[str, Any]) -> str:
    """
    Handler for statistical analysis function calls from the assistant
    To be integrated into _handle_function_call in assistant.py
    """
    try:
        analysis_type = args.get("analysis_type")
        parameters = args.get("parameters", {})
        
        # Add session context if available
        if "session_id" not in parameters:
            parameters["session_id"] = args.get("session_id", "default")
        
        # Run the analysis
        result = run_statistical_analysis(analysis_type, parameters)
        
        # Return as JSON string for the assistant
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error_message": str(e),
            "error_type": type(e).__name__
        }
        return json.dumps(error_result)

# Test function
def test_statistical_engine():
    """Test the statistical analysis engine"""
    print("Testing Monte Carlo Analysis...")
    
    # Test Monte Carlo
    mc_params = {
        "therapy_area": "oncology",
        "region": "US", 
        "lifecycle_phase": "launch",
        "iterations": 100
    }
    
    mc_result = run_statistical_analysis("monte_carlo", mc_params)
    print(f"MC Status: {mc_result['status']}")
    
    # Test Bayesian
    print("\nTesting Bayesian Analysis...")
    
    bayesian_params = {
        "therapy_area": "cardiology",
        "evidence": {
            "clinical_efficacy": "high",
            "safety_profile": "good", 
            "cost_effectiveness": "superior"
        }
    }
    
    bayesian_result = run_statistical_analysis("bayesian_inference", bayesian_params)
    print(f"Bayesian Status: {bayesian_result['status']}")
    
    return mc_result, bayesian_result

if __name__ == "__main__":
    test_statistical_engine()
