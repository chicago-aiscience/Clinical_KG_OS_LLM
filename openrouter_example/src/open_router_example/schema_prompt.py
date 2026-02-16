"""
Unified Schema Definition for Clinical Knowledge Graph Extraction
All architectures use this same schema for fair comparison.
"""

# === Unified Node Types ===
NODE_TYPES = """## NODE TYPES:
- SYMPTOM: Patient-reported or observed symptoms (chest pain, shortness of breath, nausea, dizziness)
- DIAGNOSIS: Active or suspected conditions being worked up in this encounter (COPD exacerbation, COVID-19, asthma exacerbation, pneumonia)
- TREATMENT: Medications, therapies, lifestyle interventions (Aspirin, Metformin, DASH diet, cardiac rehab)
- PROCEDURE: Tests, exams, surgeries (ECG, stress test, CT angiography, blood test)
- LOCATION: Body parts and anatomical locations (chest, left arm, heart, aortic valve)
- MEDICAL_HISTORY: Pre-existing chronic conditions, past conditions, risk factors, family history (diabetes, hypertension, smoking, family heart disease)
- LAB_RESULT: Quantitative lab values and vital signs (A1C 7.2%, BP 148/90, BNP elevated, LDL 165)"""

# === Unified Edge Types ===
EDGE_TYPES = """## EDGE TYPES:
- CAUSES: Risk factor causes/contributes to condition (smoking CAUSES heart disease)
- INDICATES: Symptom indicates possible diagnosis (chest pain INDICATES angina)
- LOCATED_AT: Symptom or condition at body location (pain LOCATED_AT chest)
- RULES_OUT: Test/procedure rules out condition (ECG RULES_OUT arrhythmia)
- TAKEN_FOR: Treatment used for condition (Aspirin TAKEN_FOR angina)
- CONFIRMS: Lab result or test confirms diagnosis (elevated BNP CONFIRMS heart failure)"""

# === Unified Output Format ===
OUTPUT_FORMAT = """## OUTPUT FORMAT:
Return a JSON object with:
- nodes: array of nodes, each with:
  - id: sequential ID (N_001, N_002, ...)
  - text: entity text (under 30 characters)
  - type: one of the NODE TYPES above
  - evidence: quote from transcript (under 50 characters)
  - turn_id: speaker turn label (e.g. "D-3" or "P-5")
- edges: array of edges, each with:
  - source_id: source node ID
  - target_id: target node ID
  - type: one of the EDGE TYPES above
  - evidence: quote from transcript (under 50 characters)
  - turn_id: speaker turn label (e.g. "D-3" or "P-5")

Output ONLY valid JSON, no other text."""

# === Combined Base Prompt (for one-shot / initial extraction) ===
BASE_EXTRACTION_PROMPT = f"""You are a Clinical Knowledge Graph Extractor. Extract medical entities and relationships from the transcript.

{NODE_TYPES}

{EDGE_TYPES}

## TRANSCRIPT:
{{transcript}}

{OUTPUT_FORMAT}"""

# === Version for reference ===
SCHEMA_VERSION = "2.0"
