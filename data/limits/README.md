# Structured performance / injury / leakage limits per regulation.
# Seed + spot-check:
#   python -m ingestion.extract_limits --seed-known --spot-check
# LLM extract after ingest (optional):
#   python -m ingestion.extract_limits --regulation-id UN-ECE-R95
#   EXTRACT_LIMITS_ON_INGEST=1 python -m ingestion.run ...
#
# Spot-check known clauses before trusting LLM merges:
#   HPC = 1000, ThCC = 42 mm, fuel leakage = 30 g/min
