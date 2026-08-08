# Architecture

Standing constraints for Passive Safety RAG. Prefer extending existing paths
over introducing parallel gateways, caches, or eval/production wiring.

## Known historical risks

- FreeLLMAPI (localhost:3001) is an optional local overflow proxy. Eval scoring
  must use the **pinned** judge config ``config/portkey/eval_judge_pinned.json``
  (single model, no production fallback chain) so RAGAS/DeepEval scores stay
  comparable run-over-run. FreeLLMAPI must NEVER be referenced from generation/,
  retrieval/, or app/ — those use the production Portkey configs only. If a future
  task needs more production LLM headroom, extend the existing production Portkey
  configs directly; do not repurpose eval judge wiring for production traffic.
