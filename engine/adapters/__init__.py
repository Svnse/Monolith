"""Producer adapters for the Turn Pipeline.

Each adapter satisfies monokernel.turn_pipeline.ProducerAdapter and wraps a
concrete producer (the local llama.cpp LLMEngine, the remote CONNECT
bridge). Adapters depend on the pipeline — the pipeline does NOT depend on
engine/* code. This file's existence is what gives the pipeline its
source-agnostic property at the import-graph level.
"""
