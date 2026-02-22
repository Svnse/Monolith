from __future__ import annotations

from engine.llm import GeneratorWorker, LLMEngine


class AgentGeneratorWorker(GeneratorWorker):
    """Dedicated agent-mode generation thread for AGENT addon."""


class AgentLLMEngine(LLMEngine):
    """LLM engine variant that always uses the dedicated AgentGeneratorWorker."""

    def _create_generator_worker(self, messages, temp, top_p, max_tokens):
        return AgentGeneratorWorker(
            self.llm,
            messages,
            temp,
            top_p,
            max_tokens,
            runtime=self._runtime,
            agent_mode=True,
        )
