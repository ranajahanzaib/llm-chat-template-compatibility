# Experiment design

LLM Chat Template Compatibility separates rendered serialization, tokenizer output, and generation behavior. Native comparison captures each configured renderer and token sequence. A shared-input experiment should only submit one verified token sequence to both engines. Crossover experiments hold each runtime's model representation fixed while changing prompt source, where tokenizer semantics permit.

A rendered string does not prove which internal prompt a higher-level chat API ultimately executes. The controlled completion route observes the token IDs the tool submits directly. Shared token IDs with different outputs indicate remaining inference-path differences; they do not identify one cause. Prefer within-runtime comparisons before attributing behavior to templates.
