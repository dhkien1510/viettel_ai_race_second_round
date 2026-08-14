"""Span-producing backends. Each implements SpanBackend.predict(text) -> [Entity].

  RuleBackend    — dictionary/regex extractors (always-on base)
  EncoderBackend — token-classification NER (group A/B; group B adds word-seg)
  LLMBackend     — generative Qwen, zero-shot -> JSON -> relocate (group C)
"""
