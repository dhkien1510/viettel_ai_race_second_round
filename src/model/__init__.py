"""NER model scaffolding (PhoBERT / ViHealthBERT token classification).

This is a RECALL BOOSTER for symptoms/diagnoses, to complement the dictionary
extractors and reduce overfitting to the public vocabulary. It requires
torch/transformers (see requirements.txt) and is NOT needed to run the
rule-based pipeline. Imports are guarded so `import src.model` never breaks the
rule pipeline.
"""
