"""Entity linking (ICD-10 for CHẨN_ĐOÁN, RxNorm for THUỐC).

Scaffolding: the interfaces and dictionary fast-path are real, but the pipeline
currently runs with EMIT_CANDIDATES=False (see infer.py), so candidates are
left empty until the full ICD-10-CM / RxNorm databases + embedding retrieval
are wired in.
"""
