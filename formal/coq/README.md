# Machine-checked provenance-evidence theorem

`ProvenanceEvidence.v` mechanizes the theorem stated in the paper:

- each support represented by a provenance circuit is sufficient for the
  circuit verdict when its signed facts hold;
- the bottom-up necessary-evidence rules compute exactly the facts occurring
  in every represented support;
- the bottom-up possible-evidence rules compute exactly the facts occurring
  in at least one represented support; and
- necessary evidence is therefore a subset of possible evidence.

The production certificate is a finite hash-consed DAG with n-ary `AND` and
`OR` nodes. The formal model uses the standard binary unfolding: sharing does
not change the represented support family, and each finite n-ary node can be
folded into binary nodes. The proof is intentionally relative to derivations
represented by that circuit. It does not claim completeness over proofs
obtained by arbitrary rewrites of the temporal formula, and it is not an
extraction or formal verification of the Python implementation.

The retained verification artifact at
`evidence/provenance_evidence_coq_verification.json` records the compiler
version, source digest, command, and successful result. To rerun with Coq 8.13
or later:

```text
coqc -q ProvenanceEvidence.v
```
