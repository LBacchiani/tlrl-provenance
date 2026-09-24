(** Machine-checked proof of the evidence theorem in the paper.

    The production representation is a finite, hash-consed DAG with n-ary
    AND/OR nodes.  Unfolding sharing and folding each n-ary node into binary
    nodes gives the inductive circuit below without changing its represented
    support family.  The theorem is deliberately relative to that family: it
    does not claim completeness under arbitrary logical rewrites. *)

From Coq Require Import List Bool Classical_Prop.
Import ListNotations.

Section ProvenanceEvidence.

Context {Fact : Type}.

Inductive circuit : Type :=
| COne : circuit
| CFact : Fact -> circuit
| CAnd : circuit -> circuit -> circuit
| COr : circuit -> circuit -> circuit.

(** [Support c s] means that [s] is one derivation represented by [c].
    Lists implement finite fact sets here; duplicate facts do not affect any
    membership statement proved below. *)
Inductive Support : circuit -> list Fact -> Prop :=
| Support_one : Support COne []
| Support_fact : forall f, Support (CFact f) [f]
| Support_and : forall p q sp sq,
    Support p sp ->
    Support q sq ->
    Support (CAnd p q) (sp ++ sq)
| Support_or_left : forall p q s,
    Support p s ->
    Support (COr p q) s
| Support_or_right : forall p q s,
    Support q s ->
    Support (COr p q) s.

(** Closed-form bottom-up characterizations used by the implementation.
    At AND, a fact is necessary if it is necessary to at least one child;
    at OR, it must be necessary to both children.  Possible evidence uses
    union at both node kinds. *)
Fixpoint Necessary (c : circuit) (f : Fact) : Prop :=
  match c with
  | COne => False
  | CFact g => f = g
  | CAnd p q => Necessary p f \/ Necessary q f
  | COr p q => Necessary p f /\ Necessary q f
  end.

Fixpoint Possible (c : circuit) (f : Fact) : Prop :=
  match c with
  | COne => False
  | CFact g => f = g
  | CAnd p q => Possible p f \/ Possible q f
  | COr p q => Possible p f \/ Possible q f
  end.

Lemma support_exists :
  forall c, exists s, Support c s.
Proof.
  induction c as [|f|p IHp q IHq|p IHp q IHq].
  - exists []; constructor.
  - exists [f]; constructor.
  - destruct IHp as [sp Hsp].
    destruct IHq as [sq Hsq].
    exists (sp ++ sq). now constructor.
  - destruct IHp as [sp Hsp].
    exists sp. now apply Support_or_left.
Qed.

Lemma support_one_inv :
  forall s, Support COne s -> s = [].
Proof.
  intros s H. inversion H. reflexivity.
Qed.

Lemma support_fact_inv :
  forall g s, Support (CFact g) s -> s = [g].
Proof.
  intros g s H. inversion H. reflexivity.
Qed.

Lemma support_and_inv :
  forall p q s,
    Support (CAnd p q) s ->
    exists sp sq,
      Support p sp /\ Support q sq /\ s = sp ++ sq.
Proof.
  intros p q s H. inversion H; subst.
  eauto.
Qed.

Lemma support_or_inv :
  forall p q s,
    Support (COr p q) s -> Support p s \/ Support q s.
Proof.
  intros p q s H. inversion H; subst; auto.
Qed.

(** The recursive necessary-evidence rule computes exactly the intersection
    of every represented support. *)
Theorem necessary_exact :
  forall c f,
    Necessary c f <-> forall s, Support c s -> In f s.
Proof.
  induction c as [|g|p IHp q IHq|p IHp q IHq]; intros f; simpl.
  - split.
    + contradiction.
    + intro Hall.
      specialize (Hall [] Support_one). contradiction.
  - split.
    + intros Heq s Hs. subst f.
      rewrite (support_fact_inv g s Hs). simpl. auto.
    + intro Hall.
      specialize (Hall [g] (Support_fact g)).
      simpl in Hall. destruct Hall as [Hall | Hall]; [symmetry; assumption | contradiction].
  - split.
    + intros Hnecessary s Hs.
      destruct (support_and_inv p q s Hs)
        as [sp [sq [Hsp [Hsq ->]]]].
      destruct Hnecessary as [Hp | Hq].
      * apply in_or_app. left.
        apply (proj1 (IHp f) Hp sp Hsp).
      * apply in_or_app. right.
        apply (proj1 (IHq f) Hq sq Hsq).
    + intro Hall.
      destruct (classic (Necessary p f)) as [Hp | Hnp]; [now left |].
      destruct (classic (Necessary q f)) as [Hq | Hnq]; [now right |].
      exfalso.
      assert (Hpmiss : exists sp, Support p sp /\ ~ In f sp).
      { apply NNPP. intro Hnone.
        apply Hnp. apply (proj2 (IHp f)).
        intros sp Hsp. apply NNPP. intro Hnotin.
        apply Hnone. exists sp. auto. }
      assert (Hqmiss : exists sq, Support q sq /\ ~ In f sq).
      { apply NNPP. intro Hnone.
        apply Hnq. apply (proj2 (IHq f)).
        intros sq Hsq. apply NNPP. intro Hnotin.
        apply Hnone. exists sq. auto. }
      destruct Hpmiss as [sp [Hsp Hnsp]].
      destruct Hqmiss as [sq [Hsq Hnsq]].
      specialize (Hall (sp ++ sq) (Support_and p q sp sq Hsp Hsq)).
      apply in_app_or in Hall. destruct Hall; contradiction.
  - split.
    + intros [Hp Hq] s Hs.
      destruct (support_or_inv p q s Hs) as [Hsp | Hsq].
      * apply (proj1 (IHp f) Hp s Hsp).
      * apply (proj1 (IHq f) Hq s Hsq).
    + intro Hall. split.
      * apply (proj2 (IHp f)). intros s Hs.
        apply Hall. now apply Support_or_left.
      * apply (proj2 (IHq f)). intros s Hs.
        apply Hall. now apply Support_or_right.
Qed.

(** The recursive possible-evidence rule computes exactly the union of every
    represented support. *)
Theorem possible_exact :
  forall c f,
    Possible c f <-> exists s, Support c s /\ In f s.
Proof.
  induction c as [|g|p IHp q IHq|p IHp q IHq]; intros f; simpl.
  - split.
    + contradiction.
    + intros [s [Hs Hin]].
      rewrite (support_one_inv s Hs) in Hin. contradiction.
  - split.
    + intro Heq. subst f.
      exists [g]. split; [constructor | simpl; auto].
    + intros [s [Hs Hin]].
      rewrite (support_fact_inv g s Hs) in Hin.
      simpl in Hin. destruct Hin as [Hin | Hin]; [symmetry; assumption | contradiction].
  - split.
    + intros [Hp | Hq].
      * destruct (proj1 (IHp f) Hp) as [sp [Hsp Hin]].
        destruct (support_exists q) as [sq Hsq].
        exists (sp ++ sq). split.
        -- now apply Support_and.
        -- apply in_or_app. now left.
      * destruct (proj1 (IHq f) Hq) as [sq [Hsq Hin]].
        destruct (support_exists p) as [sp Hsp].
        exists (sp ++ sq). split.
        -- now apply Support_and.
        -- apply in_or_app. now right.
    + intros [s [Hs Hin]].
      destruct (support_and_inv p q s Hs)
        as [sp [sq [Hsp [Hsq ->]]]].
      apply in_app_or in Hin. destruct Hin as [Hin | Hin].
      * left. apply (proj2 (IHp f)). exists sp. auto.
      * right. apply (proj2 (IHq f)). exists sq. auto.
  - split.
    + intros [Hp | Hq].
      * destruct (proj1 (IHp f) Hp) as [s [Hs Hin]].
        exists s. split; [now apply Support_or_left | assumption].
      * destruct (proj1 (IHq f) Hq) as [s [Hs Hin]].
        exists s. split; [now apply Support_or_right | assumption].
    + intros [s [Hs Hin]].
      destruct (support_or_inv p q s Hs) as [Hsp | Hsq].
      * left. apply (proj2 (IHp f)). exists s. auto.
      * right. apply (proj2 (IHq f)). exists s. auto.
Qed.

Corollary necessary_is_possible :
  forall c f, Necessary c f -> Possible c f.
Proof.
  intros c f Hnec.
  apply (proj2 (possible_exact c f)).
  destruct (support_exists c) as [s Hs].
  exists s. split; [assumption |].
  apply (proj1 (necessary_exact c f) Hnec s Hs).
Qed.

(** A represented derivation is sufficient for the monotone circuit verdict:
    if every signed fact selected by the derivation holds, the circuit holds.
    In the production certificate, leaves are signed observations that agree
    with the trace and the root circuit explains the already-computed verdict. *)
Variable valuation : Fact -> bool.

Fixpoint denotes (c : circuit) : bool :=
  match c with
  | COne => true
  | CFact f => valuation f
  | CAnd p q => denotes p && denotes q
  | COr p q => denotes p || denotes q
  end.

Definition facts_hold (s : list Fact) : Prop :=
  forall f, In f s -> valuation f = true.

Theorem represented_support_sound :
  forall c s,
    Support c s -> facts_hold s -> denotes c = true.
Proof.
  intros c s Hsupport.
  induction Hsupport; intro Hhold; simpl.
  - reflexivity.
  - apply Hhold. simpl. auto.
  - apply andb_true_iff. split.
    + apply IHHsupport1. intros f Hin.
      apply Hhold. apply in_or_app. now left.
    + apply IHHsupport2. intros f Hin.
      apply Hhold. apply in_or_app. now right.
  - apply orb_true_iff. left.
    apply IHHsupport. exact Hhold.
  - apply orb_true_iff. right.
    apply IHHsupport. exact Hhold.
Qed.

End ProvenanceEvidence.
