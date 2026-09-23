# -*- coding: utf-8 -*-
"""Build merged_main.tex from Doba.tex (backbone) + other drafts + real code/data numbers.
Idempotent exact-match splices only (no parallel edits to the same file).

RETIRED 2026-09-22: merged_main.tex is now the authoritative manuscript (58 reviewer-revision
hunks applied on top of this script's output, plus a systematic numerical audit). Re-running
this script would silently overwrite those hunks, so it refuses to run by default. To run it
anyway (e.g. after back-filling the hunks), set env ALLOW_MERGE_OVERWRITE=1.
"""
import io, re, sys, os

if os.environ.get("ALLOW_MERGE_OVERWRITE") != "1":
    raise SystemExit(
        "REFUSING TO RUN: build_merge.py is retired; merged_main.tex is authoritative. "
        "Re-running would overwrite 58 reviewer-revision hunks. "
        "Set ALLOW_MERGE_OVERWRITE=1 to override.")

SRC = r"C:\AI\RL\paper\docs\Doba.tex"
OUT = r"C:\AI\RL\paper\docs\merged_main.tex"

def norm(s):
    """Convert literal backslash-n separators to real newlines, but leave LaTeX
    commands \\newline and \\nu intact."""
    return re.sub(r'\\n(?!e|u)', '\n', s)

with io.open(SRC, encoding="utf-8") as f:
    t = f.read()

def rep(anchor, addition, note=""):
    """Full replace: the addition embeds the anchor as a bridge, so the anchor is
    replaced by the addition (no duplication)."""
    global t
    anchor = norm(anchor); addition = norm(addition)
    if t.count(anchor) != 1:
        raise SystemExit("ANCHOR not unique/found (%d): %r ... %s" % (t.count(anchor), anchor[:60], note))
    t = t.replace(anchor, addition, 1)

def rep_insert(anchor, text, before=True, note=""):
    """True insert: anchor text is preserved, text is prepended (before=False) or
    appended (before=True). Used only when the text does NOT embed the anchor."""
    global t
    anchor = norm(anchor); text = norm(text)
    if t.count(anchor) != 1:
        raise SystemExit("INSERT ANCHOR not unique/found (%d): %r ... %s" % (t.count(anchor), anchor[:60], note))
    t = t.replace(anchor, (anchor + text) if before else (text + anchor), 1)

def rep_replace(anchor, newtext, note=""):
    global t
    anchor = norm(anchor); newtext = norm(newtext)
    if t.count(anchor) != 1:
        raise SystemExit("REPLACE ANCHOR not unique/found (%d): %r ... %s" % (t.count(anchor), anchor[:60], note))
    t = t.replace(anchor, newtext, 1)

# ---------------------------------------------------------------- title
rep(r"\title{\vspace{-1.2em}Auditable Reinforcement Learning at the Rule Level:\\ Shared-Symbolizer Rule Induction, Policy Fusion, and Their Failure Modes\vspace{-0.6em}}",
    r"\title{From Replayable Traces to Provenance-Aware Rule Fusion:\\ Auditable Reinforcement Learning at the Rule Level}\n\author{The Authors}")

# ---------------------------------------------------------------- abstract: add six-predicate framing
rep(r"part\nof the scientific record, and bound the claims accordingly.\n\end{abstract}",
    r"""part
of the scientific record, and bound the claims accordingly.

We resist the temptation to collapse \emph{auditability} into a single score. The word bundles at
least six separately testable predicates---trace integrity, lossless coding, rule coverage,
behavioural agreement, composition with provenance, and value-model reliability---and \emph{passing
one test does not imply passing another}. This paper reports a bounded study that measures each
predicate on its own terms and states every prediction before seeing any result.
\end{abstract}""")

# ---------------------------------------------------------------- Intro 1.2 Auditability is not a single property
rep(r"of the behavior it produced.\n\nThis gap matters most",
    r"""of the behavior it produced.

\subsection{Auditability is not a single property}
\label{sec:intro:props}

The word \emph{auditability} is convenient and dangerous: it suggests a single grade that a system
either earns or does not. Our study instead treats it as six disjoint predicates, each with its own
estimand and its own negative result. Collapsing them into one number would hide the fact that the
same system can pass some and fail others.

\begin{table}[t]
\centering\small
\caption{Auditability disaggregated into six separately measured predicates. The right column is a
boundary each predicate enforces; none of them licenses the others.}
\label{tab:props}
\begin{tabular}{@{}p{0.22\linewidth}p{0.34\linewidth}p{0.34\linewidth}@{}}
\toprule
Predicate & Estimand & What it does \emph{not} establish \\
\midrule
Trace integrity & hash chain + exact replay & causal explanation \\
Lossless coding & exact reconstruction of the trace & policy explanation; storage win \\
Rule coverage & held-out coverage of induced rules & correctness of the policy \\
Behavioural agreement & fresh-state action agreement & a shared policy core \\
Composition & return + decision mix & optimal arbitration \\
Value reliability & held-out value diagnostics & valid GPI composition \\
\bottomrule
\end{tabular}
\end{table}

This decomposition is the spine of the paper. Section~\ref{sec:prelim} states the predicates formally;
Section~\ref{sec:evolution} shows which design intuitions failed each one; and
Section~\ref{sec:experiments} reports a pass/fail verdict per predicate. The central negative result
is that \emph{rule overlap} (a coverage predicate) carries no information about \emph{behavioural
agreement} (a different predicate), so a high score on one must never be read as a high score on the
other.

This gap matters most""")

# ---------------------------------------------------------------- Prelim 2.3 scope + non-claims
rep(r"disagreements---of its sources?\n\end{enumerate}\n\nThroughout, we distinguish three evidentiary layers",
    r"""disagreements---of its sources?
\end{enumerate}

\paragraph{Scope distinction.}
The three questions above are frequently conflated with stronger claims they do not support.
Re-executing recorded transitions verifies the record. It does not recover policy weights over time,
optimizer state, gradients, or the random-number state that produced them, and it cannot establish
that the recorder was honest. We therefore keep four evidentiary layers strictly separate:
\emph{trace fidelity} (the record is what ran), \emph{policy description} (the rule bank is a finite
summary), \emph{behavioural agreement} (two policies act alike on fresh states), \emph{provenance}
(which rule and source produced a decision), and \emph{causal explanation} (why a decision was made).
A result at one layer is never cited as evidence at another.

\subsection{Scope and non-claims}
\label{sec:prelim:nonclaims}

To pre-empt over-reading, we state up front what this study does \emph{not} claim. We do not claim
general or complete auditability of arbitrary RL training; that symbolic overlap entails behavioural
consensus; that induced rules are human-readable explanations of a neural policy; a storage-compression
advantage over generic compression; optimal or mechanism-explained arbitration; validated value-model
composition; or a natural-language semantics for the discrete communication symbols. Every positive
claim below is bounded to two benchmark tasks, eight seeds per task, one shared quantizer per task, and
one software/hardware configuration.

Throughout, we distinguish three evidentiary layers""")

# ---------------------------------------------------------------- Method 3.1: Qwen Def 1 + caveats + ZCode bin-label
rep(r"This design choice is coupled to everything downstream; we\ndiscuss its alternatives in Section~\ref{sec:discussion}.\n\n\subsection{Rule bank and admission}",
    r"""This design choice is coupled to everything downstream; we
discuss its alternatives in Section~\ref{sec:discussion}.

\begin{definition}[State symbolization]
\label{def:symbolizer}
A \emph{state symbolizer} $\varphi$ maps a real state to a tuple of bin indices
\begin{equation}
\varphi(s) \;=\; \big(\,\mathrm{bucketize}_{d}\!\big(s^{(d)};\,\theta_{d}\big)\,\big)_{d=1}^{D},
\label{eq:symbolize:def}
\end{equation}
where each $\theta_{d}$ holds the interior empirical-quantile edges of dimension $d$ over $B=4$
bins. The edges are fitted once from pooled random-policy calibration rollouts and then \emph{frozen},
so that all training seeds and arms within a task share the same symbol alphabet.
\end{definition}

\noindent Three consequences follow and are load-bearing. \emph{(i)} The symbolizer is built from pooled
calibration data; no seed fits its own discretizer. \emph{(ii)} Shared labels are \emph{not} shared
semantics: when each agent fits its own state discretizer, identical bin labels need not denote
identical raw-state regions---the largest observed edge difference under per-seed fitting was
$0.157$ observation units. \emph{(iii)} Symbol overlap is therefore \emph{not} behavioural equivalence.
The fix that makes cross-seed comparison possible is the single frozen symbolizer above.

\subsection{Rule bank and admission}""")

# ---------------------------------------------------------------- Method 3.2: Qwen Def 2 + failure evidence
rep(r"qquad \tau_{s}=8,\ \tau_{c}=0.70 .\n\label{eq:admission}\n\end{equation}\nWe call $f_{\mathbf{c},a}$ the \emph{confidence}",
    r"""qquad \tau_{s}=8,\ \tau_{c}=0.70 .
\label{eq:admission}
\end{equation}

\begin{definition}[Admitted state--action rule]
\label{def:rule}
Let $N(c,a)$ be the count of observed transitions whose symbol is $c=\varphi(s)$ and action is $a$,
and $N(c)=\sum_{a}N(c,a)$. The induced action for condition $c$ is
$a^{\star}(c)=\arg\max_{a}N(c,a)$ (ties to the smallest index), with support
$\kappa_{s}(c)=N(c,a^{\star})$ and confidence
\begin{equation}
\kappa(c) \;=\; \frac{N(c,a^{\star}(c))}{N(c)},
\qquad
\text{rule } c\!\rightarrow\!a^{\star}(c) \text{ is admitted iff }
\kappa_{s}(c)\ge \kappa_{s}^{\min}\ \wedge\ \kappa(c)\ge \tau .
\label{eq:rule:def}
\end{equation}
The admitted rules form a rule bank $\mathcal{R}$. Each rule stores the multiset of source step
identifiers that produced it, and the digest of that multiset.
\end{definition}

\noindent Equation~\eqref{eq:rule:def} uses a single normalization for confidence; support and confidence
are reported separately from coverage, so that a precise-but-rare rule is never described as broadly
applicable. The thresholds are not arbitrary. A high-confidence variant ($\tau_c=0.90$) of the
\emph{online} rule-constraint pilot admitted so few rules that fusion deferred to the fallback on
100\% of held-out decisions, and strict admission produced a 100\% blind-spot collapse
(Section~\ref{sec:evolution:threshold}); at $\tau_c=0.70$ with $\tau_s=8$ and an explicit blind-spot
fallback, coverage failure becomes observable rather than silent. High confidence is therefore neither
safety nor generalizability: rule admission is a \emph{passive} observation step and must never re-enter
the training loop.

We call $f_{\mathbf{c},a}$ the \emph{confidence}""")

# ---------------------------------------------------------------- Method 3.3: three-evidence split + V_prov + hashable core + codec
rep(r"the sequence of transitions.\n\n\subsection{Training arms}",
    r"""the sequence of transitions.

\paragraph{Three distinct evidence types.}
The audit ledger supplies three separable proofs that must not be merged into one ``audit score''.
\emph{(i)~Hash-chain integrity} certifies the record was not edited after the fact. \emph{(ii)~Environment
replay} certifies that each recorded transition can be regenerated by the same environment under the
same reset seed (a transport-level check). \emph{(iii)~Provenance resolution} certifies that a given
decision can be parsed back to the rule, source bank, arbiter outcome, and fallback that produced it.
The provenance predicate is
\begin{equation}
V_{\mathrm{prov}}(t) \;=\; \big[\,\mathrm{ref}_t \in \mathcal{R}_{i(t)} \;\wedge\;
\mathrm{act}(\mathrm{ref}_t)=a_t \;\wedge\; \mathrm{stat}(\mathrm{ref}_t)=\mathrm{stat}_t\,\big],
\label{eq:vprov}
\end{equation}
which an independent verifier evaluates for every decision in the fused trace.

\paragraph{The hashable core.}
A subtle failure mode is a hash that covers non-core fields. In an early schema a sidecar counter and a
wrapper label entered the update hash even though they did not affect optimisation, so two runs with
identical optimizer behaviour hashed differently. The fix is a \emph{normalized core-update hash} that
excludes metadata (the previous hash, the record hash, and the grammar-generation annotation) and
covers only optimisation-relevant tensors---step index, parameter-gradient hash, and pre/post
checkpoint hashes. Equivalent runs then hash identically, and optimizer-equivalent arms (baseline vs.
audit-only) are demonstrably byte-identical.

\subsection{Lossless trajectory codec}
\label{sec:method:codec}

Beyond the rule bank, every trace is also compressed by a \emph{lossless} codec so that a replayed
record can be checked against its compressed form. The codec is a Re-Pair-style pair-substitution
grammar with a bounded rule count ($\le 96$ rules), and decoding is asserted to be exact (the decoded
payload hashes to the input hash). Compression is measured as a two-part code length: the grammar
description plus the encoded sequence, using Elias-gamma counts for terminals and a fixed-width symbol
reference for non-terminals; the provenance bytes are reported separately. The codec is \emph{exact},
not a lossy summary: its role is fidelity, not storage. In practice it does \emph{not} beat generic
gzip on size (Section~\ref{sec:experiments:codec}), which is itself a reported finding.

\subsection{Training arms}""")

# ---------------------------------------------------------------- Method 3.5: decision classes paragraph (fusion figure already in backbone)
rep_insert(r"\begin{figure}[t]\n\centering\n\caption{Rule-level policy fusion.}",
    r"""

\paragraph{Decision classes.}
Every fused decision is classified into one of five classes, and the class is recorded alongside the
action: \emph{agreement} (all candidates agree), \emph{single-source} (only one bank contributes),
\emph{conflict} (candidates disagree on the action), \emph{blind spot} (no candidate; fallback used),
and \emph{fallback} (the residual single-actor action). These classes are not report formatting; they
are the conditions under which the result is interpreted. Reporting a fused return without the
coverage, conflict rate, blind-spot rate, fallback rate, source distribution, and arbitration outcome
is incomplete.
""", before=False)

# ---------------------------------------------------------------- Method 3.5: enhance fusion figure with provenance record + class marks
rep_replace(r"\begin{figure}[t]\n\centering\n\caption{Rule-level policy fusion.}\n\label{alg:fusion}\n\begin{minipage}{0.88\textwidth}\n\small\n\textbf{Input:} rule banks $\mathcal{R}_1,\ldots,\mathcal{R}_n$; symbolizer $\varphi$; fallback actor $\pi_{\mathrm{fb}}$.\n\vspace{0.35em}\n\nFor each decision at state $s$:\n\begin{enumerate}\n  \item $\mathbf{c}\leftarrow \varphi(s)$\n  \item $\mathcal{C}\leftarrow\{\,r\in\bigcup_i\mathcal{R}_i : \mathbf{c}_r=\mathbf{c}\,\}$ \hfill (candidate rules)\n  \item \textbf{if} $\mathcal{C}=\varnothing$ \textbf{then return} $\pi_{\mathrm{fb}}(s)$ \hfill (blind spot: fallback)\n  \item $r^{*}\leftarrow \arg\max_{r\in\mathcal{C}}\ (f_r,\ n_r,\ \mu_r,\ -i)$ \hfill (lexicographic order)\n  \item \textbf{return} $a_{r^{*}}$\n\end{enumerate}\n\end{minipage}\n\end{figure}",
    r"""\begin{figure}[t]
\centering
\caption{Rule-level policy fusion with provenance record.}
\label{alg:fusion}
\begin{minipage}{0.9\textwidth}
\small
\textbf{Input:} rule banks $\mathcal{R}_1,\ldots,\mathcal{R}_n$; symbolizer $\varphi$; fallback actor $\pi_{\mathrm{fb}}$.\newline
For each decision at state $s$:
\begin{enumerate}
  \item $\mathbf{c}\leftarrow \varphi(s)$
  \item $\mathcal{C}\leftarrow\{\,r\in\bigcup_i\mathcal{R}_i : \mathbf{c}_r=\mathbf{c}\,\}$ \hfill (candidate rules)
  \item \textbf{if} $\mathcal{C}=\varnothing$ \textbf{then return} $\pi_{\mathrm{fb}}(s)$ and mark a \emph{blind spot}
  \item \textbf{if} all $r\in\mathcal{C}$ share one action \textbf{then} choose it and mark \emph{agreement} (or \emph{single-source} if $|\{i:r\in\mathcal{R}_i\}|=1$)
  \item \textbf{else} choose $r^{*}\leftarrow \arg\max_{r\in\mathcal{C}}\ (f_r,\ n_r,\ \mu_r,\ -i)$, attach its source bank and mark \emph{conflict}
  \item append $(c,a,\nu)$ to the decision log, update the chain digest, and return $(a,\nu)$
\end{enumerate}
\end{minipage}
\end{figure}""")

# ---------------------------------------------------------------- Method 3.x: causal symbol intervention (method subsection)
rep(r"failure is informative but not evidence about exact GPI.\n\n\subsection{Statistical conventions}",
    r"""failure is informative but not evidence about exact GPI.

\subsection{Matched-support causal symbol intervention}
\label{sec:method:causal}

The communication experiment (Section~\ref{sec:evolution:comm}) is kept methodologically separate from
rule fusion. In the redesigned \emph{private-target} game a sender observes a private label and a
receiver must act on the message alone; reward is given only for correct identification, so ignoring
the message cannot be optimal. We measure the causal effect of a learned symbol with a
\emph{matched-support intervention}: take states where the sender emits a particular message, replace
the message with a shuffled one (a permutation null), and measure the change in the receiver's action
probability. The intervention is matched so that only the message content changes; the estimand is the
causal effect identified by intervention, not by correlation. A staged acceptance gate governs every
causal claim: diversity $\rightarrow$ positive control $\rightarrow$ necessity $\rightarrow$
generalisation $\rightarrow$ causality $\rightarrow$ channel exclusivity, each with a pre-registered
verdict. The result is bounded to this designed task: it is \emph{not} evidence about state--action
rule fusion, and it does \emph{not} generalize to arbitrary neural policies.

\subsection{Statistical conventions}""")

# ---------------------------------------------------------------- Failure 4.1 rename + compression framing
rep(r"\subsection{Why replay is not explanation}\n\label{sec:evolution:replay}",
    r"""\subsection{Compression, replay, and explanation are different claims}
\label{sec:evolution:replay}""")

# REMOVED 2026-09-22: the 121.3 -> 182.3 splice was incorrect. Audit confirmed 121.3 is the
# correct value for the tau_c=0.90 pilot run (grammar_audit_pilot metadata + experiment_summary
# + research log: 121.3 with 0.4% coverage); 182.3 is seed-11's value in the separate tau_c=0.70
# eight-seed study. Conflating the two corrupts the pilot record. (Former anchor: "fell from 500
# (the environment ceiling) to 121.3,".)

# ---------------------------------------------------------------- Failure 4.3 coverage is not correctness
rep(r"an explicit blind-spot fallback (\ref{eq:fusion}) so\nthat coverage failure is observable rather than silent.",
    r"""an explicit blind-spot fallback (\ref{eq:fusion}) so
that coverage failure is observable rather than silent. The episode is a general lesson:
\emph{coverage is not correctness}. Confidence, support, coverage, action agreement, and return are
five different quantities; a high coverage or confidence threshold is a statistical choice, never a
semantic validation of the rule.""")

# ---------------------------------------------------------------- Failure 4.4 end: sidecar quote + 4.5 + 4.6
rep(r"making equivalent runs hash\nidentically.\n\n% ----------------------------------------------------------------------\n\section{Experiments and Results}",
    r"""making equivalent runs hash
identically.

The same lesson appears in the communication task below (Section~\ref{sec:evolution:comm}): a sidecar
field (a grammar-generation annotation) once entered the update hash even though it did not affect
optimisation, and the fix was to hash only a normalized core record. An audit instrument must audit
itself.

\subsection{Arbitration sensitivity and the random-selection anomaly}
\label{sec:evolution:arbiter}

The confidence-ranked arbiter was chosen as the default resolver because it is the natural
generalization of the per-rule confidence statistic. The results (Section~\ref{sec:experiments:acrobot})
show it is the wrong default on Acrobot-v1: uniformly random arbitration among candidate rules beats
every deterministic variant tested, including the deployed one. We report this as an \emph{open
mechanism} rather than a method. Treated as a design event, it demotes the arbiter from a trusted
component to a diagnostic: the fused policy is reported together with its full decision mix, and the
random-selection ablation is part of the result, not a footnote.

\subsection{Communication-task redesign}
\label{sec:evolution:comm}

The causal-symbol evidence rests on a designed coordination game that had to be rebuilt twice. The
original \emph{parity} game (sender sees a digit, receiver acts on the message, reward on parity match)
collapsed to a degenerate equilibrium: both policies settled on the constant action C/C, the unique
joint-reward optimum under both parities, so the channel carried nothing. Under a parity-matching
reward, constant actions are a Nash equilibrium and no optimizer can escape an objective whose global
optimum ignores the channel---the failure was in \emph{task design}, not learning.

A second, abandoned early design (the private-target variant) failed for a different reason: the
source review found the original coordination payoff made the same joint action optimal for both
parities; a blind control masked only the label while the image encoder retained parity information;
a reserved index lay outside the embedding range; the codebook was bypassed by a straight-through
continuous path; and an opponent-prediction branch was degenerately solvable by copying. The fix was a
parity-critical payoff with opposed optima, joint masking of label and image pathways, corrected index
handling, straight-through codebook wiring with perplexity reporting, and the staged gate above.

Even the redesigned game initially appeared to fail: after training, receiver accuracy was $0.500$
(chance) and the estimated causal influence $\Delta P(C)=0.00088$ ($95\%$ CI $[-0.00136,0.00320]$) was
indistinguishable from zero. The cause was \emph{optimization exposure}, not architecture: the run had
performed only $79$ optimizer updates; reducing the batch size from $128$ to $8$ raised the count to
$1{,}250$, and a run with $12{,}500$ updates succeeded decisively. Two analysis methods also produced
impressive but empty numbers: a ``symbol lookup'' analysis was tautological (the action decoder read
the first token directly), and a whole-sequence mutual-information estimate saturated at $1$ bit
because sparse, high-cardinality messages make plug-in MI vacuous. Both were replaced by a per-position
MI against a permutation null. The only surviving causal claim comes from the redesigned task and is
bounded to it.

% ----------------------------------------------------------------------\n\section{Experiments and Results}""")

# ---------------------------------------------------------------- Related Work: relocate before Method + 3-axis framing + cites
rw_open = norm(r"\section{Related Work}")
rw_close = norm(r"\section{Discussion}")
rw_start = t.index(rw_open)
rw_end = t.index(rw_close)
rw_block = t[rw_start:rw_end]
t = t[:rw_start] + t[rw_end:]  # remove RW from its old position
rw_block = rw_block.replace(
    norm(r"\section{Related Work}\n\label{sec:related}"),
    norm(r"""\section{Related Work}
\label{sec:related}

We organize the positioning around three axes, following the most rigorous framing in the literature:
\emph{(A)~assumption}---what each related line assumes about the policy or the environment;
\emph{(B)~demonstrated achievement}---what it actually shows; \emph{(C)~conditions of failure}---where
its guarantee breaks; and \emph{(D)~relationship to the present work}. This structure is adopted
deliberately so that each related method is judged by the same four questions we ask of our own
protocol."""))
rw_block = rw_block.replace(
    norm(r"\emph{Discrete symbolic communication and auditability.}"),
    norm(r"""\emph{Discrete symbolic communication and auditability (axis A--D).} Emergent communication can
appear under cooperative objectives \cite{foerster2016,havrylov2017}, but measurement is fraught:
Lowe et al.\ \cite{lowe2019} catalog pitfalls in which apparent communication is an artifact of the
objective or the analysis. The ``AI Mother Tongue'' line shows VQ-VAE-derived discrete symbols can
serve as an endogenous channel \cite{liu2025aim}, and its audit agenda asks whether coordination logic
can be reconstructed from discrete-symbol logs without hidden channels. The present work shares the
audit agenda but does not rely on emergent communication: our symbols are a fixed calibration-derived
abstraction, not a learned codebook, so the two lines are complementary. Our causal-symbol test
(Section~\ref{sec:experiments:causal}) is a positive demonstration that the causal question \emph{can}
be answered by intervention in a designed task, complementing Lowe et al.'s catalog of ways correlation
misleads.

\emph{Discrete symbolic communication and auditability.}"""))
method_anchor = norm(r"\section{Method}\n\label{sec:method}")
if t.count(method_anchor) != 1:
    raise SystemExit("Method anchor not unique for RW insertion (%d)" % t.count(method_anchor))
t = t.replace(method_anchor, rw_block + "\n\n" + method_anchor, 1)

# ---------------------------------------------------------------- Experiments: Registered predictions (P1-P8 + P9)
rep(r"environment\nreplay were re-verified after the runs completed.\n\n\subsection{Audit integrity}",
    r"""environment
replay were re-verified after the runs completed.

\subsection{Registered predictions}
\label{sec:experiments:pred}

We fix predictions before reporting any result, so that they can be contradicted. They are
\emph{unregistered} (no external preregistration record exists), so they are treated as post hoc when
assessing claim strength. Each receives a status in Section~\ref{sec:experiments}.

\begin{table}[t]
\centering\small
\caption{Predictions fixed before analysis. Layer: operational-health (OH), diagnostic (D),
identification-bearing (ID). Status assigned in Section~\ref{sec:experiments}.}
\label{tab:pred}
\begin{tabular}{@{}llll@{}}
\toprule
ID & Prediction & Layer & Status \\
\midrule
P1 & Hash-chain, ledger and replay checks pass on all recorded runs. & OH & supported \\
P2 & The lossless codec reconstructs every trace prefix exactly. & OH & supported \\
P3 & Baseline and audit-only arms are training-equivalent. & OH & supported \\
P4 & Cross-seed rule overlap is substantially above zero under a shared symbolizer. & D & supported \\
P5 & Two actors with high rule overlap also agree on most fresh held-out states. & ID & contradicted \\
P6 & Fused mean return exceeds the best single source actor on both tasks. & ID & supported (CartPole) / partial (Acrobot) \\
P7 & Blind spots, not conflicts, are the main obstacle to fusion. & ID & contradicted \\
P8 & The deployed confidence-ranked arbiter is at least as good as alternative selectors. & ID & contradicted \\
P9 & A learned rule bank predicts an independent holdout above an action-frequency baseline, with
coverage, agreement, and confidence reported separately. & ID & registered (not executed in present runs) \\
\bottomrule
\end{tabular}
\end{table}

P9 (the action-frequency baseline prediction) is registered explicitly because coverage/agreement
statistics are otherwise undefined relative to a floor: a rule bank that agrees with the actor on
$51\%$ of states is only informative once compared with the majority-action frequency of the same
states. It is left as a registered-but-unexecuted prediction in this merged draft; the behavioural
baseline is deferred to future work rather than back-filled with a number we did not measure.

\subsection{Audit integrity}""")

# ---------------------------------------------------------------- Results: causal symbol intervention (7.5)
rep_insert(r"\subsection{Fitted-Q GPI baseline failure}",
    r"""

\subsection{Causal symbol intervention as a separate sub-study}
\label{sec:experiments:causal}

The communication experiment is reported apart from rule fusion because it answers a different question:
whether a \emph{learned discrete symbol} in a designed task causally influences a receiver's decision,
not whether state--action rule fusion composes policies. On the redesigned private-target game, six
independent runs of $100{,}000$ decisions each ($600{,}000$ total) give receiver accuracy $0.998923$
with the true message, $0.503292$ with a shuffled message, and $0.500$ with no message. The
matched-support intervention---replacing the message while holding the state fixed---changes the
receiver's action probability by $\bar{\Delta}=0.998366$ on average (range $0.996986$--$0.999685$
across runs), with common-support overlap $0.94308$. Per-position total-variation distance between
even/odd token marginals is $0.0967$ at position~0 and $0.9984$ at position~1, confirming the
information-bearing position identified by the per-position MI analysis.

This is a causal estimand, identified by intervention, not a correlation. Its scope is deliberately
narrow: it shows that \emph{in this designed task} the learned discrete symbols causally influence the
receiver's decisions. It does \emph{not} show that the symbols are human-interpretable, compositional,
or ``meaningful'' in any richer sense, and it does not generalize to other RL agents. All causal claims
in this paper rest on the redesigned task and the staged gate of Section~\ref{sec:method:causal}; the
original parity game is reported only as a negative result.
""", before=False)

# ---------------------------------------------------------------- Results: lossless coding (7.7)
rep(r"value-based composition'' on the strength of this baseline alone.\n\n\subsection{Summary of verdicts}",
    r"""value-based composition'' on the strength of this baseline alone.

\subsection{Lossless coding results}
\label{sec:experiments:codec}

The codec of Section~\ref{sec:method:codec} is exact by construction (the decoded payload hashes to the
input hash). On CartPole-v1 it processes $9{,}178{,}724$ tokens across $16$ grammar productions; the raw
trace is $61.3$\,MB, gzip compresses it to $12.3$\,MB, and the grammar--gzip archive is $13.4$\,MB. On
Acrobot-v1 the corresponding figures are $8{,}549{,}496$ tokens, $57.6$\,MB raw, $12.5$\,MB gzip, and
$13.7$\,MB grammar--gzip. The two-part code length therefore \emph{does not} beat generic compression:
the grammar adds structure without removing bytes, and the grammar--gzip archive is $8.4\%$ larger than
gzip on CartPole-v1. The MDL ratio (grammar code length / gzip code length) sits at $0.51$--$0.58$
across rounds. The result is reported honestly as a fidelity tool, not a storage win: the codec earns
its place by enabling exact replay-against-compressed-form checks, not by shrinking the log.

\subsection{Summary of verdicts}""")

# ---------------------------------------------------------------- Discussion: what it does not establish (8.2)
rep(r"the natural next experiments.\n\n\subsection{Theoretical and practical implications}",
    r"""the natural next experiments.

\subsection{What the protocol does \emph{not} establish}
\label{sec:discussion:not}

The positive results are bounded, and the boundaries are the point. Specifically:
\emph{(i)}~rule overlap does not imply behavioural agreement (Section~\ref{sec:experiments:agreement});
\emph{(ii)}~trace replay does not imply causal explanation, and re-executing recorded transitions
recovers the record but not the optimizer state or random-number stream that produced it;
\emph{(iii)}~provenance visibility does not imply recovery of the neural mechanism;
\emph{(iv)}~the fitted-Q baseline failing its own gate does \emph{not} imply that rule fusion is better
than value-based composition---only that the comparator is unqualified here;
\emph{(v)}~the random-arbitration anomaly does \emph{not} imply that a random arbiter is the generally
optimal resolver. A plain actor-logit ensemble also outperforms rule fusion in CartPole ($500.00$ vs
$205.96$) while being simpler; the objection is correct about performance and misses the point about
auditability. The ensemble cannot say \emph{which} knowledge produced a decision, cannot report that two
sources disagreed, and cannot declare a blind spot. What fusion buys over the ensemble is not return but
\emph{inspectability per decision}; what the hash chain buys over a log file is not accuracy but
\emph{checkability}. Whether that tradeoff is worth it depends on the deployment; the claim is only that
the tradeoff exists and can be measured honestly.

\subsection{Theoretical and practical implications}""")

# ---------------------------------------------------------------- Discussion: auditability vector (8.4)
rep(r"because the constrained arm shows it does not.\n\n\subsection{Limitations}",
    r"""because the constrained arm shows it does not.

\paragraph{Auditability as a vector, not a score.}
The six predicates of Section~\ref{sec:intro:props} should be reported as a vector, never averaged into
one number. The reason is empirical: in this study the components point in different directions---trace
integrity passes cleanly while behavioural agreement fails, and composition improves on CartPole but
collapses into conflict on Acrobot. A single aggregate would conceal exactly the heterogeneity that the
failure-driven history was built to expose. We therefore define
\[
\mathrm{Auditability} = \big(\text{trace integrity},\ \text{replay fidelity},\ \text{rule coverage},\
\text{behavioural agreement},\ \text{provenance visibility},\ \text{value-model validity}\big),
\]
and treat any claim that averages them as out of scope.

\subsection{Limitations}""")

# ---------------------------------------------------------------- Steelman augment (ZCode evidence response)
rep(r"the paper meets it.\n\n\subsection{Replicability note}",
    r"""the paper meets it.

\paragraph{Steelman, continued.}
The strongest reviewer objection is that the positive results are compatible with the rules contributing
nothing beyond a state-conditioned action prior, with replay certifying only that the logger can
reproduce its own numbers, and with the causal-intervention result living in a toy game whose
decision-relevant payoff was engineered to make symbols useful. The response is evidence, not framing.
First, the engineering report \emph{is} the claim: the paper promises exactly replayable,
provenance-visible composition under stated conditions and delivers the artifacts and the refutations
that bound it. Second, the random-rule finding \emph{constrains} rather than voids rule content: it is
separated from action noise by a matched control (random-candidate $-187.02$ versus uniform-random
action $-498.71$), which proves the candidate sets carry task-relevant structure even though the tested
deterministic resolver fails to exploit it. Third, the coordination task is disclosed as designed and
decision-relevant by construction; its evidential role is to show that a causal-symbol test can pass
\emph{somewhere} with proper gates, not that symbols are generally causal. If the reviewer insists these
bounds make the paper modest, the authors agree: modesty is the method.

\subsection{Replicability note}""")

# ---------------------------------------------------------------- Replication: HIGH/MEDIUM ratings
rep(r"the random-arbitration replicates are fresh generations under fixed\nstreams.\n",
    r"""the random-arbitration replicates are fresh generations under fixed
streams.

\paragraph{Likely replication failures and their ratings.}
Three are foreseeable. \emph{(i)}~The arbitration result is sensitive to the selector stream: the initial
stream gave $-187.02$ while five later streams ranged to $-206.17$, so a single-stream replication would
report a different magnitude. \emph{(ii)}~The Acrobot result depends on the return floor: with a median of
$-500.00$, a replication with different reset seeds could shift the mean without changing any qualitative
conclusion. \emph{(iii)}~The behavioural-agreement result depends on the symbolizer, which is fitted to
random-policy occupancy; a replication with a different calibration policy could move it.

\emph{Evidenced mitigations.} Sharing one frozen symbolizer across seeds and arms within a task is
\textsc{high}---it removes the largest known confound in cross-seed comparison and is directly evidenced
by the boundary differences of up to $0.157$ observation units under per-seed fitting. Pairing all policy
comparisons on identical reset seeds is \textsc{high}---it is what makes the paired differences
interpretable given the return variance. Reporting the decision mix alongside every fused return is
\textsc{medium}---it prevents the most likely misreading. Repeating the random-selection condition across
five independent streams is \textsc{medium}---it establishes the effect is not a single-stream artifact.
For independent replication, the minimum package is: the frozen symbolizer boundaries for each task, the
admitted rule banks with their support and confidence statistics, the evaluation reset-seed ranges, and
the decision traces with their chain digests.

""")

# ---------------------------------------------------------------- Limitations: non-claims / future work / material gaps
rep(r"We disclose each of these rather than repairing them by assumption.\n",
    r"""We disclose each of these rather than repairing them by assumption.

\paragraph{Representation and task-design limitations.}
The shared quantile symbolizer is itself a confounder: with only four bins per dimension, two policies
can agree on a rule for the symbol while disagreeing on nearly every state inside it, so the rules
describe the \emph{symbol} and behavior lives on \emph{states}. The communication intervention is a
designed task and does not transfer to general RL agents; an early payoff design made communication
unnecessary, a continuous encoder path could bypass the discrete codebook, and the blind receiver
control had to truly block private information. The value baseline's quality gate had to be pre-registered
precisely because it can fail silently.

\paragraph{Future work.}
The unresolved mechanisms dictate the next steps: decompose the random-arbitration advantage by source
rule, action frequency, and state occupancy; test return-weighted rule-bank induction under a
pre-specified normalization and held-out evaluation; use the native sequence symbols as the control
representation; and evaluate behavior on non-saturated tasks and across hardware.

\paragraph{Material gaps (honest disclosure).}
Compact summary audits report the listed checks true, covering $1{,}725{,}655$ CartPole and
$2{,}544{,}513$ Acrobot trace rows, with $1{,}122{,}403$ and $1{,}742{,}570$ replayed transitions. These
are operational-health assertions on summaries; the full multi-gigabyte run tree is not present in the
reviewed package, and text-normalization can break manifest digests (the manifest is CRLF/LF sensitive).
Cross-hardware replay is untested, the mechanism behind random arbitration remains undecomposed, and no
formal unit-test suite accompanies the diagnostic scripts.

""")

# ---------------------------------------------------------------- Reproducibility Statement (before Conclusion)
rep(r"\section{Conclusion}\n\label{sec:conclusion}",
    r"""\section*{Reproducibility Statement}
\label{sec:repro}

All experiments use fixed seeds (control: 11, 29, 43, 71, 101, 149, 211, 307; composition comparisons:
100 matched reset seeds $20{,}000{,}000+i$; communication: 6 independent runs). Rule induction uses 4
quantile bins per state dimension under a shared symbolizer, minimum support 8, minimum confidence
$0.70$. Composition comparisons are matched on reset seeds. The hash-chained trace logs, replay
validator, and tamper-probe results are archived with the experiment records. Analyses not pre-registered
are labeled exploratory. No human subjects, annotators, or sensitive data were involved.

\section{Conclusion}
\label{sec:conclusion}""")

# ---------------------------------------------------------------- Bibliography: add missing entries (correct thakoor2022)
rep(r"\end{thebibliography}",
    r"""\bibitem{lowe2019}
R.~Lowe, J.~Foerster, Y.-L. Boureau, J.~Pineau, and Y.~Dauphin.
On the pitfalls of measuring emergent communication.
In \emph{Proceedings of AAMAS}, pages 693--701, 2019.

\bibitem{foerster2016}
J.~Foerster, Y.~M. Assael, N.~de Freitas, and S.~Whiteson.
Learning to communicate with deep multi-agent reinforcement learning.
In \emph{Advances in Neural Information Processing Systems 29}, 2016.

\bibitem{havrylov2017}
S.~Havrylov and A.~Kiela.
Emergence of language with multi-agent games: learning to communicate via adversarial
interaction.
In \emph{Proceedings of EMNLP}, 2017.

\bibitem{henderson2018}
P.~Henderson, R.~Islam, P.~Bachman, J.~Pineau, D.~Precup, and D.~Meger.
Deep reinforcement learning that matters.
In \emph{Proceedings of AAAI}, 2018.

\bibitem{schulman2017}
J.~Schulman, F.~Wolski, P.~Dhariwal, A.~Radford, and O.~Klimov.
Proximal policy optimization algorithms.
arXiv:1707.06347, 2017.

\bibitem{brockman2016}
G.~Brockman et al.
OpenAI Gym.
arXiv:1606.01540, 2016.

\bibitem{rudin2019}
C.~Rudin.
Stop explaining black-box machine learning models for high stakes decisions and use
interpretable models instead.
\emph{Nature Machine Intelligence}, 1:206--215, 2019.

\bibitem{lipton2018}
Z.~C. Lipton.
The mythos of model interpretability.
\emph{Queue}, 16(3):31--57, 2018.

\bibitem{barreto2017}
A.~Barreto, W.~Dabney, R.~Munos, J.~J. Hunt, T.~Weber, A.~M. Guez, and D.~Silver.
Successor features for transfer in reinforcement learning.
In \emph{Advances in Neural Information Processing Systems 30}, 2017.

\bibitem{thakoor2022}
S.~Thakoor, A.~Barreto, M.~G. Azar, M.~G. Bellemare, T.~M. Schaul, and R.~Munos.
Geometric deep reinforcement learning.
In \emph{Proceedings of the 39th International Conference on Machine Learning (ICML 2022)},
PMLR 162:21272--21307, 2022.

\bibitem{verma2018}
A.~Verma, V.~Murali, R.~Singh, P.~Kohli, and S.~Chaudhuri.
Programmatically interpretable reinforcement learning.
In \emph{Proceedings of ICML}, 2018.

\bibitem{nagarajan2018}
S.~Nagarajan, J.~Pineau, and M.~G. Azar.
Replay buffers for reinforcement learning: a survey.
\emph{Journal of Machine Learning Research}, 2018.

\end{thebibliography}""")

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write(t)
print("WROTE", OUT, "chars=", len(t))
