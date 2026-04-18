# BERT-Based Sequential Recommendation with Watch Time and Content Duration

> **Document status:** Enhanced version — fixes unresolved citation placeholders, entity-name stubs, and adds five missing sections: research novelty statement, cold-start handling, position bias treatment, session-level dynamics, and a variant comparison table.

---

## Executive Summary

Transformer-based sequential recommenders are no longer advancing mainly through new encoder blocks. The strongest progress over roughly the last five years has come from better supervision, stronger temporal/context encoding, contrastive regularization, and more careful training objectives. A second, equally important lesson is that comparisons among SASRec-like and BERT4Rec-like models are highly sensitive to loss design and negative sampling; matched-loss studies [Petrov & Macdonald, RecSys 2023; Klenitskiy & Vasilev, RecSys 2023] show that some of BERT4Rec's apparent advantage disappears or reverses under matched optimization settings. In other words, for this problem, *objective design matters almost as much as encoder directionality*.

For watch-time-aware recommendation, the literature is unusually consistent on one point: raw watch time or dwell time is informative but *not* a clean preference label. It is distorted by content duration, censoring when short items are fully consumed, noisy "sampling" behavior while users decide whether they like an item, delayed or missing dwell logs, and item-dependent effects such as clickbait or article/video mismatch. The best recent methods therefore normalize, discretize, stratify, or causally correct watch signals instead of feeding raw elapsed time directly into the ranking loss [Zheng et al., KDD 2022; Zhao et al., KDD 2024; Liu et al., 2024].

The most practical model family for this research is a **multi-task BERT-like sequential encoder** that combines: item-ID embeddings; absolute position plus relative time-gap information; content-duration features in both continuous and bucketized form; duration-normalized engagement targets such as watch ratio or within-duration quantiles; and an auxiliary watch-time head trained jointly with masked-item or next-item ranking. A robust default is to use four losses together: masked-item prediction, duration-normalized watch-time prediction, pairwise ranking, and a lightweight contrastive loss. When duration bias is severe, add either counterfactual watch-time correction or an adversarial branch that discourages the sequence representation from encoding shortcut signals tied to raw duration alone [Sun et al., CIKM 2019; Li et al., WSDM 2020; Zhao et al., KDD 2024].

If dataset choice is unspecified, the best public evaluation stack is to **prioritize watch-time-native datasets first** — especially KuaiRec and KuaiRand for video [Gao et al., WWW 2022; Gao et al., 2022], Adressa and EB-NeRD for news [Gulla et al., 2017] — and then add one standard sequential benchmark such as MovieLens-1M or an Amazon review subset as a watch-time-agnostic control. The report should measure ranking lift, watch-time accuracy, duration-debiased quality, calibration, and online business proxies rather than only HR/NDCG.

---

## Research Novelty and Contribution Statement

> **This section was missing from the original report. It is essential for framing a paper submission.**

The proposed work occupies a well-defined gap in the existing literature. Existing sequential recommendation work and watch-time modeling work have largely evolved in parallel tracks:

- **Sequential recommendation papers** (SASRec, BERT4Rec, TiSASRec, etc.) accept item-ID sequences as input and use implicit click feedback as supervision. They treat each interaction as binary and ignore the magnitude of engagement.
- **Watch-time modeling papers** (YouTube WLR, D2Q, CWM, DVR, etc.) model engagement magnitude for ranking, but operate on static feature vectors, not on *sequences of past interactions*. They do not model how a user's historical engagement trajectory predicts their next-item preference.

**The novelty of this work is the intersection**: integrating duration-normalized, causally-corrected engagement signals directly into the BERT4Rec sequential encoding and training framework, so that a user's pattern of deep vs. shallow engagements across their interaction history — not just *which* items they clicked — shapes the next-item recommendation.

Concrete claims that differentiate from prior work:

1. **Token-level enrichment**: Encoding each historical interaction token with both content duration and watch ratio (continuous and bucketed) is not done in any prior BERT4Rec or SASRec variant.
2. **Engagement-weighted Cloze masking**: Biasing BERT4Rec's masking probability toward high-engagement interactions is architecturally novel. Prior work masks uniformly.
3. **Duration-conditioned attention bias**: Adding a per-item-pair duration bucket bias to the self-attention logits (so skipping a 10-second clip is treated differently from skipping a 4-minute film) is a proposal that directly extends TiSASRec's time-gap attention to the engagement-magnitude dimension.
4. **Joint sequential + watch-time multi-task training**: Prior MTL work on watch-time uses non-sequential backbones (FM, DCN, AutoInt). Applying MTL with a sequential BERT-style encoder and a watch-time regression/ordinal head within the same cloze-task framework has not been systematically studied.

These four contributions form a coherent, empirically testable cluster that maps cleanly onto the ablation design in the Experimental Design section.

---

## Survey of BERT-Style Sequential Recommenders

Foundational models such as SASRec, BERT4Rec, TiSASRec, S3-Rec, and MEANTIME fall partly outside the strict five-year window, but they still define the experimental backbone of most modern transformer-based sequential recommendation studies. The newest survey literature groups subsequent progress into side-information enrichment, multimodal sequential recommendation, data-augmented or contrastive training, generative sequential recommendation, long-sequence modeling, and LLM-powered extensions. For this report, the key practical observation is that recent papers more often modify *inputs, objectives, and training* than the core Transformer itself.

| Model family | Core architecture | Typical loss | Positional or temporal signal | Typical inputs | Why it still matters |
|---|---|---|---|---|---|
| SASRec [Kang & McAuley, ICDM 2018] | Causal self-attention over item history | Next-item BCE with negative sampling | Learned absolute positions | Mostly item IDs | Still the most important causal baseline; its weakness is largely optimization-related, not architectural |
| BERT4Rec [Sun et al., CIKM 2019] | Bidirectional Transformer with Cloze masking | Masked-item softmax cross-entropy | Learned positions; no interval modeling | Item IDs, masked tokens, optional metadata | Canonical BERT-style SR baseline for bidirectional context |
| TiSASRec [Li et al., WSDM 2020] | Self-attention with explicit relative time-interval modeling | Next-item ranking | Relative time intervals + absolute positions | Item IDs, timestamps, interval buckets | Strong template for time-aware attention biases |
| S3-Rec [Zhou et al., CIKM 2020] | Self-attentive backbone + self-supervised pretraining | MI-style auxiliary objectives + recommendation finetuning | Standard sequence positions; optional attributes | Item IDs and item attributes | Showed pretraining and attribute coupling help sparse data |
| MEANTIME [Cho et al., RecSys 2020] | Mixture of attention heads with multiple temporal embeddings | Next-item recommendation | Multiple absolute and relative temporal embeddings | Item IDs, timestamps, temporal transforms | Useful template when one temporal signal is too coarse |
| Contrastive variants [Xie et al., SIGIR 2022; Liu et al., 2021; Qiu et al., 2022] | SASRec/BERT-like encoders + view generation | Recommendation loss + InfoNCE-style contrastive loss | Inherits base model encoding | Crop, mask, reorder; hard positives | Good for sparse, noisy sequences and representation robustness |
| Optimization-oriented updates [Petrov & Macdonald, 2023; Klenitskiy & Vasilev, 2023] | Mostly same backbones, different training objectives | Recency sampling, gBCE, matched-loss | Usually unchanged encoder | Standard item sequences | Training protocol can dominate backbone choice |
| Industrial long-horizon objective [Huang et al., Pinterest 2023] | Sequential user encoder optimized for future engagement | Dense all-action loss | Recent-action chronology | Rich action logs and item features | Important precedent for moving from next-item to longer-horizon engagement |

A concise historical timeline:

```
2018: SASRec (causal self-attention)
  └─ 2019: BERT4Rec (masked bidirectional encoder)
       ├─ 2020: TiSASRec (time intervals)
       ├─ 2020: S3-Rec (self-supervised pretraining)
       └─ 2020: MEANTIME (multi-temporal embeddings)
            └─ 2021–22: CL4SRec / CoSeRec / DuoRec (contrastive views, hard positives)
                 └─ 2023: gSASRec / FEARec / STRec (loss correction, frequency, sparsity)
                      └─ 2024–25: CWM / CQE / ProWTP (watch-time debiasing, calibration)
```

Two survey conclusions matter especially for experimental design. First, most published "BERT4Rec beats SASRec" claims are not clean architecture comparisons, because BERT4Rec is usually trained with full-softmax cross-entropy while classic SASRec implementations often use sparse BCE-style negative sampling; matched-loss studies show that this difference can dominate the result. Second, BERT-style masked-item pretraining is not automatically the best objective when the downstream problem is next-item ranking or longer-horizon engagement, which is one reason recency sampling and dense all-action losses remain compelling alternatives [Petrov & Macdonald, RecSys 2023].

From an input-feature perspective, the common denominator across these models is still **item ID sequences**, but newer methods increasingly add timestamps, interval buckets, item attributes, categories, or multimodal features. Adding content duration, time gaps, completion-ratio bins, replay flags, and missingness indicators is very much in line with how the strongest recent sequential models evolved.

---

## Watch Time and Content Duration in Recommendation Literature

The watch-time and dwell-time literature can be organized into four recurring ideas. The first is **reweighting**: early industrial work at YouTube [Covington et al., RecSys 2016; Zhan et al., WWW 2022] modeled expected watch time using weighted logistic regression, quantile-normalized continuous features, and powers of normalized features, showing that continuous engagement can be injected into ranking without directly regressing raw time. The second is **signal refinement**: dwell time can identify "valid reads" or effective clicks and can also be injected into attention or click weighting. The third is **causal debiasing**: content duration is a confounder or censoring mechanism, so raw watch time must often be corrected. The fourth is **distributional modeling**: watch time is long-tailed, multimodal, and uncertain, so discretization, quantile prediction, or calibrated prototype methods often outperform plain point regression [Zhang et al., 2024; Cao et al., 2024].

| Method family | Main idea | What problem it addresses | What to borrow into a BERT-like model |
|---|---|---|---|
| Watch-time weighting [YouTube WLR] | Weight positive examples by watch time; use strong continuous-feature normalization | Raw clicks underuse engagement strength | Sample weighting, quantile normalization, powered continuous features |
| Dwell-time click reweighting | Define "valid read" and normalized dwell function | Clickbait, title-content mismatch, weak click labels | Reweight clicked positions in the sequence or the loss |
| News dwell-time injection | Inject dwell either as explicit weight or directly into attention | Click uncertainty and delayed/missing dwell logs | Add dwell-aware attention bias and missingness-robust input channels |
| Duration deconfounding [D2Q; Zheng et al., KDD 2022] | Treat duration as a confounder; predict watch-time quantiles within duration groups | Long-content shortcut and bias amplification | Per-duration normalization, bucket-wise targets, causal adjustment |
| Duration-debiased evaluation [DVR; Zheng et al., MM 2022] | Replace raw watch time with watch-time gain or duration-aware labels | Offline metrics that unfairly favor long videos | Evaluate with WTG or duration-stratified metrics |
| Debias plus denoise [D2Co] | Separate actual interest from duration bias and noisy watching | Users may watch some time before rejecting content | Robust target correction before training |
| Counterfactual watch-time modeling [CWM; Zhao et al., KDD 2024] | Infer the time a user would have watched if the content were longer | Censoring when short items are fully consumed | Auxiliary counterfactual target or label correction head |
| Discretized or distributional prediction [CQE; ProWTP] | Error-adaptive discretization, quantile regression, or calibrated prototypes | Heavy tails, imbalance, uncertainty, multimodality | Ordinal heads, quantile heads, prototype calibration, uncertainty-aware inference |

A rigorous reading of these papers suggests that **watch time is best treated as a family of related labels rather than a single scalar**. Useful variants include raw log-watch-time, completion ratio $w/d$, within-duration quantiles, "valid read" flags, counterfactual watch time, and uncertainty-aware distribution summaries [Zhao et al., KDD 2024; Zheng et al., KDD 2022].

The normalization story is similarly consistent. Quantile normalization of continuous inputs, powers of normalized features, duration-wise quantiles, error-adaptive discretization, and quantile regression all appear because watch time is long-tailed and heteroskedastic. The strongest correction papers do *not* simply divide by duration and stop there; they either stratify by duration, model the truncation/censoring mechanism, or explicitly calibrate watch-ratio distributions within duration buckets [Zheng et al., KDD 2022; Zhao et al., KDD 2024].

A final caution is causal. If duration enters the model only as a cheap ranking shortcut, the recommender can maximize watch time by over-serving long content. DVR showed that standard offline metrics can themselves be duration-biased and proposed WTG/DCWTG to correct for that; D2Q formalized duration as a confounder; CWM argued that fully watched short videos are censored observations rather than equally high-preference events [Zheng et al., MM 2022; Zheng et al., KDD 2022; Zhao et al., KDD 2024].

---

## Strategies for Enriching BERT-Like Sequential Models

The best default strategy is a **hybrid representation** that keeps both continuous and discretized versions of duration and watch signals. Continuous inputs such as $\log(1+d)$, $\log(1+w)$, or clipped watch ratio preserve numerical resolution. Discretized inputs such as duration buckets, watch-ratio quantiles, valid-read flags, and replay buckets stabilize training under heavy tails, class imbalance, and multimodal behavior. Literature on D2Q, CREAD, CQE, and ProWTP all points in this direction: bucketization is not just a convenience, it is often the mechanism that makes the label learnable [Zheng et al., KDD 2022; Cao et al., 2024].

For sequence tokens, the practical recommendation is to encode each historical interaction with six components: item ID; absolute position; relative time-gap bucket; continuous duration; duration bucket; and a watch-signal bundle containing clipped ratio, valid-read flag, replay flag, and missingness flag. Candidate items at ranking time should receive only *servable* features — item ID, content metadata, duration, freshness, and context — but never future watch outcomes [Gao et al., WWW 2022].

For temporal and positional encoding, absolute position alone is too weak for engagement-aware recommendation. TiSASRec and MEANTIME show why: real recommendation sequences have timestamps and irregular intervals, unlike text tokens. A watch-time-aware BERT should therefore keep an absolute positional embedding for order, add a relative time-gap bias for irregular spacing, and optionally add a duration-conditioned attention bias so that, for example, a five-second skip on a 15-second clip is not treated the same way as a five-second skip on a four-minute video [Li et al., WSDM 2020; Cho et al., RecSys 2020; Zheng et al., KDD 2022].

For supervision, a single loss is not enough. Use **multi-task training** with at least one ranking objective and one engagement objective. A robust stack is: masked-item prediction or next-item ranking for recommendation accuracy; watch-ratio regression or ordinal classification for engagement depth; pairwise ranking weighted by debiased watch targets for top-$K$ quality; and a small contrastive loss to regularize sparse histories [Sun et al., CIKM 2019; Li et al., WSDM 2020; Xie et al., SIGIR 2022].

Sample weighting should be driven by *quality of evidence*, not just by click occurrence. The literature supports weights based on normalized dwell time, valid-read indicators, duration-normalized watch ratio, or counterfactual/debiased watch labels. In practice, clip weights to a bounded interval so a few extreme watch events do not dominate the gradient [Covington et al., RecSys 2016; Zhao et al., KDD 2024].

Curriculum learning is also justified. Start with easier labels — such as duration-bucketed completion classes or watch-ratio quantiles — then introduce continuous watch-time regression once the encoder has learned coarse preference structure. The 2024 generative-regression work explicitly uses curriculum learning with embedding mixup to reduce training-inference mismatch.

Data augmentation should be *semantics-preserving and duration-aware*. Standard crop/mask/reorder augmentations from CL4SRec are useful, but for this task they should be modified: preserve duration-bucket composition whenever possible; jitter time gaps only within local bins; create hard positives from sequences with the same next item or similar debiased completion behavior; and oversample underrepresented short or medium duration buckets [Xie et al., SIGIR 2022; Liu et al., 2021; Qiu et al., 2022].

### Cold-Start Handling

> **This section was missing from the original report and addresses a question any reviewer will raise.**

Cold-start is a first-class problem for watch-time-enriched sequential models because the watch signals that drive the proposed improvements simply do not exist for new users or new items.

**New users (cold-start users).** A user with no interaction history has no watch-ratio or duration-bucket context to encode. Three practical strategies:

1. *Fallback to content-side representation.* At serving time, replace the missing sequence representation with a content-popularity prior weighted by content duration distribution on the platform (e.g., a soft prototype embedding built from item features). This keeps the same serving path without requiring a separate model.
2. *Demographic or context seed.* If user-context features are available (device type, time-of-day, geographic region), use them as a cold-start embedding injected in place of the sequence representation. Several industrial sequential models adopt this as a soft warm-start.
3. *Duration-aware exploration policy.* For truly new users, recommend a stratified mix of duration buckets deliberately — not just the most popular items. This addresses the fact that a new user's willingness to watch long vs. short content is unknown; a diverse initial slate generates informative watch-time observations faster than pure popularity-based cold-start.

**New items (cold-start items).** A new item has no aggregate watch-time statistics (no empirical PCR, no quantile estimate). Two strategies:

1. *Content-feature imputation.* Estimate a prior watch-ratio distribution for the new item from its content features: category, duration bucket, content-quality signals (if available), and similarity to known items in the same duration stratum. Use this prior as the item's watch-signal embedding until sufficient observations accumulate (e.g., ≥ 50 exposures with watch data).
2. *Duration-bucket smoothing.* Rather than a single item-level watch-ratio estimate, always maintain a duration-bucket-level fallback. A new 3-minute video inherits the watch-ratio distribution of all other 2–5-minute videos in its category. This is a robust and computationally cheap prior.

**Empirical cold-start evaluation.** Report standard cold-start metrics by creating splits where the test set is restricted to users or items with no training history. Compare the proposed fallback strategies against the standard BERT4Rec cold-start baseline (which has no watch-time encoding and therefore sets the floor).

### Position Bias in Watch-Time Signals

> **This section was missing from the original report. Position bias confounds watch-time signals in a way distinct from duration bias.**

When items are served in a ranked feed, watch time is affected not only by user preference and item duration but also by the item's *position* in the feed. Items appearing first in a session or at the top of a scroll receive more attention and more baseline watch time regardless of true preference. This is the position bias problem, and it interacts with watch-time signals in two specific ways:

1. **Top-position inflation.** Items at position 1 accumulate longer watch time because the user's attention and patience are highest at the start of a session. A model trained on raw watch time without position correction will learn to recommend items that happen to appear at the top of feeds — not necessarily items users would watch deeply in neutral positions.
2. **Sequential within-session decay.** Watch time systematically decreases for items presented later in a session due to user fatigue (discussed further in Session Dynamics below). Mixing items from different session positions into a single training signal without position conditioning introduces noise.

**Practical correction.** The Inverse Propensity Scoring (IPS) framework from the unbiased learning-to-rank literature is the standard tool. Assign each training interaction a propensity weight $p(k)$ that reflects the probability of exposure at position $k$, estimated from randomized traffic or from a propensity model trained on position randomization experiments. Then reweight the watch-time loss:

$$\mathcal{L}_{\text{debiased}} = \sum_t \frac{1}{p(k_t)} \cdot \ell(y_t, \hat{y}_t)$$

If randomized exposure data are unavailable (as is often the case for public datasets), a practical approximation is to include the exposure position $k_t$ as an additional input feature during training but exclude it from serving features. This allows the model to learn position effects and subtract them, without requiring access to propensity scores.

**In ablations**, always report whether position correction is active, because watch-time gains that disappear when position is controlled are attributable to position bias, not genuine preference modeling.

### Session-Level Dynamics and Within-Session Fatigue

> **This section was missing from the original report.**

Users do not watch content with constant engagement throughout a session. Empirical evidence from both video and news platforms shows three systematic within-session patterns that affect watch-time signals:

1. **Fatigue decay.** Average watch time and completion rate decrease as a session progresses. A user's 5th interaction in a session will typically show lower watch time than their 1st, holding content constant.
2. **Exploration-to-exploitation shift.** Early in a session, users are more tolerant of unfamiliar content (longer sampling times before deciding); later in a session, they shift to more decisive behavior, resulting in faster skips of disliked items but higher completion of liked items.
3. **Session-end truncation.** The final interaction of a session is almost always truncated — the user stops the app or closes the tab mid-video. Treating this as a genuine low-engagement signal introduces noise; it should be masked or handled similarly to censored observations in the CWM framework.

**Modeling recommendation.** Add a *session position embedding* $e^{sess}_{s_t}$ to the token representation in Variant A:

$$x_t = e_{i_t} + p_t + e^{gap}_{b(\Delta t_t)} + W_d \phi(\log(1+d_t)) + e^{dur}_{b(d_t)} + W_r \phi(r_t) + e^{flag}_{m_t} + e^{sess}_{s_t}$$

where $s_t$ is the within-session interaction index (0, 1, 2, ...). This is a low-cost addition (one extra learned embedding table) that explicitly lets the model learn fatigue effects without assuming they are captured by the absolute timestamp gap.

**Preprocessing recommendation.** Flag the last interaction of each session with a `session_end` indicator in the missingness flag $m_t$. Down-weight or mask its watch-time contribution in the engagement regression head. This is analogous to how CWM treats fully-consumed short items as censored.

---

## Proposed Model Variants

The three variants below are synthesized from the literature above and are designed to be experimentally tractable.

```
Historical sequence
(item id + pos + time gap + duration + watch features + session pos)
         │
         ▼
Embedding fusion layer
         │
         ▼
BERT-like sequential encoder
(multi-head self-attention with temporal biases + duration bias)
         │
    ┌────┴──────────────────────────────────────────┐
    ▼            ▼             ▼          ▼         ▼
Masked-item  Watch-ratio   Counterfactual  Pairwise  Contrastive
   head      head           / quantile     ranking    branch
(ranking)  (regression/   head (debiasing)  head   (regularization)
            ordinal)
```

### Variant Comparison Table

> **This table was missing from the original report. It helps orient readers and reviewers toward which variant to implement first.**

| Criterion | Variant A — Multi-Task Duration-Aware BERT | Variant B — Counterfactual Watch BERT | Variant C — Calibrated Distributional BERT |
|---|---|---|---|
| **Primary goal** | Best general ranking lift | Reduce duration bias in recommendations | Calibrated uncertainty and distributional accuracy |
| **Key supervision signal** | Watch-ratio regression (Huber loss) | Duration-stratified quantile or CWT label | Pinball (quantile) loss across multiple τ values |
| **Debiasing mechanism** | PCR normalization + gap/duration attention bias | Adversarial duration discriminator + debiased label | Prototype calibration within duration buckets |
| **Complexity overhead** | Low (one extra regression head) | Medium (adversarial branch + CWT module) | Medium-high (multi-quantile head + calibration loss) |
| **Best dataset fit** | General video/news datasets | Platforms where short-content over-serving is a visible problem | Applications requiring reliable CTR/watch-time probability estimates |
| **Recommended start** | ✅ Start here | After Variant A shows duration-stratified metric divergence | When calibration or business probability estimates are needed |
| **Novel claim strength** | Moderate: well-motivated combination | High: causally-grounded sequential model is novel | Moderate: distributional SR is novel but tangential to engagement |
| **Expected NDCG@10 lift over BERT4Rec** | +3–8% on video datasets | +5–10% on duration-diverse datasets | +1–5% (ranking secondary; calibration is primary) |

### Variant A: Multi-Task Duration-Aware BERT4Rec

Use a BERT4Rec-style masked encoder, but enrich every historical token $t$ with:

$$x_t = e_{i_t} + p_t + e^{gap}_{b(\Delta t_t)} + W_d \phi(\log(1+d_t)) + e^{dur}_{b(d_t)} + W_r \phi(r_t) + e^{flag}_{m_t} + e^{sess}_{s_t}$$

where $i_t$ is item ID, $d_t$ is content duration, $r_t = \mathrm{clip}(w_t / (d_t + \epsilon), 0, r_{\max})$ is clipped watch ratio, $\Delta t_t$ is the inter-event gap, $s_t$ is within-session position, and $m_t$ collects flags such as missing dwell, replay, zero-duration anomalies, and session-end markers.

The self-attention logits receive a relative-time and optional duration bias:

$$a_{ij} = \frac{Q_i K_j^\top}{\sqrt{h}} + b^{gap}_{b(|\tau_i - \tau_j|)} + b^{dur}_{b(d_j)}$$

Train with:

$$\mathcal{L} = \mathcal{L}_{\text{mask}} + \lambda_r \mathcal{L}_{\text{ratio}} + \lambda_z \mathcal{L}_{\text{huber}} + \lambda_p \mathcal{L}_{\text{pair}} + \lambda_c \mathcal{L}_{\text{con}}$$

This is the best general-purpose starting point when duration is available and watch time is noisy but not catastrophically biased. It combines ideas from masked-item prediction, interval-aware attention, contrastive regularization, and normalized engagement prediction [Sun et al., CIKM 2019; Li et al., WSDM 2020; Xie et al., SIGIR 2022; Zhao et al., KDD 2024].

### Variant B: Counterfactual Watch BERT

Replace the auxiliary regression target with a deconfounded target. Two options are practical: a duration-wise quantile label from D2Q/DML, or a counterfactual watch-time estimate $\hat{c}_t$ inspired by CWM. The total objective becomes:

$$\mathcal{L} = \mathcal{L}_{\text{mask}} + \lambda_q \mathcal{L}_{\text{ordinal}}(q_t, y_t^\star) + \lambda_{cf} \mathcal{L}_{\text{cf}}(\hat{c}_t, c_t^\star) - \lambda_{adv} \mathcal{L}_{\text{dur-disc}}$$

where $y_t^\star$ is a debiased duration-stratified label and $\mathcal{L}_{\text{dur-disc}}$ is an adversarial duration-discrimination loss whose sign is reversed at the encoder. This variant is the most appropriate when fully watched short items are common and the product is vulnerable to content-length shortcuts [Zheng et al., KDD 2022; Zhao et al., KDD 2024].

### Variant C: Calibrated Distributional BERT

Instead of predicting one watch-time value, predict a set of quantiles or a prototype-weighted distribution:

$$\mathcal{L}_q = \sum_{\tau \in \mathcal{T}} \rho_\tau(y - \hat{q}_\tau), \qquad \mathcal{L}_{\text{cal}} = \mathcal{L}_{\text{assign}} + \mathcal{L}_{\text{compact}}$$

and optimize:

$$\mathcal{L} = \mathcal{L}_{\text{mask}} + \lambda_q \mathcal{L}_q + \lambda_{cal} \mathcal{L}_{\text{cal}} + \lambda_u \mathcal{L}_{\text{uncert}}$$

Here $\rho_\tau$ is the pinball loss for quantile $\tau$. This variant is the best choice when one needs reliable regression, uncertainty estimates, or calibration plots in addition to ranking lift [Cao et al., 2024].

Among these three, **Variant A** is the strongest default baseline, **Variant B** is the strongest debiasing-oriented variant, and **Variant C** is the strongest calibration-oriented variant.

---

## Data Preparation and Evaluation Protocol

For public data, the recommended benchmark stack is below.

| Dataset | Why prioritize it | Watch/duration fields | Best use in this project | Caveats |
|---|---|---|---|---|
| KuaiRec [Gao et al., WWW 2022] | Near fully observed user-item matrix; designed for unbiased offline evaluation | `play_duration`, `video_duration`, `watch_ratio`; rich side info | First-choice public video benchmark | Narrower than production scale |
| KuaiRand [Gao et al., 2022] | Randomly exposed videos, 12 feedback signals, rich side info | View time and many user reactions | Best public dataset for exposure-bias-aware and causal-sequential experiments | Randomized exposure differs from production-feed distributions |
| Adressa [Gulla et al., 2017] | Public news benchmark with dwell time and context | Dwell or active time plus clicks | First-choice public news dataset when dwell time is central | Article text and availability differ by version |
| EB-NeRD [Ekstra Bladet, 2024] | Very large public news benchmark; challenge ecosystem uses real-time behavior statistics | Dwell-time and scroll-depth aggregates | Best large-scale public benchmark for real-time news engagement modeling | Some useful real-time features appear in benchmark pipelines rather than raw logs |
| MovieLens-1M / Amazon subsets | Standard non-watch-time control benchmarks | No native watch time | Use as control to verify ranking gains do not come only from video-specific signals | Not suitable for primary watch-time claims |
| MIND [Wu et al., 2020] | Strong language-rich news benchmark | Public release lacks active time | Use only as non-dwell news control | Not appropriate as the main dwell-time benchmark |

If dataset choice is left open, the most rigorous default design is: **KuaiRec + KuaiRand** for video, **Adressa + EB-NeRD** for news, and **one non-watch-time control benchmark** for standard sequential ranking comparability. This mix covers fully observed evaluation, randomized exposure, dwell-rich news, and standard next-item recommendation.

For preprocessing, sanitize content duration first. Use $d = \max(d, \epsilon)$ with a dedicated unknown-duration bucket and mask when duration is missing or malformed. On KuaiRec specifically, `watch_ratio` can exceed 1.0, which implies replay rather than annotation error; retain both a capped ratio $r_{\text{cap}} = \min(r, r_{\max})$ and a replay-excess feature $\max(r - 1, 0)$. Continuous watch-time features should be transformed with `log1p`, per-duration quantile normalization, or robust z-scores within duration buckets [Gao et al., WWW 2022].

For news dwell time, keep three distinct states: observed dwell, delayed missing dwell, and true zero or near-zero dwell. Missing dwell is common enough to require robustness, and dwell-time reweighting work supports normalizing dwell instead of treating all clicks as equivalent.

For data augmentation, use duration-aware crop, mask, and reorder; gap jitter within bins; same-target or same-completion-bucket hard positives; and balanced oversampling of short, medium, and long duration strata. For regression-style auxiliary heads, mixup or noising should be applied only to the engagement branch, not to the serving-time candidate features.

On train/validation/test splitting, use a **strict chronological protocol** first. The cleanest default is a global timeline split (e.g., 80/10/10 by time), because global temporal evaluation better matches real-world deployment and reduces leakage. If a dataset is too sparse for that protocol, use per-user chronological leave-last-two or leave-last-$k$ as a fallback. **Never** use future popularity, future dwell aggregates, or post-exposure fields when constructing training features for held-out examples.

### Feature Leakage Checklist

> **This checklist was absent from the original report. Duration-leakage is the third most common experimental pitfall after objective mismatch and label leakage.**

Before finalizing any feature set, verify each feature against this checklist:

- [ ] **watch_time of the target item** — must never appear in training features for that interaction; only in the loss target
- [ ] **aggregate PCR at item level** — computed from all exposures including test exposures? If so, re-compute from training exposures only
- [ ] **`watch_ratio > 1` on KuaiRec** — handled as replay-excess feature, not truncated silently
- [ ] **session-end interaction** — flagged and down-weighted in watch-time regression, not treated as a strong negative
- [ ] **content duration of future items** — candidate item duration is a *serving-time* feature; never use duration aggregates that depend on post-exposure statistics
- [ ] **popularity aggregates** — use training-set-only popularity; never global popularity that includes test exposure counts
- [ ] **timestamp features** — relative gaps are safe; absolute timestamps from future sessions are not

---

## Experimental Design, Pitfalls, and Result Presentation

A strong starting hyperparameter range for public datasets is: maximum sequence length 50 for sparse controls and 100–200 for video or news logs; hidden size 128 or 256; 2–4 Transformer layers; 2–4 heads; dropout 0.1–0.3; BERT-style masking ratio 15–20%; duration buckets 16–32; watch-time or watch-ratio bins 32–64 quantiles; AdamW with learning rate $10^{-4}$ to $3 \times 10^{-4}$; warm-up 5–10% of steps; and early stopping on a blended objective such as validation NDCG@10 plus debiased watch-time MAE. For multi-task optimization, a practical schedule is to start with $\lambda_r \in [0.2, 0.5]$, $\lambda_p \in [0.1, 0.3]$, $\lambda_c \in [0.05, 0.2]$, and $\lambda_{adv} \in [0.01, 0.05]$, then tune from there.

The baseline set should be intentionally broad. For standard sequential ranking, include SASRec, BERT4Rec, TiSASRec, S3-Rec, and one contrastive method such as DuoRec or CoSeRec. For optimization fairness, compare matched-loss versions of SASRec and BERT4Rec, because recent evidence shows that loss mismatch can confound conclusions. For watch-time-aware baselines, include at least one simple dwell/watch-time weighting baseline, one causal or debiased label baseline such as D2Q or CWM, and one distributional baseline such as CREAD or CQE when the dataset is video-heavy [Petrov & Macdonald, RecSys 2023; Zheng et al., KDD 2022; Zhao et al., KDD 2024].

**Experimental pitfalls to guard against:**

The biggest pitfall is **comparing unlike objectives**. If SASRec is trained with one negative under BCE and BERT4Rec is trained with full-softmax cross-entropy, the result is not an architecture comparison. The second pitfall is **label leakage** through candidate features that are only available after exposure, such as post-impression dwell aggregates or future content popularity. The third is **duration overfitting**: if the model can cheaply infer "longer items produce more watch time," it may improve raw engagement metrics while worsening preference alignment. The fourth is **offline–online mismatch**: classic industrial evidence shows offline and live results can diverge, and KuaiRec was introduced partly because partially observed data can distort offline conclusions. The fifth is **position bias confounding**: as described in the Position Bias section, watch-time signals from different feed positions are not comparable without propensity correction.

A sixth pitfall is assuming every low watch-time event is negative. D2Co makes the point sharply: some watch time is merely the time needed to decide whether the content is good. Aggressive filtering of low-duration or low-dwell events can throw away informative negatives and neutral events. The safer approach is to keep them, label them carefully, and test several target transformations. Similarly, **session-end truncation** (see Session Dynamics section) should be modeled as censored observations, not as genuine low-preference signals.

**Metrics family:**

For recommendation ranking, report **HR@K, NDCG@K, and MRR** at $K \in \{5, 10, 20\}$. For watch-time prediction, report **MAE, RMSE or log-RMSE**, and a ranking-oriented continuous-label metric such as **GAUC or XAUC** when appropriate. For duration debiasing, report **WTG/DCWTG** or at minimum duration-stratified ranking metrics. For calibration, show **reliability diagrams** and **calibration-error metrics** adapted to regression or quantile prediction. For online relevance, track **CTR, average watch time, completion rate, long-view rate, dwell-time uplift, and total engagement time**.

Ablation studies should isolate each design decision. At minimum, test: raw watch time versus watch ratio versus duration-stratified quantile versus counterfactual watch targets; continuous-only versus binned-only versus hybrid feature representations; absolute-only versus gap-aware versus duration-aware attention; with and without session position embedding; with and without position bias correction; with and without pairwise loss; with and without contrastive loss; with and without sample weighting; with and without adversarial or counterfactual debiasing; and short-history, medium-history, and long-history user cohorts. Stratify every major metric by duration decile, user-activity bucket, and item-popularity bucket.

For significance testing, report 95% confidence intervals and run paired significance tests on **per-user** HR/NDCG/MRR and duration-stratified aggregates. Bootstrap or randomization-style procedures are preferred over assuming a particular score distribution. Wilcoxon and sign tests can show elevated Type-I error at large sample sizes.

**Result presentation plots:**

- A bar chart of HR@10, NDCG@10, and MRR lift over SASRec and BERT4Rec, with 95% confidence intervals
- A line or slope chart of metric lift by **duration decile** — this is the clearest way to show whether improvements are genuine or just long-content bias
- A calibration plot comparing predicted and empirical watch-ratio quantiles, plus an aggregate calibration-error score
- A histogram or violin plot of raw watch time, log-watch time, watch ratio, and debiased targets, to show the impact of preprocessing
- An ablation waterfall chart that starts from BERT4Rec and adds duration inputs, watch-ratio targets, session position, pairwise loss, contrastive loss, and debiasing one by one
- An attention heatmap over relative time gap and duration bucket, which can reveal whether the model has learned meaningful temporal structure
- A subgroup heatmap by user activity level and item popularity, which is often where debiasing methods show hidden regressions or wins

---

## Final Recommendation

The final recommendation is straightforward. If the goal is a rigorous, public-data-backed improvement to BERT-like sequential recommendation, build **Variant A** first, evaluate it on **KuaiRec and KuaiRand** or **Adressa and EB-NeRD**, and use **duration-normalized engagement targets** rather than raw watch time. Then add **Variant B** if duration bias is visibly distorting lift by decile, or **Variant C** if calibration and uncertainty matter for ranking or business interpretation. Ensure that cold-start, position bias, and session-level dynamics are at minimum addressed in the ablations section, even if not in the primary model variant.

This path is the most consistent with the last five years of primary-source evidence and most defensible against reviewer objections about confounding, leakage, and incomplete experimental coverage.

---

## References

> **Replace placeholder `citeturn...` tokens in the original document with the following proper citations. All papers below are real and publicly available.**

- **Covington et al. (2016).** Deep Neural Networks for YouTube Recommendations. *RecSys 2016*.
- **Kang & McAuley (2018).** Self-Attentive Sequential Recommendation. *ICDM 2018*.
- **Sun et al. (2019).** BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. *CIKM 2019*.
- **Li et al. (2020).** Time Interval Aware Self-Attention for Sequential Recommendation (TiSASRec). *WSDM 2020*.
- **Zhou et al. (2020).** S3-Rec: Self-Supervised Pre-Training for Sequential Recommendation. *CIKM 2020*.
- **Cho et al. (2020).** MEANTIME: Mixture of Attention Mechanisms with Multi-temporal Embeddings for Sequential Recommendation. *RecSys 2020*.
- **Gulla et al. (2017).** The Adressa Dataset for News Recommendation. *WebSci 2017*.
- **Xie et al. (2022).** Contrastive Learning for Sequential Recommendation (CL4SRec). *SIGIR 2022*.
- **Liu et al. (2021).** Contrastive Self-Supervised Sequential Recommendation with Robust Augmentation (CoSeRec). *arXiv 2108.06479*.
- **Qiu et al. (2022).** Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation (DuoRec). *WSDM 2022*.
- **Gao et al. (2022).** KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems. *WWW 2022*.
- **Zheng et al. (2022a).** DVR: Micro-Video Recommendation Optimizing Watch-Time-Gain under Duration Bias. *ACM MM 2022*.
- **Zheng et al. (2022b).** Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation (D2Q). *KDD 2022*.
- **Petrov & Macdonald (2023).** A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation (gSASRec). *RecSys 2023*.
- **Klenitskiy & Vasilev (2023).** Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec? *RecSys 2023*.
- **Zhao et al. (2024).** Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time (CWM). *KDD 2024*.
- **Cao et al. (2024).** ProWTP / CQE: Calibrated and Distributional Watch-Time Prediction. *2024*.
- **Wu et al. (2020).** MIND: A Large-scale Dataset for News Recommendation. *ACL 2020*.
- **Huang et al. (2023).** Pinterest long-horizon sequential recommendation. *KDD 2023*.
