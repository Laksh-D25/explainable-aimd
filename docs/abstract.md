# Abstract

**Title:** Cross-generator detection of AI-generated music fails against unseen
architectures: evidence from a controlled real corpus

---

Generative systems such as Suno and Udio now synthesise complete songs that listeners struggle to distinguish from human-composed music, and detectors built to identify them are known to generalise poorly: reported F1 falls from 0.99 in-distribution to 0.629 on an unseen generator. We build a detector on the frozen MERT music foundation model (94.4 million parameters) with a 0.62 million-parameter attention-pooling head and evaluate it under a protocol designed to separate two failure modes that are routinely reported as one. Withholding Suno or Udio and testing on the other costs almost nothing (F1 degradation -0.037 and -0.023 against -0.361 reported previously), a result that invites the conclusion that such detectors generalise across generators. They do not. When we hold the corpus of real music genuinely fixed and vary only the generator, so that the same 58 MusicCaps recordings supply the real class throughout while the synthetic class moves from Suno/Udio to five unseen text-to-music models, AUROC falls from 0.996 to 0.531, chance, with the false-positive rate unchanged at 0.069 on both sides. The detector recognises real music exactly as well as before; it has stopped recognising generation. Against AudioLDM2 it performs below chance (AUROC 0.277), ranking generated audio as more authentic than human recordings. Suno and Udio transfer to each other only because they are near-identical commercial systems. We further separate this loss of discrimination from a distinct threshold failure, in which changing the corpus of real music drives the false-positive rate to 1.00 while leaving ranking intact, and show that reporting F1 alone conceals both: our cross-corpus F1 of 0.750 is exactly that of a classifier labelling every input synthetic. Ablations show the architecture is not what produces the in-distribution accuracy, and explanation audits show gradient-based temporal attribution is faithful for real audio and unfaithful for generated audio.

---

**Word count:** ~299.

**Superseded.** An earlier draft argued the reverse: that cross-generator
transfer was near-complete and the corpus of real music was what broke
detection. The shared-real experiment refuted it. Holding the real class
literally fixed, AUROC still falls to 0.531 while the false-positive rate does
not move, so the discrimination loss is generator-driven. The FPR 1.00 seen on
FakeMusicCaps is a separate threshold failure.

## Limitations to state in the paper (not the abstract)

* In-distribution F1 saturates: validation reaches 1.000 at epoch 0, so the
  measured collapse is from a ceiling.
* 910 songs against SONICS's 97,164; the real class was fetched from YouTube at
  ~85 % coverage and the splits were regenerated, so the in-distribution figure
  is not directly comparable to SONICS's published ~0.97.
* Suno and Udio are both commercial end-to-end song generators, a closer pair
  than the base paper's setting; the near-complete transfer should not be read
  as generalisation to arbitrary architectures.
* Cross-corpus and cross-generator are confounded in the FakeMusicCaps arm —
  corpus, sample rate and clip length all change together. The claim is that the
  real class matters, evidenced by FPR 1.00; isolating it fully needs a shared
  real corpus.

## Journal fit (Elsevier)

Ranked by how well the *negative/diagnostic* framing fits the venue:

| Journal | Why it fits |
|---|---|
| **Forensic Science International: Digital Investigation** | Media forensics, detector reliability, false-accusation cost — the FPR framing is native here |
| **Journal of Information Security and Applications** | Published Yu et al. on explainable speech-spoof detection; audio forensics in scope |
| **Digital Signal Processing** | Audio ML with a signal-level diagnostic contribution (bandwidth, loudness confounds) |
| **Computer Speech & Language** | Audio anti-spoofing and representation-learning evaluation |
| **Expert Systems with Applications** | Applied ML with a deployment-risk angle (calibration, operating points) |
| **Information Fusion** | Weaker fit unless the three-level explanation layer is foregrounded as fusion |

Also worth considering outside Elsevier: *IEEE TIFS* (published Xie et al. on
audio-deepfake domain generalisation) and *TISMIR*, which published the base
paper and would take a direct rebuttal-style contribution.
