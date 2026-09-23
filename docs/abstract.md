# Abstract (draft)

**Title (working):** What do AI-generated music detectors actually learn?
Cross-generator and cross-corpus evaluation of a MERT-based detector with
attention pooling

---

Generative systems such as Suno and Udio now synthesise complete songs that
listeners struggle to distinguish from human-composed music, and detectors
built to identify them are known to generalise poorly: reported F1 falls from
0.99 in-distribution to 0.629 on an unseen generator. We build a detector on the
frozen MERT music foundation model (94.4 M parameters) with a 0.62 M-parameter
trainable head that aggregates all thirteen hidden states by layer attention and
pools frames and clips by learned attention, and we evaluate it under a protocol
that separates two failure modes usually reported as one. On a balanced subset
of SONICS (910 songs) the detector reaches F1 0.957 with expected calibration
error 0.0002. Holding the real class fixed and withholding an entire generator,
transfer is near-complete in both directions (Suno→Udio F1 0.963, Udio→Suno
0.977; AUROC ≥ 0.9995; F1 degradation −0.037 and −0.023 against −0.361 reported
previously). Evaluated on FakeMusicCaps, where five unseen text-to-music models
are paired with a *different* corpus of real music, the same detector falls to
near chance (AUROC 0.598) and labels every real clip as generated (FPR 1.00,
ECE 0.365). The two results reconcile under one explanation: the detector
encodes what its training corpus's real music looks like rather than what
generation leaves behind, so it survives a new generator but not a new
population of real recordings. We further show that naive evaluation overstates
performance — global loudness and dynamic-range descriptors alone separate the
classes at 0.925 AUROC before normalisation — and that gradient-based temporal
explanations are faithful for real audio but fail entirely for generated audio,
indicating that generation evidence is spectrally distributed rather than
temporally localised. We release the protocol, the shortcut and bandwidth
diagnostics, and the false-positive-rate-on-unseen-real-music test that exposes
the failure.

---

**Word count:** ~270. Trim the shortcut/explanation sentences for a 200-word cap.

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
