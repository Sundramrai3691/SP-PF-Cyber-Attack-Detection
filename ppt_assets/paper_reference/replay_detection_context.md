# Replay Detection: Paper Context vs This Prototype

The paper motivates replay detection through likelihood-ratio behaviour under its own system and observation model. This prototype uses delayed synthetic noisy measurements (20-sample window) and a normal-reference particle likelihood degradation with thresholds originally calibrated for FDIA only.

- No threshold was retuned for replay.
- Intensity is a dimensionless blend in [0,1] between the current measurement (0) and delayed history (1).
- Under the tested matrix both detectors produced 0% detection at every tested blend across five seeds. False-alarm rate was also 0%.
- Therefore this prototype does not claim replay detection.
