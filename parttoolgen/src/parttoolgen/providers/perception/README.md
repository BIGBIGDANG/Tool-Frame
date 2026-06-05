# Perception Providers

Perception providers normalize deployable sensing modules into PartToolGen
artifacts. Examples include RGB-D Florence/SAM2, FoundationPose-style pose
tracking, CAD registration, marker or mocap tracking, and mask/keypoint
trackers.

They should emit provenance fields such as provider id, source kind,
deployability, oracle/debug status, calibration chain, and frame contract.
