# Ceramic Substrate Inspection Quality Control SOP

## Purpose and Scope
This procedure covers automated visual inspection of ceramic tiles and substrates
on the production line. It defines the inspection tiers, defect categories, human
review steps, and feedback handling. It applies to all inspections performed by
the AI inspection service.

## Inspection Tiers and Disposition Policy
Every inspected part receives a tier. ACCEPT: the part passed automatically and
continues to the next process. REVIEW: the AI is uncertain, an inspector must
examine the part before disposition. REJECT: the part failed automatically and is
removed for quarantined inspection. Tier thresholds are calibrated so that at most
five percent of normal parts fall above the review threshold.

## Defect: Crack
Cracks are linear fractures that may appear on the ceramic surface. Even hairline
cracks compromise mechanical and electrical reliability. Handling: examine under
raking light and magnification. Any visible crack is an automatic reject; do not
attempt rework. Record the crack location in the inspection log.

## Defect: Glue Strip
Glue strips are adhesive residue lines on the surface, often from protective
films or fixture contact. Handling: verify whether the residue is within a
non-functional zone. If removable by approved cleaning, re-clean and re-inspect;
if on a metallization or bonding area, reject the part.

## Defect: Gray Stroke
Gray strokes are surface marking or contamination streaks. Handling: compare
against the boundary samples under standard lighting. If the stroke is cosmetic
and outside functional areas, classify as acceptable with remarks; otherwise
reject.

## Defect: Oil
Oil contamination appears as translucent patches that alter surface reflectivity.
Handling: clean with approved solvent, then re-inspect. If the patch persists or
lies on a bonding surface, reject. Oil on metallization areas risks solder and
adhesion failures downstream.

## Defect: Rough
Rough surface is a texture deviation from the reference finish, often from
grinding or polishing process drift. Handling: measure surface roughness with a
profilometer if available. Reject if outside specification limits; a drift in
roughness across multiple parts should be escalated as a process issue, not a
single-part issue.

## Human Review Procedure
For parts in the review tier: place the part on the inspection mat under standard
lighting, compare with the boundary samples, use the anomaly heatmap as a guide
for where to look, and record the verdict in the dashboard. The verdict must be
labeled normal or defective. Never approve a part that shows any crack, active
contamination, or missing material.

## Feedback and Retraining
Every human verdict is stored together with the image score and tier. When enough
reviewed cases accumulate, the model is retrained and recalibrated with the
recorded feedback so that repeated mistakes are reduced. Reviewers should label
borderline cases honestly, because these teach the model most.

## Escalation and Records
If the same defect type repeats on more than three consecutive parts, stop the
feed and escalate to the line supervisor. All inspections, tiers, and human
verdicts are retained for quality audits. Records include the image filename,
anomaly score, tier, timestamp, and reviewer label.
