# BioSentinel labeling guide (reference standard)

Write one paragraph per class before labeling more data. A labeler should be
able to decide every image from this page alone.

## burn

Thermal/chemical injury with diffuse erythema, blistering, or charred tissue
covering a contiguous area. Borderline: sunburn-like redness without tissue
break vs `cuts_infection` — redness with a clear laceration or puncture edge
is not burn. Required: in-focus wound area filling a reasonable frame, no
dominant flash washout.

## cuts_infection

Open cut, laceration, or abrasion showing infection signs: spreading erythema,
swelling, pus, or foul exudate around a visible break in the skin. Borderline:
clean fresh cut without infection signs — exclude or flag for review rather
than forcing a label. Required: wound edges visible, lighting good enough to
judge exudate color.

## necrosis

Black, brown, or dark-gray devitalized tissue (eschar/slough) present in the
frame, however small. Borderline: dark scab over an otherwise healing cut —
necrosis wins if devitalized tissue covers a non-trivial patch; otherwise
flag. Required: color fidelity (no heavy filters); dark areas must read as
tissue, not shadow — shadow doubt means re-shoot or exclude.

## tissue_wound

Exposed viable tissue/granulation bed without the defining signs above: pink
or red granular tissue, no dominant burn pattern, no frank infection signs, no
necrotic patch. Borderline: mixed wounds — if any other class's defining sign
is clearly present, prefer that class; pure granulation beds stay here.

## Reliability protocol

- Two readers label independently; a third breaks ties (consensus label).
- Double-read ~10% of images independently and report inter-rater Cohen's
  kappa per class pair; target kappa ≥ 0.80 before production use.
- Sequential check-and-correct review does not count as a reliability
  statistic — kappa needs independent parallel annotations.
- Log skin tone, device, lighting, body site, and date per image for later
  fairness and drift audits.
