---
type: Concept
title: LS-DYNA Keyword User's Manual, Volume III (Multiphysics), R17
description: Curated excerpt from LS-DYNA Keyword User's Manual, Volume III (Multiphysics),
  R17 (pymupdf-pipeline v0.1.0-first20).
tags:
- solver
- official_documentation
sources:
- document_id: lsdyna-r17-vol-iii
  locator:
    section: INTRODUCTION
    page: 15
authority: official_documentation
status: draft
created_by: scripts/generate_okf_concepts.py
---

In this manual, there are ﬁve main solvers:  two compressible ﬂow solvers, an incom-
pressible ﬂow solver, an electromagnetism solver, and a battery electrochemistry solver.
Each of them implements coupling with the structural solver in LS-DYNA.
The keywords covered in this manual ﬁt into one of three categories.  In the ﬁrst category
are the keyword cards that provide input to each of the multiphysics solvers that in turn
couple with the structural solver.  In the second category are keyword cards involving
extensions to the basic solvers.  Presently, the chemistry and stochastic particle solvers
are the two solvers in this category, and they are used in conjunction with the *CESE
compressible ﬂow solver discussed below.  In the third category are keyword cards for
support facilities.  A volume mesher that creates volume tetrahedral element meshes
from bounding surface meshes is one of these tools.  Another is a data output mechanism
for a limited set of variables from some of the solvers in this manual.  This mechanism is
accessed through *LSO keyword cards.
The CESE solver is a compressible ﬂow solver based upon the Conservation Element/So-
lution Element (CE/SE) method, originally proposed by Chang of the NASA Glenn Re-
search Center.  This method is a novel numerical framework for conservation laws.  It
has many non-traditional features, including a uniﬁed treatment of space and time, the
introduction of separate conservation elements (CE) and solution elements (SE), and a
novel shock capturing strategy without using a Riemann solver.  This method has been
used to solve many types of ﬂow problems, such as detonation waves, shock/acoustic
wave interaction, cavitating ﬂows, supersonic liquid jets, and chemically reacting ﬂows.
In LS-DYNA, it has been extended to also solve ﬂuid-structure interaction (FSI) problems.
It does this with two approaches.  The ﬁrst approach solves the compressible ﬂow equa-
tions on an Eulerian mesh while the structural mechanics is sol