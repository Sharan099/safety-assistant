*KEYWORD
$ Comment lines interspersed inside a card's data must be preserved in
$ raw_text but skipped when locating the real data line.
*MAT_PIECEWISE_LINEAR_PLASTICITY
$ material id, density, E, PR follow on the next real line
$ (this comment sits between the keyword and the data on purpose)
7,7.85e-9,2.1e5,0.3
$ trailing comment, still part of this card's raw_text
*END
