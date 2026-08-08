# Docling vs LightOnOCR — per-page OCR diff

Corpus: `H:/AutoSafety_RAG/data/ocr_compare/figure_boundary_v1`
Paired pages: **68** processed, **0** not yet processed (LightOn missing).
Section-number disagreements: **3** / 68
Pages where LightOn has figure/caption text Docling lacks: **11** / 68
Docling clause-as-caption pollution pages: **2**

## Reference case (always first)

**R94 Figure 3 / 5.2.1.7 misattribution (test p.4 = source R94 p.12)**

- Source: `UN-ECE-R94 source p.12 [figure_adjacent,r94_figure3_area]`
- Section disagreement: **no**
- Docling sections: `['5.2.1.3', '5.2.1.4', '5.2.1.5', '5.2.1.6', '5.2.1.7']`
- LightOn sections: `['5.2.1.3', '5.2.1.4', '5.2.1.5', '5.2.1.6', '5.2.1.7']`
- Docling tagged numbered clause(s) as `caption`: `5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;`

### Attribution of `5.2.1.7`

- **Docling:** `5.2.1.7` (or sibling) appears under `<!-- caption -->` after Figure 3 — **misattribution confirmed**.
- **LightOn:** `5.2.1.7` present as body text after Figure 3 / caption — **boundary held**.

<details><summary>Unified diff preview (normalized)</summary>

```diff
--- docling/page_004.md
+++ lighton/page_004.md
@@ -1,4 +1,8 @@
-[IMAGE]
+E/ECE/324/Rev.1/Add.93/Rev.4
 
-5.2.1.3. The neck bending moment about the y axis shall no exceed 57 Nm in extension 3 ;
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4
+
+[IMAGE]223,120,814,353
+
+5.2.1.3. The neck bending moment about the y axis shall no exceed 57 Nm in extension³;
 
@@ -12,6 +16,8 @@
 
-# Femur force criterion
+**Femur force criterion**
 
-[IMAGE]
+[IMAGE]300,555,755,837
 
 5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;
+
+12
```

</details>

## Ranked pages (by disagreement score)

### Page 004 ★ REFERENCE — score=5.0
- Source: `UN-ECE-R94 source p.12 [figure_adjacent,r94_figure3_area]`
- Chars: Docling=638, LightOn=618
- Section disagreement: **no** (docling=5, lighton=5)
- Docling clause-as-caption: `5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;`

<details><summary>Diff preview</summary>

```diff
--- docling/page_004.md
+++ lighton/page_004.md
@@ -1,4 +1,8 @@
-[IMAGE]
+E/ECE/324/Rev.1/Add.93/Rev.4
 
-5.2.1.3. The neck bending moment about the y axis shall no exceed 57 Nm in extension 3 ;
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4
+
+[IMAGE]223,120,814,353
+
+5.2.1.3. The neck bending moment about the y axis shall no exceed 57 Nm in extension³;
 
@@ -12,6 +16,8 @@
 
-# Femur force criterion
+**Femur force criterion**
 
-[IMAGE]
+[IMAGE]300,555,755,837
 
 5.2.1.7. The tibia compression force criterion (TCFC) shall not exceed 8 kN;
+
+12
```

</details>

### Page 061 — score=5.0
- Source: `UN-ECE-R129 source p.31 [docling_figure_page,figure_adjacent]`
- Chars: Docling=1765, LightOn=1675
- Section disagreement: **no** (docling=3, lighton=3)
- Docling clause-as-caption: `6.3.5.1. Support-leg and support-leg foot geometrical requirements`

<details><summary>Diff preview</summary>

```diff
--- docling/page_061.md
+++ lighton/page_061.md
@@ -1 +1,4 @@
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3
+
 Compliance with the requirements specified in paragraphs 6.3.5.1. and 6.3.5.2. below may be verified by a physical or computer simulation.
@@ -6,3 +9,3 @@
 
-(a) The X' axis shall be parallel to the child restraint fixture (CRF) 3 bottom surface and in the median longitudinal plane of the CRF;
+(a) The X' axis shall be parallel to the child restraint fixture (CRF)³ bottom surface and in the median longitudinal plane of the CRF;
 
@@ -14,5 +17,5 @@
 
-[IMAGE]
+[IMAGE]197,395,765,655
 
-6.3.5.1. Support-leg and support-leg foot geometrical requirements
+### 6.3.5.1. Support-leg and support-leg foot geometrical requirements
 
@@ -22,2 +25,6 @@
 
-3 Child restraint fixture (CRF) as defined in UN Regulation No. 16 (Safety-belts)
+---
+
+³ Child restraint fixture (CRF) as defined in UN Regulation No. 16 (Safety-belts)
+
+31
```

</details>

### Page 054 — score=4.1
- Source: `UN-ECE-R129 source p.9 [control]`
- Chars: Docling=3172, LightOn=2942
- Section disagreement: **no** (docling=14, lighton=14)
- LightOn figure/caption not in Docling: `2.17.2. "i-Size booster seat fixture" means a fixture, of the dimensions given in Figure 1 of Annex 17, Appendix 5 to UN Regulation No. 16 and used by an Enhanc`; `¹ Detail B describes the standard dimensions without ISOFIX attachments. Figure 1 gives the dimensions for optional stowable ISOFIX attachments.`

<details><summary>Diff preview</summary>

```diff
--- docling/page_054.md
+++ lighton/page_054.md
@@ -1,29 +1,38 @@
-2.17.2. " i -Size booster seat fixture " means a fixture, of the dimensions given in Figure 1 of Annex 17, Appendix 5 to UN Regulation No. 16 and used by an Enhanced Child Restraint System manufacturer to determine the appropriate dimensions of a i -Size booster seat and its compatibility with most vehicle seating positions and, in particular, those which have been assessed without ISOFIX attachments, 1 according to UN Regulation No. 16 as being compatible with such a category of an Enhanced Child Restraint System.
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3
 
-2.18 . " Childdsafety chair" means an Enhanced Child Restraint System incorporating a chair in which the child is held.
+---
 
-2.19 . " Chair" means a structure which is a constituent part of the Enhanced Child Restraint System and is intended to accommodate a child in a seated position.
+2.17.2. "i-Size booster seat fixture" means a fixture, of the dimensions given in Figure 1 of Annex 17, Appendix 5 to UN Regulation No. 16 and used by an Enhanced Child Restraint System manufacturer to determine the appropriate dimensions of a i-Size booster seat and its compatibility with most vehicle seating positions and, in particular, those which have been assessed without ISOFIX attachments,¹ according to UN Regulation No. 16 as being compatible with such a category of an Enhanced Child Restraint System.
 
-2.20 . " Chair support" means that part of an Enhanced Child Restraint System by which the chair can be raised.
+2.18. "Child-safety chair" means an Enhanced Child Restraint System incorporating a chair in which the child is held.
 
-2.21. " ECRS Belt" means an Enhanced Child Restraint System comprising a combination of straps with a securing buckle, adjusting devices and attachments.
+2.19. "Chair" means a structure which is a constituent part of the Enhanced Child Restraint System and is intended to accommodate a child in a seated position.
 
-2.22. " Harness belt" means an ECRS belt assembly comprising a lap strap, shoulder restraints and a crotch strap.
+2.20. "Chair support" means that part of an Enhanced Child Restraint System by which the chair can be raised.
 
-2.23. " YYshaped belt" means an ECRS belt where the combination of straps is formed by a strap to be guided between the child's legs and a strap for each shoulder.
+2.21. "ECRS Belt" means an Enhanced Child Restraint System comprising a combination of straps with a securing buckle, adjusting devices and attachments.
 
-2.24 . " Carry cot" means a restraint system intended to accommodate and restrain the child in a supine or prone position with the child's spine perpendicular to the median longitudinal plane of the vehicle. It is so designed as to distribute the restraining forces over the child's head and body excluding its limbs in the event of a collision.
+2.22. "Harness belt" means an ECRS belt assembly comprising a lap strap, shoulder restraints and a crotch strap.
 
-2.25 . " Carry-cot restraint" means a device used to restrain a carry-cot to the structure of the vehicle.
+2.23. "Y-shaped belt" means an ECRS belt where the combination of straps is formed by a strap to be guided between the child’s legs and a strap for each shoulder.
 
-2.26 . " Infant carrier" means a restraint system intended to accommodate the child in a rearward -facing semi-recumbent position. It is so designed as to distribute the restraining forces over the child's head and body excluding its limbs in the event of the frontal collision.
+2.24. "Carry cot" means a restraint system intended to accommodate and restrain the child in a supine or prone position with the child’s spine perpendicular to the median longitudinal plane of the vehicle. It is so designed as to distribute the restraining forces over the child’s head and body excluding its limbs in the event of a collision.
 
-2.27 . " Child support" means that part of an Enhanced Child Restraint System by which the child can be raised within the Enhanced Child Restraint System.
+2.25. "Carry-cot restraint" means a device used to restrain a carry-cot to the structure of the vehicle.
 
-2.28 . " Impact shield" means a device secured in front of the child and designed to distribute the restraining forces over the greater part of the height of the child's body in the event of a frontal impact.
+2.26. "Infant carrier" means a restraint system intended to accommodate the child in a rearward-facing semi-recumbent position. It is so designed as to distribute the restraining forces over the child’s head and body excluding its limbs in the event of the frontal collision.
 
```

</details>

### Page 032 — score=4.1
- Source: `UN-ECE-R95 source p.71 [audit_flagged_neighbor,docling_figure_page,figure_adjacent]`
- Chars: Docling=1580, LightOn=1806
- Section disagreement: **no** (docling=0, lighton=0)
- LightOn figure/caption not in Docling: `Prior to the impact a switch $S_1$ and a known discharge resistor $R_c$ is connected in parallel to the relevant capacitance (see Figure 2).`; `When $V_1$ , $V_2$ (see Figure 1) are measured at a point in time between 5 seconds and 60 seconds after the impact and the capacitances of the Y-capacitors ( $`

<details><summary>Diff preview</summary>

```diff
--- docling/page_032.md
+++ lighton/page_032.md
@@ -1,20 +1,28 @@
-Figure 1
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 9 
 
-# Measurement of Vb, V1, V2
+## Figure 1 
+**Measurement of $V_b$ , $V_1$ , $V_2$ **
 
-[IMAGE]
+[IMAGE]175,138,785,387
 
-# Electrical Chassis
+Electrical Chassis
 
-Assessment procedure for low electrical energy
+### 3. Assessment procedure for low electrical energy
 
-Prior to the impact a switch S1 and a known discharge resistor R e is connected in parallel to the relevant capacitance (see Figure 2).
+Prior to the impact a switch $S_1$ and a known discharge resistor $R_c$ is connected in parallel to the relevant capacitance (see Figure 2).
 
-Not earlier than 5 seconds and not later than 60 seconds after the impact the switch S1 shall be closed while the voltage Vb and the current I e are measured and recorded. The product of the voltage Vb and the current I e shall be integrated over the period of time , starting from the moment when the switch S1 is closed (t c ) until the voltage Vb falls below the high voltage threshold of 60 V DC (th) . The resulting integration equals the total energy (TE) in joules.
+Not earlier than 5 seconds and not later than 60 seconds after the impact the switch $S_1$ shall be closed while the voltage $V_b$ and the current $I_c$ are measured and recorded. The product of the voltage $V_b$ and the current $I_c$ shall be integrated over the period of time, starting from the moment when the switch $S1$ is closed ( $t_c$ ) until the voltage $V_b$ falls below the high voltage threshold of 60 V DC ( $t_h$ ). The resulting integration equals the total energy (TE) in joules.
 
-When Vb is measured at a point in time between 5 seconds and 60 seconds after the impact and the capacitance of the X-capacitors (C x ) is specified by the manufacturer, total energy (TE) shall be calculated according to the following formula:
+#### (a) $TE = \int_{t_c}^{t_h} V_b \times I_c dt$
 
-(b) TE = 0.5 x C x x(Vb 2 – 3 600)
+When $V_b$ is measured at a point in time between 5 seconds and 60 seconds after the impact and the capacitance of the X-capacitors ( $C_x$ ) is specified by the manufacturer, total energy (TE) shall be calculated according to the following formula:
 
-When V1, V2 (see Figure 1) are measured at a point in time between 5 seconds and 60 seconds after the impact and the capacitances of the Y -capacitors (Cy1, Cy2) are specified by the manufacturer, total energy (TEy1, TEy2) shall be calculated according to the following formulas:
+#### (b) $TE = 0.5 \times C_x \times (V_b^2 - 3600)$
+
+When $V_1$ , $V_2$ (see Figure 1) are measured at a point in time between 5 seconds and 60 seconds after the impact and the capacitances of the Y-capacitors ( $C_{y1}$ , $C_{y2}$ ) are specified by the manufacturer, total energy ( $TE_{y1}$ , $TE_{y2}$ ) shall be calculated according to the following formulas:
+
+#### (c) $TE_{y1} = 0.5 \times C_{y1} \times (V_1^2 - 3600)$ 
+$TE_{y2} = 0.5 \times C_{y2} \times (V_2^2 - 3600)$
```

</details>

### Page 026 — score=4.0
- Source: `UN-ECE-R95 source p.30 [docling_figure_page,figure_adjacent]`
- Chars: Docling=2632, LightOn=2552
- Section disagreement: **no** (docling=16, lighton=16)
- LightOn figure/caption not in Docling: `2.1.1.3. The blocks must be centred on the six zones defined in Figure 1 and each block (including incomplete cells) should cover completely the area defined fo`; `2.1.2.2. Blocks 1, 2 and 3 should be crushed by $10 \pm 2$ mm on the top surface prior to testing to give a depth of $500 \pm 2$ mm (Figure 2).`

<details><summary>Diff preview</summary>

```diff
--- docling/page_026.md
+++ lighton/page_026.md
@@ -1,41 +1,47 @@
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 5
+
 # Annex 5
 
-# Mobile deformable barrier characteristics
+## Mobile deformable barrier characteristics
 
-Characteristics of the mobile deformable barrier
+1. Characteristics of the mobile deformable barrier
 
-1.1. The mobile deformable barrier (MDB) includes both an impactor and a trolley.
+ 1.1. The mobile deformable barrier (MDB) includes both an impactor and a trolley.
 
-1.2. The total mass shall be 950 ± 20 kg.
+ 1.2. The total mass shall be $950 \pm 20$ kg.
 
-1.3. The centre of gravity shall be situated in the longitudinal median vertical plane within 10 mm, 1,000 ± 30 mm behind the front axle and 500 ± 30 mm above the ground.
+ 1.3. The centre of gravity shall be situated in the longitudinal median vertical plane within 10 mm, $1,000 \pm 30$ mm behind the front axle and $500 \pm 30$ mm above the ground.
 
-1.4. The distance between the front face of the impactor and the centre of gravity of the barrier shall be 2,000 ± 30 mm.
+ 1.4. The distance between the front face of the impactor and the centre of gravity of the barrier shall be $2,000 \pm 30$ mm.
 
-1.5. The ground clearance of the impactor shall be 300 ± 5 mm measured in static conditions from the lower edge of the lower front plate, before the impact.
+ 1.5. The ground clearance of the impactor shall be $300 \pm 5$ mm measured in static conditions from the lower edge of the lower front plate, before the impact.
 
-1.6. The front and rear track width of the trolley shall be 1,500 ± 10 mm.
+ 1.6. The front and rear track width of the trolley shall be $1,500 \pm 10$ mm.
 
-1.7. The wheelbase of the trolley shall be 3,000 ± 10 mm.
+ 1.7. The wheelbase of the trolley shall be $3,000 \pm 10$ mm.
 
-Characteristics of the impactor
+2. Characteristics of the impactor
 
-The impactor consists of six single blocks of aluminium honeycomb, which have been processed in order to give a progressively increasing level of force with increasing deflection (see paragraph 2.1. below). Front and rear aluminium plates are attached to the aluminium honeycomb blocks.
```

</details>

### Page 031 — score=4.0
- Source: `UN-ECE-R95 source p.70 [audit_flagged,audit_flagged_neighbor,figure_adjacent]`
- Chars: Docling=1938, LightOn=1903
- Section disagreement: **no** (docling=1, lighton=1)
- LightOn figure/caption not in Docling: `Before the vehicle impact test conducted, the high voltage bus voltage ($V_b$) (see Figure 1) shall be measured and recorded to confirm that it is within the op`; `After the impact test, determine the high voltage bus voltages ($V_b$, $V_1$, $V_2$) (see Figure 1).`

<details><summary>Diff preview</summary>

```diff
--- docling/page_031.md
+++ lighton/page_031.md
@@ -1,10 +1,16 @@
-# Annex 9
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 9
 
-# Test procedures for the protection of the occupants of vehicles operating on electrical power high voltage and electrolyte spillage
+---
 
-This annex describes test procedures to demonstrate compliance to the electrical safety requirements of paragraph 5.3.7. For example, megohmmeter or oscilloscope measurements are an appropriate alternative to the procedure described below for measuring isolation resistance. In this case it may be necessary to deactivate the on-board isolation resistance monitoring system .
+## Annex 9
 
-Before the vehicle impact test conducted, the high voltage bus voltage (Vb) (see Figure 1) shall be measured and recorded to confirm that it is within the operating voltage of the vehicle as specified by the vehicle manufacturer.
+### Test procedures for the protection of the occupants of vehicles operating on electrical power high voltage and electrolyte spillage
 
-Test setup and equipment
+This annex describes test procedures to demonstrate compliance to the electrical safety requirements of paragraph 5.3.7. For example, megohmmeter or oscilloscope measurements are an appropriate alternative to the procedure described below for measuring isolation resistance. In this case it may be necessary to deactivate the on-board isolation resistance monitoring system.
+
+Before the vehicle impact test conducted, the high voltage bus voltage ($V_b$) (see Figure 1) shall be measured and recorded to confirm that it is within the operating voltage of the vehicle as specified by the vehicle manufacturer.
+
+#### 1. Test setup and equipment
 
@@ -12,3 +18,3 @@
 
-However, if the high voltage disconnect is integral to the REESS or the energy conversion system and the high-voltage bus of the REESS or the energy conversion system is protected according to protection degree IPXXB following the impact test, measurements may only be taken between the device performing the disconnect function and the electrical loads .
+However, if the high voltage disconnect is integral to the REESS or the energy conversion system and the high-voltage bus of the REESS or the energy conversion system is protected according to protection degree IPXXB following the impact test, measurements may only be taken between the device performing the disconnect function and the electrical loads.
 
@@ -16,5 +22,5 @@
 
-The following instructions may be used if voltage is measured.
+#### 2. The following instructions may be used if voltage is measured.
 
-After the impact test, determine the high voltage bus voltages (Vb, V1, V2) (see Figure 1).
+After the impact test, determine the high voltage bus voltages ($V_b$, $V_1$, $V_2$) (see Figure 1).
 
@@ -23 +29,3 @@
 This procedure is not applicable if the test is performed under the condition where the electric power train is not energized.
+
```

</details>

### Page 025 — score=3.2
- Source: `UN-ECE-R95 source p.29 [docling_figure_page,figure_adjacent]`
- Chars: Docling=971, LightOn=1300
- Section disagreement: **YES** (docling=0, lighton=1)
  - only LightOn: `['1.25']`

<details><summary>Diff preview</summary>

```diff
--- docling/page_025.md
+++ lighton/page_025.md
@@ -1,4 +1,10 @@
-# Annex 4 -Appendix 2
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 4 - Appendix 2 
 
-# The procedure for calculating the viscous criterion for EUROSID 1
+---
+
+## Annex 4 - Appendix 2
+
+### The procedure for calculating the viscous criterion for EUROSID 1
 
@@ -6,5 +12,13 @@
 
+$$
+C_{(t)} = \frac{D_{(t)}}{0.14}
+$$
+
 The rib deflection velocity at time (t) is calculated from the filtered deflection as:
 
-where D(t) is the deflection at time (t) in metres and t is the time interval in seconds between the measurements of deflection. The maximum value of t shall be 1,25 x 10 -4 seconds.
+$$
+V_{(t)} = \frac{8 \left[ D_{(t+1)} - D_{(t-1)} \right] - \left[ D_{(t+2)} - D_{(t-2)} \right]}{12 \partial t}
+$$
+
+where $D_{(t)}$ is the deflection at time (t) in metres and $\partial t$ is the time interval in seconds between the measurements of deflection. The maximum value of $\partial t$ shall be $1.25 \times 10^{-4}$ seconds.
 
@@ -12,2 +26,7 @@
 
-[IMAGE]
+[IMAGE]257,445,798,730
+<div style="text-align: center; margin: 20px 0;">
+ [IMAGE]
+</div>
+
+29
```

</details>

### Page 048 — score=3.1
- Source: `UN-ECE-R16 source p.55 [audit_flagged_neighbor,figure_adjacent,r16_extra]`
- Chars: Docling=2592, LightOn=2887
- Section disagreement: **YES** (docling=4, lighton=3)
  - only Docling: `['1.10']`

<details><summary>Diff preview</summary>

```diff
--- docling/page_048.md
+++ lighton/page_048.md
@@ -1,4 +1,8 @@
-4.
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 6 
 
-Stopping device
+---
+
+4. Stopping device
 
@@ -6,9 +10,6 @@
 
-An outer casing formed from a steel tube;
-
-A polyurethane energy-absorber tube;
-
-A polished-steel olive-shaped knob penetrating into the absorber; and
-
-A shaft and an impact plate.
+- An outer casing formed from a steel tube;
+- A polyurethane energy-absorber tube;
+- A polished-steel olive-shaped knob penetrating into the absorber; and
+- A shaft and an impact plate.
 
@@ -18,66 +19,74 @@
 
-# Table 1
+**Table 1** 
+*Characteristics of the absorbing material* 
+*(ASTM Method D 735 unless otherwise stated)*
 
-# Characteristics of the absorbing material
+<table>
+ <thead>
+ <tr>
+ <th>Shore hardness A</th>
+ <th>95 ± 2 at 20 ± 5 °C temperature</th>
```

</details>

### Page 056 — score=3.1
- Source: `UN-ECE-R129 source p.11 [control]`
- Chars: Docling=3201, LightOn=2951
- Section disagreement: **YES** (docling=15, lighton=16)
  - only LightOn: `['2.43']`

<details><summary>Diff preview</summary>

```diff
--- docling/page_056.md
+++ lighton/page_056.md
@@ -1,6 +1,11 @@
-2.42 . " Inclined position" means a special position of the chair which allows the child to recline.
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3 
 
-2 . 43 . " Lying down/supine/prone position" means a position where at least the child's head and body excluding its limbs are on a horizontal surface when at rest in the restraint.
+---
 
-2.44. " Vehicle seat" means a structure, which may or may not be integral with the vehicle structure, complete with trim and intended to seat one adult person. In this respect:
+2.42. "Inclined position" means a special position of the chair which allows the child to recline.
+
+2.43. "Lying down/supine/prone position" means a position where at least the child's head and body excluding its limbs are on a horizontal surface when at rest in the restraint.
+
+2.44. "Vehicle seat" means a structure, which may or may not be integral with the vehicle structure, complete with trim and intended to seat one adult person. In this respect:
 
@@ -8,24 +13,28 @@
 
-2.44.2. " Vehicle bench seat" means a structure complete with trim and intended to seat more than one adult person.
+2.44.2. "Vehicle bench seat" means a structure complete with trim and intended to seat more than one adult person.
 
-2.44.3. " Vehicle front seats" means the group of seats situated foremost in the passenger compartment, i.e. having no other seat directly in front of them.
+2.44.3. "Vehicle front seats" means the group of seats situated foremost in the passenger compartment, i.e. having no other seat directly in front of them.
 
-2.44.4. "Vehicle rear seats" are fixed, forward -facing seats situated behind another group of vehicle seats.
+2.44.4. "Vehicle rear seats" are fixed, forward-facing seats situated behind another group of vehicle seats.
 
-2.45 . " Seat type" means a category of adult seats which do not differ in such essential respects as the shape, dimensions and materials of the seat structure, the types and dimensions of the seat-lock adjustment and locking systems, and the type and dimensions of the adult safety-belt anchorage on the seat, of the seat anchorage, and of the affected parts of the vehicle structure.
+2.45. "Seat type" means a category of adult seats which do not differ in such essential respects as the shape, dimensions and materials of the seat structure, the types and dimensions of the seat-lock adjustment and locking systems, and the type and dimensions of the adult safety-belt anchorage on the seat, of the seat anchorage, and of the affected parts of the vehicle structure.
 
-2.46 . " Adjustment system" means the complete device by which the vehicle seat or its parts can be adjusted to suit the physique of the seat's adult occupant; this device may, in particular, permit longitudinal displacement, and/or vertical displacement, and/or angular displacement.
+2.46. "Adjustment system" means the complete device by which the vehicle seat or its parts can be adjusted to suit the physique of the seat's adult occupant; this device may, in particular, permit longitudinal displacement, and/or vertical displacement, and/or angular displacement.
 
-2.47 . " Vehicle seat anchorage " means the system, including the affected parts of the vehicle structure, by which the adult seat as a whole is secured to the vehicle structure.
+2.47. "Vehicle seat anchorage" means the system, including the affected parts of the vehicle structure, by which the adult seat as a whole is secured to the vehicle structure.
 
-2.48 . " Displacement system" means a device enabling the adult seat or one of its parts to be displaced angularly or longitudinally, without a fixed intermediate position, to facilitate the entry and exit of passengers and the loading and unloading of objects.
+2.48. "Displacement system" means a device enabling the adult seat or one of its parts to be displaced angularly or longitudinally, without a fixed intermediate position, to facilitate the entry and exit of passengers and the loading and unloading of objects.
 
```

</details>

### Page 062 — score=2.1
- Source: `UN-ECE-R129 source p.32 [docling_figure_page,figure_adjacent]`
- Chars: Docling=2531, LightOn=2343
- Section disagreement: **no** (docling=3, lighton=3)
- LightOn figure/caption not in Docling: `(a) Minimum support-leg contact surface shall be 2,500 mm², measured as a projected surface 10 mm above the lower edge of the support-leg foot (see Figure 0(d))`

<details><summary>Diff preview</summary>

```diff
--- docling/page_062.md
+++ lighton/page_062.md
@@ -1,4 +1,7 @@
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3
+
 (b) In length by two planes parallel to the Z'-Y' plane and positioned at distances of 585 mm and 695 mm forward of the origin along the X' axis; and
 
-(c) In height by a plane parallel to the X'-Y' plane, positioned at a distance of 70 mm above the origin and measured perpendicular to the X' -Y' plane. Rigid, non-adjustable parts of the support leg shall not extend beyond a plane parallel to the X'-Y' plane, positioned at a distance of 285 mm below the origin and perpendicular to the X'-Y' plane.
+(c) In height by a plane parallel to the X'-Y' plane, positioned at a distance of 70 mm above the origin and measured perpendicular to the X'-Y' plane. Rigid, non-adjustable parts of the support leg shall not extend beyond a plane parallel to the X'-Y' plane, positioned at a distance of 285 mm below the origin and perpendicular to the X'-Y' plane.
 
@@ -6,3 +9,3 @@
 
-6.3.5.2. Support-leg foot adjustability requirements
+### 6.3.5.2. Support-leg foot adjustability requirements
 
@@ -20,3 +23,3 @@
 
-6.3.5.3. Support-leg foot dimensions
+### 6.3.5.3. Support-leg foot dimensions
 
@@ -24,6 +27,8 @@
 
-(a) Minimum support-leg contact surface shall be 2 , 500 mm 2 , measured as a projected surface 10 mm above the lower edge of the support-leg foot (see Figure 0(d));
+(a) Minimum support-leg contact surface shall be 2,500 mm², measured as a projected surface 10 mm above the lower edge of the support-leg foot (see Figure 0(d));
 
-(b) Minimum outside dimensions shall be 30 mm in the X' and Y' directions, with maximum dimensions being limited by the supportleg foot assessment volume;
+(b) Minimum outside dimensions shall be 30 mm in the X' and Y' directions, with maximum dimensions being limited by the support-leg foot assessment volume;
 
 (c) Minimum radius of the edges of the support-leg foot shall be 3.2 mm.
+
+32
```

</details>

### Page 035 — score=2.1
- Source: `UN-ECE-R16 source p.10 [control]`
- Chars: Docling=3265, LightOn=3090
- Section disagreement: **no** (docling=8, lighton=8)
- LightOn figure/caption not in Docling: `2.38. "Child restraint fixture" (CRF) means a fixture according to one out of the seven ISOFIX size classes defined in paragraph 4. of Annex 17 – Appendix 2 of `

<details><summary>Diff preview</summary>

```diff
--- docling/page_035.md
+++ lighton/page_035.md
@@ -1,27 +1,32 @@
-(c) Or a semi-universal ISOFIX rearward facing child restraint system as defined in Regulation No. 44,
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
 
-(d) Or a semi-universal ISOFIX lateral facing position child restraint system as defined in Regulation No. 44,
+(c) Or a semi-universal ISOFIX rearward facing child restraint system as defined in Regulation No. 44, 
 
-(e) Or a specific vehicle ISOFIX child restraint system as defined in Regulation No. 44.
+(d) Or a semi-universal ISOFIX lateral facing position child restraint system as defined in Regulation No. 44, 
 
-2.32. " ISOFIX anchorages system" means a system made up of two ISOFIX low anchorages, fulfilling the requirements of Regulation No. 14, and which is designed for attaching an ISOFIX child restraint system in conjunction with an anti -rotation device.
+(e) Or a specific vehicle ISOFIX child restraint system as defined in Regulation No. 44. 
 
-2.33. " ISOFIX low anchorage" means one 6 mm diameter rigid round horizontal bar, extending from vehicle or seat structure to accept and restrain an ISOFIX child restraint system with ISOFIX attachments.
+2.32. "ISOFIX anchorages system" means a system made up of two ISOFIX low anchorages, fulfilling the requirements of Regulation No. 14, and which is designed for attaching an ISOFIX child restraint system in conjunction with an anti-rotation device. 
 
-2.34. "Anti -rotation device"
+2.33. "ISOFIX low anchorage" means one 6 mm diameter rigid round horizontal bar, extending from vehicle or seat structure to accept and restrain an ISOFIX child restraint system with ISOFIX attachments. 
 
-(a) An anti-rotation device for an ISOFIX universal child restraint system
+2.34. "Anti-rotation device" 
 
-consists of the ISOFIX top-tether, (b) An anti-rotation device for an ISOFIX semi-universal child restraint system consists of a top tether, the vehicle dashboard or a support leg intended to limit the rotation of the restraint during a frontal impact,
+(a) An anti-rotation device for an ISOFIX universal child restraint system consists of the ISOFIX top-tether, 
 
-(c) For ISOFIX, universal and semi-universal, child restraint systems the vehicle seat itself does not constitute an anti -rotation device.
+(b) An anti-rotation device for an ISOFIX semi-universal child restraint system consists of a top tether, the vehicle dashboard or a support leg intended to limit the rotation of the restraint during a frontal impact, 
 
-2.35. " ISOFIX top tether anchorage" means a feature, fulfilling the requirements of Regulation No. 14, such as a bar, located in a defined zone, designed to accept an ISOFIX top tether strap connector and transfer its restraint force to the vehicle structure.
+(c) For ISOFIX, universal and semi-universal, child restraint systems the vehicle seat itself does not constitute an anti-rotation device. 
 
-2.36. A "guidance device" is intended to help the person installing the ISOFIX child restraint system by physically guiding the ISOFIX attachments on the ISOFIX child restraint into correct alignment with the ISOFIX low anchorages to facilitate engagement.
+2.35. "ISOFIX top tether anchorage" means a feature, fulfilling the requirements of Regulation No. 14, such as a bar, located in a defined zone, designed to accept an ISOFIX top tether strap connector and transfer its restraint force to the vehicle structure. 
 
-2.37. " ISOFIX marking fixture " means something that informs someone wishing to install an ISOFIX child restraint system of the ISOFIX positions in the vehicle and the position of each corresponding ISOFIX anchorages system.
+2.36. A "guidance device" is intended to help the person installing the ISOFIX child restraint system by physically guiding the ISOFIX attachments on the ISOFIX child restraint into correct alignment with the ISOFIX low anchorages to facilitate engagement. 
 
```

</details>

### Page 038 — score=2.0
- Source: `UN-ECE-R16 source p.23 [r16_extra]`
- Chars: Docling=3280, LightOn=3182
- Section disagreement: **no** (docling=17, lighton=17)
- LightOn figure/caption not in Docling: `### 7.3. Micro-slip test (see Annex 11, Figure 3 to this Regulation)`

<details><summary>Diff preview</summary>

```diff
--- docling/page_038.md
+++ lighton/page_038.md
@@ -1,20 +1,34 @@
-7.3. Micro -slip test (see Annex 11, Figure 3 to this Regulation)
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7
 
-7.3.1. The samples to be submitted to the micro-slip test shall be kept for a minimum of 24 hours in an atmosphere having a temperature of 20 + 5 °C and a relative humidity of 65 + 5 per cent. The test shall be carried out at a temperature between 15 and 30 °C.
+---
 
-7.3.2. It shall be ensured that the free section of the adjusting device points either up or down on the test bench, as in the vehicle.
+### 7.3. Micro-slip test (see Annex 11, Figure 3 to this Regulation)
 
-7.3.3. A 5 daN load shall be attached to the lower end of the section of strap. The other end shall be subjected to a back and forth motion, the total amplitude being 300 + 20 mm (see figure).
+#### 7.3.1.
+The samples to be submitted to the micro-slip test shall be kept for a minimum of 24 hours in an atmosphere having a temperature of $20 \pm 5^\circ \text{C}$ and a relative humidity of $65 \pm 5$ per cent. The test shall be carried out at a temperature between 15 and $30^\circ \text{C}$ .
 
-7.3.4. If there is a free end serving as reserve strap, it must in no way be fastened or clipped to the section under load.
+#### 7.3.2.
+It shall be ensured that the free section of the adjusting device points either up or down on the test bench, as in the vehicle.
 
-7.3.5. It shall be ensured that on the test bench the strap, in the slack position, descends in a concave curve from the adjusting device, as in the vehicle. The 5 daN load applied on the test bench shall be guided vertically in such a way as to prevent the load swaying and the belt twisting. The attachment shall be fixed to the 5 daN load as in the vehicle.
+#### 7.3.3.
+A 5 daN load shall be attached to the lower end of the section of strap. The other end shall be subjected to a back and forth motion, the total amplitude being $300 \pm 20$ mm (see figure).
 
-7.3.6. Before the actual start of the test, a series of 20 cycles shall be completed so that the selfftightening system settles properly.
+#### 7.3.4.
+If there is a free end serving as reserve strap, it must in no way be fastened or clipped to the section under load.
 
-7.3.7. 1,000 cycles shall be completed at a frequency of 0.5 cycles per second, the total amplitude being 300 + 20 mm. The 5 daN load shall be applied only during the time corresponding to a shift of 100 + 20 mm for each half period.
+#### 7.3.5.
+It shall be ensured that on the test bench the strap, in the slack position, descends in a concave curve from the adjusting device, as in the vehicle. The 5 daN load applied on the test bench shall be guided vertically in such a way as to prevent the load swaying and the belt twisting. The attachment shall be fixed to the 5 daN load as in the vehicle.
 
-7.4. Conditioning of straps and breaking-strength test (static)
+#### 7.3.6.
+Before the actual start of the test, a series of 20 cycles shall be completed so that the self-tightening system settles properly.
 
-# 7.4.1. Conditioning of straps for the breaking-strength test
+#### 7.3.7.
+1,000 cycles shall be completed at a frequency of 0.5 cycles per second, the total amplitude being $300 \pm 20$ mm. The 5 daN load shall be applied only during the time corresponding to a shift of $100 \pm 20$ mm for each half period.
```

</details>

### Page 040 — score=2.0
- Source: `UN-ECE-R16 source p.25 [r16_extra]`
- Chars: Docling=3175, LightOn=3101
- Section disagreement: **no** (docling=14, lighton=14)
- LightOn figure/caption not in Docling: `The total back and forth motion shall be $300 \pm 20$ mm but the 5 daN load shall only be applied during a shift of $100 \pm 20$ mm for each half period (see An`

<details><summary>Diff preview</summary>

```diff
--- docling/page_040.md
+++ lighton/page_040.md
@@ -1,2 +1,7 @@
-7.4.1.6.4.2. Procedure 2: for cases where the strap changes direction in passing through a rigid part.
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+
+---
+
+### 7.4.1.6.4.2. Procedure 2: for cases where the strap changes direction in passing through a rigid part.
 
@@ -8,28 +13,56 @@
 
-7.4.1.6.4.3. Procedure 3: for cases where the strap is fixed to a rigid part by sewing or similar means.
+---
 
-The total back and forth motion shall be 300 + 20 mm but the 5 daN load shall only be applied during a shift of 100 + 20 mm for each half period (see Annex 11, Figure 3, to this Regulation).
+### 7.4.1.6.4.3. Procedure 3: for cases where the strap is fixed to a rigid part by sewing or similar means.
 
-7.4.2. Test of breaking strength of strap (static test)
+The total back and forth motion shall be $300 \pm 20$ mm but the 5 daN load shall only be applied during a shift of $100 \pm 20$ mm for each half period (see Annex 11, Figure 3, to this Regulation).
 
-7.4.2.1. The test shall be carried out each time on two new samples of strap, of sufficient length, conditioned in conformity with the provisions of paragraph 7.4.1.
+---
 
-7.4.2.2. Each strap shall be gripped between the clamps of a tensile-testing machine. The clamps shall be so designed as to avoid breakage of the strap at or near them. The speed of traverse shall be about 100 mm/min. The free length of the specimen between the clamps of the machine at the start of the test shall be 200 mm + 40 mm.
+### 7.4.2. Test of breaking strength of strap (static test)
 
-7.4.2.3. The tension shall be increased until the strap breaks, and the breaking load shall be noted.
+#### 7.4.2.1.
 
-7.4.2.4. If the strap slips or breaks at or within 10 mm of either of the clamps the test shall be invalid and a new test shall be carried out on another specimen.
+The test shall be carried out each time on two new samples of strap, of sufficient length, conditioned in conformity with the provisions of paragraph 7.4.1.
 
-7.4.3. Width under load
+#### 7.4.2.2.
 
-7.4.3.1. The test shall be carried out each time on two new samples of strap, of sufficient length conditioned in conformity with the provisions of paragraph 7.4.1.
+Each strap shall be gripped between the clamps of a tensile-testing machine. The clamps shall be so designed as to avoid breakage of the strap at or near them. The speed of traverse shall be about 100 mm/min. The free length of the specimen between the clamps of the machine at the start of the test shall be $200 \text{ mm} \pm 40 \text{ mm}$ .
 
```

</details>

### Page 047 — score=2.0
- Source: `UN-ECE-R16 source p.54 [annex,audit_flagged,audit_flagged_neighbor,figure_adjacent,r16_extra]`
- Chars: Docling=2816, LightOn=2743
- Section disagreement: **no** (docling=5, lighton=5)
- LightOn figure/caption not in Docling: `The anchorages shall be positioned as shown in Figure 1. The marks which correspond to the arrangement of the anchorages show where the ends of the belt are to `

<details><summary>Diff preview</summary>

```diff
--- docling/page_047.md
+++ lighton/page_047.md
@@ -1,12 +1,16 @@
-# Annex 6
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 6
 
-# Description of trolley, seat, anchorages and stopping device
+---
 
-1.
+## Annex 6
 
-400 + 20 kg. For tests on restraint systems the trolley with the attached vehicle the trolley and vehicle structure may be increased by increments of 200 kg. In no case shall the total mass differ from the nominal value by more than + 40 kg.
+### Description of trolley, seat, anchorages and stopping device
 
-Trolley For tests on safety-belts the trolley, carrying the seat only, shall have a mass of structure shall have a mass of 800 kg. However, if necessary, the total mass of
+#### 1. Trolley
 
-Seat
+For tests on safety-belts the trolley, carrying the seat only, shall have a mass of $400 \pm 20$ kg. For tests on restraint systems the trolley with the attached vehicle structure shall have a mass of 800 kg. However, if necessary, the total mass of the trolley and vehicle structure may be increased by increments of 200 kg. In no case shall the total mass differ from the nominal value by more than $\pm 40$ kg.
+
+#### 2. Seat
 
@@ -14,16 +18,24 @@
 
-Anchorages
+#### 3. Anchorages
 
-3.1. In the case of a belt equipped with a belt adjustment device for height as defined in paragraph 29.6. of this Regulation, this device shall be secured either to a rigid frame, or to a part of the vehicle on which it is normally mounted which shall be securely fixed on the test trolley.
+##### 3.1.
 
-3.2. The anchorages shall be positioned as shown in Figure 1. The marks which correspond to the arrangement of the anchorages show where the ends of the belt are to be connected to the trolley or to the load transducer, as the case may be. The anchorages for normal use are the points A, B and K if the strap length between the upper edge of the buckle and the hole for attachment of the strap support is not more than 250 mm. Otherwise, the points A1 and B1 shall be used. The tolerance on the position of the anchorage points is such that each anchorage point shall be situated at most at 50 mm from corresponding points A, B and K indicated in Figure 1 or A1, B1 and K, as the case may be.
+In the case of a belt equipped with a belt adjustment device for height as defined in paragraph 29.6. of this Regulation, this device shall be secured either to a rigid frame, or to a part of the vehicle on which it is normally mounted which shall be securely fixed on the test trolley.
 
-3.3. The structure carrying the anchorages shall be rigid. The upper anchorage must not be displaced by more than 0.2 mm in the longitudinal direction when a load of 98 daN is applied to it in that direction. The trolley shall be so constructed that no permanent deformation shall occur in the parts bearing the anchorages during the test.
+##### 3.2.
 
-3.4. If a fourth anchorage is necessary to attach the retractor, this anchorage:
```

</details>

### Page 030 — score=2.0
- Source: `UN-ECE-R95 source p.69 [audit_flagged_neighbor,docling_figure_page,figure_adjacent]`
- Chars: Docling=404, LightOn=454
- Section disagreement: **no** (docling=3, lighton=3)
- LightOn figure/caption not in Docling: `### 3.2. Body block impactor (Figure 3)`

<details><summary>Diff preview</summary>

```diff
--- docling/page_030.md
+++ lighton/page_030.md
@@ -1,5 +1,17 @@
-3.2. Body block impactor (Figure 3)
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 8 
 
-3.2.1. This apparatus consists of a fully guided linear impactor, rigid, with a mass of 30 kg. Its dimensions and transversal section is presented in Figure 3.
+---
 
-3.2.2. The body block shall be fitted with two accelerometers and a speed-measuring device, all capable of measuring values in the impact direction.
+### 3.2. Body block impactor (Figure 3)
+
+#### 3.2.1.
+This apparatus consists of a fully guided linear impactor, rigid, with a mass of 30 kg. Its dimensions and transversal section is presented in Figure 3.
+
+#### 3.2.2.
+The body block shall be fitted with two accelerometers and a speed-measuring device, all capable of measuring values in the impact direction.
+
+---
+
+69
```

</details>

### Page 066 — score=2.0
- Source: `UN-ECE-R129 source p.53 [docling_figure_page,figure_adjacent]`
- Chars: Docling=2161, LightOn=2172
- Section disagreement: **no** (docling=1, lighton=1)
- LightOn figure/caption not in Docling: `Fit load cell 1 to the outboard position as shown Figure 1. Install the Enhanced Child Restraint System in the correct position. If a lock-off device is fitted `

<details><summary>Diff preview</summary>

```diff
--- docling/page_066.md
+++ lighton/page_066.md
@@ -1,14 +1,18 @@
-[IMAGE]
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3
 
-Figure 3 Load cell positions
+## Figure 3 
+**Load cell positions**
 
-Fit load cell 1 to the outboard position as shown Figure 1. Install the Enhanced Child Restraint System in the correct position. If a lock-off device is fitted to the Enhanced Child Restraint System and acts upon the diagonal belt, place load cell 2 at a convenient position behind the Enhanced Child Restraint System between the lock-off device and the buckle as shown above. If no lock -off device is fitted or if the lock -off device is fitted at the buckle, position the load cell at a convenient position between the pillar loop and the Enhanced Child Restraint System.
+[IMAGE]197,152,675,345
 
-Adjust the lap portion of the reference belt to achieve a tension load of 50 N ± 5 N at load cell 1. Make a chalk mark on the webbing where it passes through the simulated buckle.
+Fit load cell 1 to the outboard position as shown Figure 1. Install the Enhanced Child Restraint System in the correct position. If a lock-off device is fitted to the Enhanced Child Restraint System and acts upon the diagonal belt, place load cell 2 at a convenient position behind the Enhanced Child Restraint System between the lock-off device and the buckle as shown above. If no lock-off device is fitted or if the lock-off device is fitted at the buckle, position the load cell at a convenient position between the pillar loop and the Enhanced Child Restraint System.
 
-While maintaining the belt at this position, adjust the diagonal to achieve a tension of 50 N ± 5 N at load cell 2 by either locking the webbing at the Enhanced Child Restraint System webbing locker or by pulling the belt between the belt clamping mechanism and the standard retractor. If the tension in load cell 2 is achieved by pulling the belt between the clamping mechanism and the retractor, the clamping mechanism shall now be locked.
+Adjust the lap portion of the reference belt to achieve a tension load of $50 \, \text{N} \pm 5 \, \text{N}$ at load cell 1. Make a chalk mark on the webbing where it passes through the simulated buckle.
 
-Extract all webbing from the retractor spool and rewind the excess webbing keeping a tension of 4 ± 3 N in the belt between the retractor and the pillar loop. The spool shall be locked before the dynamic test. Conduct the dynamic crash test.
+While maintaining the belt at this position, adjust the diagonal to achieve a tension of $50 \, \text{N} \pm 5 \, \text{N}$ at load cell 2 by either locking the webbing at the Enhanced Child Restraint System webbing locker or by pulling the belt between the belt clamping mechanism and the standard retractor. If the tension in load cell 2 is achieved by pulling the belt between the clamping mechanism and the retractor, the clamping mechanism shall now be locked.
 
-# 7.1.3.5.2.3. After installation
+Extract all webbing from the retractor spool and rewind the excess webbing keeping a tension of $4 \pm 3 \, \text{N}$ in the belt between the retractor and the pillar loop. The spool shall be locked before the dynamic test. Conduct the dynamic crash test.
+
+### 7.1.3.5.2.3. After installation
 
@@ -23 +27,3 @@
 Legs shall be positioned parallel to one another or at least symmetrically.
+
+53
```

</details>

### Page 068 — score=2.0
- Source: `UN-ECE-R129 source p.72 [annex]`
- Chars: Docling=6312, LightOn=1553
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_068.md
+++ lighton/page_068.md
@@ -1,19 +1,24 @@
-| 5. | If applicable, name of his representative ................................ .......................... | If applicable, name of his representative ................................ .......................... |
-|-------|------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
-| 6. | Address ................................ ................................ ................................ ............. | Address ................................ ................................ ................................ ............. |
-| 7. | Submitted for approval on ................................ ................................ ................ | Submitted for approval on ................................ ................................ ................ |
-| 8. | Technical Service conducting approval tests ................................ .................... | Technical Service conducting approval tests ................................ .................... |
-| 9. | Type of device: deceleration/acceleration 2 | Type of device: deceleration/acceleration 2 |
-| 10. | Date of test report issued by that Service ................................ .......................... | Date of test report issued by that Service ................................ .......................... |
-| 11. | Number of test report issued by that Service ................................ .................... | Number of test report issued by that Service ................................ .................... |
-| 12. | Approval granted/extended/refused/withdrawn 2 for size range x to x for i Size specific vehicle or for use as a "special needs restraint", position in vehicle | Approval granted/extended/refused/withdrawn 2 for size range x to x for i Size specific vehicle or for use as a "special needs restraint", position in vehicle |
-| 13. | Position and nature of the marking ................................ ................................ ... | Position and nature of the marking ................................ ................................ ... |
-| 14. | Place ................................ ................................ ................................ ................. | Place ................................ ................................ ................................ ................. |
-| 15. | Date ................................ ................................ ................................ ................... | Date ................................ ................................ ................................ ................... |
-| 16. | Signature ................................ ................................ ................................ ........... | Signature ................................ ................................ ................................ ........... |
-| 17. | The following documents, bearing the approval number shown above, are attached to this communication: | The following documents, bearing the approval number shown above, are attached to this communication: |
-| | (a) | Drawings, diagrams and plans of the child restraint, including any retractor, chair assembly, impact shield fitted; |
-| | (b) | Drawings, diagrams and plans of the vehicle structure and the seat structure, as well as of the adjustment system and the attachments, including any energy absorber fitted; |
-| | (c) | Photographs of the child restraint and/or vehicle structure and seat structure; |
-| | (d) | Instructions for fitting and use; |
-| | (e) | List of vehicle models for which the restraint is intended. |
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3 
+Annex 1 
+
+5. If applicable, name of his representative ……………………………………………………………… 
+6. Address ………………………………………………………………………………………………………… 
+7. Submitted for approval on ………………………………………………………………………………… 
+8. Technical Service conducting approval tests ………………………………………………………… 
+9. Type of device: deceleration/acceleration² 
+10. Date of test report issued by that Service …………………………………………………………… 
+11. Number of test report issued by that Service ……………………………………………………… 
+12. Approval granted/extended/refused/withdrawn² for size range x to x for i-Size specific vehicle or for use as a "special needs restraint", position in vehicle 
+13. Position and nature of the marking …………………………………………………………………… 
+14. Place ………………………………………………………………………………………………………… 
+15. Date ………………………………………………………………………………………………………… 
+16. Signature …………………………………………………………………………………………………… 
+17. The following documents, bearing the approval number shown above, are attached to this communication: 
+ (a) Drawings, diagrams and plans of the child restraint, including any retractor, chair assembly, impact shield fitted; 
```

</details>

### Page 041 — score=0.9
- Source: `UN-ECE-R16 source p.42 [annex]`
- Chars: Docling=3625, LightOn=1800
- Section disagreement: **no** (docling=11, lighton=11)

<details><summary>Diff preview</summary>

```diff
--- docling/page_041.md
+++ lighton/page_041.md
@@ -1,6 +1,10 @@
-# Annex 1A
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 1A 
 
-# Communication
+---
 
-[IMAGE]
+## Annex 1A
+
+### Communication
 
@@ -8,15 +12,15 @@
 
-1
+[IMAGE]197,224,360,337
 
-concerning 2 :
+issued by: 
+Name of administration: 
+___________________________ 
+___________________________
 
-Approval granted
-
-Approval extended
-
-Approval refused
-
-Approval withdrawn
-
-Production definitively discontinued
+concerning²: 
+Approval granted 
+Approval extended 
+Approval refused 
```

</details>

### Page 007 — score=0.7
- Source: `UN-ECE-R94 source p.20 [annex]`
- Chars: Docling=3143, LightOn=1696
- Section disagreement: **no** (docling=3, lighton=3)

<details><summary>Diff preview</summary>

```diff
--- docling/page_007.md
+++ lighton/page_007.md
@@ -1,6 +1,10 @@
-# Annex 1
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4 
+Annex 1 
 
-# Communication
+---
 
-[IMAGE]
+## Annex 1
+
+### Communication
 
@@ -8,15 +12,15 @@
 
-1
+[IMAGE]200,213,340,325
 
-Concerning 2
+issued by : 
+Name of administration: 
+___________________________ 
+___________________________
 
-: Approval granted
-
-Approval extended
-
-Approval refused
-
-Approval withdrawn
-
-Production definitively discontinued
+Concerning²: 
+Approval granted 
+Approval extended 
+Approval refused 
```

</details>

### Page 020 — score=0.5
- Source: `UN-ECE-R95 source p.20 [audit_flagged,audit_flagged_neighbor,figure_adjacent]`
- Chars: Docling=2983, LightOn=1907
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_020.md
+++ lighton/page_020.md
@@ -1,6 +1,10 @@
-# Annex 1
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 1
 
-# Communication
+---
 
-[IMAGE]
+## Annex 1
+
+### Communication
 
@@ -8,10 +12,13 @@
 
-Concerning:
+[IMAGE]197,216,340,333
 
-2 Approval granted
+issued by: 
+Name of administration: 
+.................................................... 
+....................................................
 
-Approval extended
-
-Approval refused Approval withdrawn
-
+Concerning:² Approval granted 
+Approval extended 
+Approval refused 
+Approval withdrawn 
 Production definitively discontinued
@@ -20,108 +27,21 @@
 
-Approval No.
+Approval No. ........................................ Extension No. ........................................ 
```

</details>

### Page 067 — score=0.4
- Source: `UN-ECE-R129 source p.71 [annex]`
- Chars: Docling=2169, LightOn=1323
- Section disagreement: **no** (docling=4, lighton=4)

<details><summary>Diff preview</summary>

```diff
--- docling/page_067.md
+++ lighton/page_067.md
@@ -1,103 +1,54 @@
-# Annex 1
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3 
+Annex 1
 
-# Communication
+---
 
-[IMAGE]
+## Annex 1
 
-(Maximum format: A4 (210 x 297 mm)
+### Communication
 
-1
+(Maximum format: A4 (210 x 297 mm))
 
-issued by:
+[IMAGE]198,213,385,353
 
-Name of administration:
+issued by: 
+Name of administration: 
+___________________________ 
+___________________________ 
+___________________________
 
-................................
-
-.......
-
-................................
-
-.......
-
-................................
-
```

</details>

### Page 064 — score=0.3
- Source: `UN-ECE-R129 source p.51 [docling_figure_page,figure_adjacent]`
- Chars: Docling=3963, LightOn=3310
- Section disagreement: **no** (docling=7, lighton=7)

<details><summary>Diff preview</summary>

```diff
--- docling/page_064.md
+++ lighton/page_064.md
@@ -1,21 +1,76 @@
-| | | Frontal impact | Frontal impact | Frontal impact | Rear impact | Rear impact | Rear impact | Lateral impact | Lateral impact |
-|---------------------------|------------------|-------------------|-------------------|---------------------------------------|----------------|-----------------|---------------------------------------|--------------------------------|---------------------------------------------------------|
-| Test | Restraint | Speed km/h | Test pulse No. | Stopping distance during test (mm) | Speed km/h | Test pulse No. | Stopping distance during test (mm) | Relative door/bench velocity | Stopping distance during test (mm) Maximum intrusion |
-| Trolley with test bench | Forward Facing | 50 + 0 - 2 | 1 | 650 ± 50 | NA | NA | NA | 3 | 250 ± 50 |
-| Trolley with test bench | Rearward Facing | 50 + 0 - 2 | 1 | 650 ± 50 | 30 + 2 - 0 | 2 | 275 ± 25 | 3 | 250 ± 50 |
-| Trolley with test bench | Lateral Facing | 50 + 0 - 2 | 1 | 650 ± 50 | 30 + 2 - 0 | 2 | 275 ± 25 | 3 | 250 ± 50 |
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3
 
-Table 6
+7.1.3.3.5. The front seats, if adjustable for inclination, shall be locked as specified by the manufacturer or, in the absence of any specification, at an actual seat-back angle as near as possible to 25°.
 
-# Legend:
+7.1.3.3.6. After impact, the child restraint shall be inspected visually, without opening the buckle, to determine whether there has been any failure or breakage.
 
-Test pulse No. 1 – As prescribed in Annex 7 / Appendix 1 – Frontal impact.
+7.1.3.4. The conditions for dynamic test are summarized in Table 6:
 
-Test pulse No. 2 – As prescribed in Annex 7 / Appendix 2 – Rear impact.
+**Table 6**
 
-Test velocity corridor curve No. 3 – As prescribed in Annex 7 / Appendix 3 – Lateral impact
+<table>
+ <thead>
+ <tr>
+ <th rowspan="2">Test</th>
+ <th rowspan="2">Restraint</th>
+ <th colspan="3">Frontal impact</th>
+ <th colspan="3">Rear impact</th>
+ <th colspan="2">Lateral impact</th>
+ </tr>
+ <tr>
+ <th>Speed km/h</th>
+ <th>Test pulse No.</th>
+ <th>Stopping distance during test (mm)</th>
+ <th>Speed km/h</th>
+ <th>Test pulse No.</th>
```

</details>

### Page 042 — score=0.3
- Source: `UN-ECE-R16 source p.45 [audit_flagged_neighbor,figure_adjacent,r16_extra]`
- Chars: Docling=1658, LightOn=1043
- Section disagreement: **no** (docling=1, lighton=1)

<details><summary>Diff preview</summary>

```diff
--- docling/page_042.md
+++ lighton/page_042.md
@@ -1,45 +1,17 @@
-11 . Type of device: deceleration/acceleration 2
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 1B 
 
-12 . Approval granted/refused/extended/withdrawn 2 for general use/for use in a particular vehicle or in particular types of vehicles 2 , 4
+11. Type of device: deceleration/acceleration² 
+12. Approval granted/refused/extended/withdrawn² for general use/for use in a particular vehicle or in particular types of vehicles²,⁴ 
+13. Position and nature of the marking ……………………………………………………………… 
+14. Place …………………………………………………………………………………………………… 
+15. Date …………………………………………………………………………………………………… 
+16. Signature ……………………………………………………………………………………………… 
+17. Annexed to this communication is a list of documents in the approval file deposited at the administration services having delivered the approval and which can be obtained upon request. 
 
-13 . Position and nature of the marking
+---
 
-................................
+⁴ If a safety-belt is approved following the provisions of paragraph 6.4.1.3.3. of this Regulation, this safety-belt shall only be installed in an outboard front seating position protected by an airbag in front of it, under the condition that the vehicle concerned is approved to Regulation No. 94, 01 series of amendments or its later version in force. 
 
-................................
-
-..............
-
-14 . Place
-
-................................
-
-................................
-
-................................
-
-..............................
-
-15 . Date
-
-................................
```

</details>

### Page 008 — score=0.2
- Source: `UN-ECE-R94 source p.21 [annex]`
- Chars: Docling=2473, LightOn=2099
- Section disagreement: **no** (docling=4, lighton=4)

<details><summary>Diff preview</summary>

```diff
--- docling/page_008.md
+++ lighton/page_008.md
@@ -1,39 +1,26 @@
-Drive: front-wheel/rear-wheel 2
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4 
+Annex 1 
 
-Mass of the Vehicle
+7. Drive: front-wheel/rear-wheel² 
+8. Mass of the Vehicle 
+8.1. Mass of vehicle submitted for testing: 
+Front axle: ........................................................................................................ 
+Rear axle: ......................................................................................................... 
+Total: .............................................................................................................. 
+8.2. Where paragraph 5.3.1. or 5.3.2. applies: 
+Total permissible mass ..................................................................................... 
+Proof of compliance with UN Regulation 137 (i.e. type approval number or test report): 
+9. Vehicle submitted for approval on .................................................................. 
+10. Technical Service responsible for conducting approval tests ........................... 
+11. Date of report issued by that Service ............................................................ 
+12. Number of report issued by that Service ....................................................... 
+13. Approval granted/refused/extended/withdrawn² 
+14. Position of approval mark on vehicle ............................................................ 
+15. Place ............................................................................................................ 
+16. Date ............................................................................................................. 
+17. Signature ...................................................................................................... 
+18. The following documents, bearing the approval number shown above, are annexed to this communication: ........................................................................................................ 
+(Photographs and/or diagrams and drawings permitting the basic identification of the type(s) of vehicle and its possible variants which are covered by the approval) 
 
-8.1. Mass of vehicle submitted for testing:
-
-Front axle: ....................................................................................................................
-
-Rear axle: .....................................................................................................................
-
-Total: .............................................................................................................................
-
-8.2. Where paragraph 5.3.1. or 5.3.2. applies:
-
```

</details>

### Page 053 — score=0.1
- Source: `UN-ECE-R16 source p.60 [r16_extra]`
- Chars: Docling=1839, LightOn=1546
- Section disagreement: **no** (docling=4, lighton=4)

<details><summary>Diff preview</summary>

```diff
--- docling/page_053.md
+++ lighton/page_053.md
@@ -1,8 +1,14 @@
-# Annex 7
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 7 
 
-# Description of manikin
+---
 
-Specifications of the manikin
+## Annex 7
 
-1.1. General
+### Description of manikin
+
+#### 1. Specifications of the manikin
+
+##### 1.1. General
 
@@ -10,29 +16,16 @@
 
-Figure 1 Side view of head, neck and torso;
+- Figure 1: Side view of head, neck and torso;
+- Figure 2: Front view of head, neck and torso;
+- Figure 3: Side view of hip, thighs and lower leg;
+- Figure 4: Front view of hip, thighs and lower leg;
+- Figure 5: Principal dimensions;
+- Figure 6: Manikin in sitting position, showing:
+ - Location of the centre of gravity;
+ - Location of points at which displacement shall be measured; and shoulder height.
+- Table 1: References, names, materials and principal dimensions of components of the manikin; and
+- Table 2: Masses of head, neck, torso, thighs and lower leg.
 
-Figure 2 Front view of head, neck and torso;
+##### 1.2. Description of the manikin
 
-Figure 3 Side view of hip, thighs and lower leg;
-
```

</details>

### Page 011 — score=0.1
- Source: `UN-ECE-R94 source p.44 [docling_figure_page,figure_adjacent]`
- Chars: Docling=1520, LightOn=1812
- Section disagreement: **no** (docling=1, lighton=1)

<details><summary>Diff preview</summary>

```diff
--- docling/page_011.md
+++ lighton/page_011.md
@@ -1,4 +1,12 @@
-# 5. Presentation of results
+E/ECE/324/Rev.1/Add.93/Rev.4
 
-The results should be presented on A4 size paper (ISO/R 216). Results presented as diagrams should have axes scaled with a measurement unit corresponding to a suitable multiple of the chosen unit (for example, 1, 2, 5, 10, 20 millimetres). SI units shall be used, except for vehicle velocity, where km/h may be used, and for accelerations due to impact where g, with g = 9.8 m/s 2 , may be used.
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4
+
+Annex 8
+
+---
+
+5. Presentation of results
+
+The results should be presented on A4 size paper (ISO/R 216). Results presented as diagrams should have axes scaled with a measurement unit corresponding to a suitable multiple of the chosen unit (for example, 1, 2, 5, 10, 20 millimetres). SI units shall be used, except for vehicle velocity, where km/h may be used, and for accelerations due to impact where g, with $g = 9.8 \, \text{m/s}^2$ , may be used.
 
@@ -6,14 +14,77 @@
 
-# Frequency response curve
+**Frequency response curve**
 
-[IMAGE]
+[IMAGE]203,253,763,570
 
-| | | | | N Logarithmic scale | N Logarithmic scale | N Logarithmic scale |
-|-------|-------|-------|-------|------------------------|------------------------|------------------------|
-| CFC | FL | FH | FN | a  | 0.5 dB | |
-| | | | Hz | | b + 0.5; -1 dB | |
-| | Hz | Hz | | | c + 0.5; -4 dB | |
-| 1,000 | < 0.1 | 1,000 | 1,650 | | d - 9 dB/octave | |
-| 600 | < 0.1 | 600 | 1,000 | | e - 24 dB/octave | |
-| 180 | < 0.1 | 180 | 300 | f |  | |
-| 60 | < 0.1 | 60 | 100 | g - 30 | | |
+<table>
+ <thead>
+ <tr>
+ <th>CFC</th>
+ <th> $F_L$ Hz</th>
+ <th> $F_H$ Hz</th>
```

</details>

### Page 005 — score=0.1
- Source: `UN-ECE-R94 source p.13 [figure_adjacent,r94_figure3_area]`
- Chars: Docling=3512, LightOn=3247
- Section disagreement: **no** (docling=19, lighton=19)

<details><summary>Diff preview</summary>

```diff
--- docling/page_005.md
+++ lighton/page_005.md
@@ -1 +1,4 @@
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4 
+
 5.2.1.8. The tibia index (TI), measured at the top and bottom of each tibia, shall not exceed 1,3 at either location;
@@ -10,3 +13,3 @@
 
-5.2.3.1.1. If testing in accordance with Annex 3, paragraph 1.4.3.5.2.1., the manufacturer shall in addition demonstrate to the satisfaction of the Technical Service (e.g. manufacturer's in-house data) that, in the absence of the system or when the system is de-activated, no door will open in case of the impact.
+5.2.3.1.1. If testing in accordance with Annex 3, paragraph 1.4.3.5.2.1., the manufacturer shall in addition demonstrate to the satisfaction of the Technical Service (e.g. manufacturer’s in-house data) that, in the absence of the system or when the system is de-activated, no door will open in case of the impact.
 
@@ -20,3 +23,3 @@
 
-5.2.4.2.1. If testing in accordance with Annex 3, paragraph 1.4.3.5.2.1., the manufacturer shall in addition demonstrate to the satisfaction of the Technical Service (e.g. manufacturer's in-house data) that, in the absence of the system or when the system is de-activated, no locking of the side doors shall occur during the impact.
+5.2.4.2.1. If testing in accordance with Annex 3, paragraph 1.4.3.5.2.1., the manufacturer shall in addition demonstrate to the satisfaction of the Technical Service (e.g. manufacturer’s in-house data) that, in the absence of the system or when the system is de-activated, no locking of the side doors shall occur during the impact.
 
@@ -26,5 +29,3 @@
 
-5.2.5.1. To open at least one door per row of seats. Where there is no such door, it shall be possible to allow the evacuation of all the occupants by activating the displacement system of seats, if necessary. This is not applicable to convertibles where the top can be easily opened to allow the evacuation of the occupants.
-
-This shall be assessed for all configurations or worst-case configuration for the number of doors on each side of the vehicle and for both left-hand drive and right-hand drive vehicles, when applicable.
+5.2.5.1. To open at least one door per row of seats. Where there is no such door, it shall be possible to allow the evacuation of all the occupants by activating the displacement system of seats, if necessary. This is not applicable to convertibles where the top can be easily opened to allow the evacuation of the occupants. This shall be assessed for all configurations or worst-case configuration for the number of doors on each side of the vehicle and for both left-hand drive and right-hand drive vehicles, when applicable.
 
@@ -35 +36,3 @@
 5.2.6. In the case of a vehicle propelled by liquid fuel, no more than slight leakage of liquid from the fuel feed installation shall occur on collision.
+
+13
```

</details>

### Page 016 — score=0.1
- Source: `UN-ECE-R95 source p.8 [control]`
- Chars: Docling=3150, LightOn=2896
- Section disagreement: **no** (docling=18, lighton=18)

<details><summary>Diff preview</summary>

```diff
--- docling/page_016.md
+++ lighton/page_016.md
@@ -1,14 +1,19 @@
-2.35. " Automatically activated door locking system" means a system that locks the doors automatically at a pre-set speed or under any other condition as defined by the manufacturer.
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3
 
-2.36. " Latched" means any coupling condition of the door latch system, where the latch is in a fully latched position, a secondary latched position, or in between a fully latched position and a secondary latched position.
+---
 
-2.37. " Latch" is a device employed to maintain the door in a closed position relative to the vehicle body with provisions for deliberate release (or operation).
+2.35. "Automatically activated door locking system" means a system that locks the doors automatically at a pre-set speed or under any other condition as defined by the manufacturer.
 
-2.38. " Fully latched position" is the coupling condition of the latch that retains the door in a completely closed position.
+2.36. "Latched" means any coupling condition of the door latch system, where the latch is in a fully latched position, a secondary latched position, or in between a fully latched position and a secondary latched position.
 
-2.39. " Secondary latched position" refers to the coupling condition of the latch that retains the door in a partially closed position.
+2.37. "Latch" is a device employed to maintain the door in a closed position relative to the vehicle body with provisions for deliberate release (or operation).
 
-2.40. " Displacement system" means a device by which the seat or one of its parts can be displaced and/or rotated, without a fixed intermediate position, to permit easy access of occupants to and from the space behind the seat concerned.
+2.38. "Fully latched position" is the coupling condition of the latch that retains the door in a completely closed position.
 
-# 3. Application for approval
+2.39. "Secondary latched position" refers to the coupling condition of the latch that retains the door in a partially closed position.
+
+2.40. "Displacement system" means a device by which the seat or one of its parts can be displaced and/or rotated, without a fixed intermediate position, to permit easy access of occupants to and from the space behind the seat concerned.
+
+## 3. Application for approval
 
@@ -18,3 +23,3 @@
 
-3.2.1. A detailed description of the vehicle type with respect to its structure, dimensions, lines and constitutent materials;
+3.2.1. A detailed description of the vehicle type with respect to its structure, dimensions, lines and constituent materials;
 
@@ -22,3 +27,3 @@
 
-3.2.3. Particulars of the vehicle's mass as defined by paragraph 2.11. of this Regulation;
+3.2.3. Particulars of the vehicle’s mass as defined by paragraph 2.11. of this Regulation;
 
@@ -35 +40,3 @@
```

</details>

### Page 034 — score=0.1
- Source: `UN-ECE-R16 source p.6 [control]`
- Chars: Docling=2917, LightOn=2663
- Section disagreement: **no** (docling=15, lighton=15)

<details><summary>Diff preview</summary>

```diff
--- docling/page_034.md
+++ lighton/page_034.md
@@ -1,4 +1,7 @@
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+
 A belt which is essentially a combination of a lap strap and a diagonal strap.
 
-2.1.4. S -type belt
+### 2.1.4. S-type belt
 
@@ -6,7 +9,7 @@
 
-2.1.5. Harness belt
+### 2.1.5. Harness belt
 
-A S -type belt arrangement comprising a lap belt and shoulder straps; a harness belt may be provided with an additional crotch strap assembly.
+A S-type belt arrangement comprising a lap belt and shoulder straps; a harness belt may be provided with an additional crotch strap assembly.
 
-2.2. Belt type
+### 2.2. Belt type
 
@@ -14,9 +17,9 @@
 
-2.2.1. Rigid parts (buckle, attachments, retractor, etc.);
+#### 2.2.1. Rigid parts (buckle, attachments, retractor, etc.);
 
-2.2.2. The material, weave, dimensions and colour of the straps; or
+#### 2.2.2. The material, weave, dimensions and colour of the straps; or
 
-2.2.3. The geometry of the belt assembly.
+#### 2.2.3. The geometry of the belt assembly.
 
-# 2.3. Strap
+### 2.3. Strap
 
@@ -24,3 +27,3 @@
 
-2.4. Buckle
+### 2.4. Buckle
```

</details>

### Page 001 — score=0.1
- Source: `UN-ECE-R94 source p.8 [control]`
- Chars: Docling=3440, LightOn=3212
- Section disagreement: **no** (docling=14, lighton=14)

<details><summary>Diff preview</summary>

```diff
--- docling/page_001.md
+++ lighton/page_001.md
@@ -1 +1,4 @@
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4 
+
 Where electric circuits are galvanically connected to each other and fulfil the specific voltage condition, only the components or parts of the electric circuit that operate on high voltage are classified as high voltage bus.
@@ -6,3 +9,3 @@
 
-2.32. "Open type traction battery" means a type of battery requiring filling with liquid and generating hydrogen gas that is released to the atmosphere ..
+2.32. "Open type traction battery" means a type of battery requiring filling with liquid and generating hydrogen gas that is released to the atmosphere.
 
@@ -20,3 +23,3 @@
 
-2.39. "Normal operating conditions" includes operating modes and conditions that can reasonably be encountered during typical operation of the vehicle including driving at legally posted speeds, parking and standing in traffic, as well as, charging using chargers that are compatible with the specific charging ports installed on the vehicle. It does not include, conditions where the vehicle is damaged, either by a crash, road debris or vandalization, subjected to fire or water submersion, or in a state where service and or maintenance is needed or being performed.
+2.39. "Normal operating conditions" includes operating modes and conditions that can reasonably be encountered during typical operation of the vehicle including driving at legally posted speeds, parking and standing in traffic, as well as, charging using chargers that are compatible with the specific charging ports installed on the vehicle. It does not include, conditions where the vehicle is damaged, either by a crash, road debris or vandalism, subjected to fire or water submersion, or in a state where service and or maintenance is needed or being performed.
 
@@ -31 +34,3 @@
 2.43. "Explosion" means the sudden release of energy sufficient to cause pressure waves and/or projectiles that may cause structural and/or physical damage to the surrounding of the vehicle.
+
+8
```

</details>

### Page 015 — score=0.1
- Source: `UN-ECE-R94 source p.51 [docling_figure_page,figure_adjacent]`
- Chars: Docling=121, LightOn=344
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_015.md
+++ lighton/page_015.md
@@ -1,5 +1,16 @@
-[IMAGE]
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4 
+Annex 9 
 
-[IMAGE]
+---
 
-Figure 2 Locations of samples for certification
+**Figure 2** 
+*Locations of samples for certification*
+
+[IMAGE]263,142,775,497
+
+If $a \geq 900 \text{ mm}$ : $x = 1/3 \cdot (b - 600 \text{mm})$ and $y = 1/3 \cdot (a - 600 \text{mm})$ (for $a \leq b$ )
+
+[IMAGE]263,570,785,745
+
+51
```

</details>

### Page 055 — score=0.1
- Source: `UN-ECE-R129 source p.10 [control]`
- Chars: Docling=3213, LightOn=2990
- Section disagreement: **no** (docling=15, lighton=15)

<details><summary>Diff preview</summary>

```diff
--- docling/page_055.md
+++ lighton/page_055.md
@@ -1,26 +1,31 @@
-2.31. " Shoulder strap" means that part of an ECRS belt which restrains the child's upper torso.
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3 
 
-2.32 . " Crotch strap" means a strap (or divided straps, where two or more pieces of webbing make it) attached to the Child Restraint System and the lap strap and is so positioned as to pass between the child's thighs; it is designed to prevent the child sliding under the lap strap in normal use and prevent the lap strap moving up off the pelvis in an impact.
+---
 
-2.33. " Childdrestraining strap" means a strap which is a constituent part of the ECRS belt (harness) and restrains only the body of the child.
+2.31. "Shoulder strap" means that part of an ECRS belt which restrains the child's upper torso.
 
-2.34 . " Buckle" means a quick release device which enables the child to be held by the restraint or the restraint by the structure of the car and can be quickly opened. The buckle may incorporate the adjusting device.
+2.32. "Crotch strap" means a strap (or divided straps, where two or more pieces of webbing make it) attached to the Child Restraint System and the lap strap and is so positioned as to pass between the child's thighs; it is designed to prevent the child sliding under the lap strap in normal use and prevent the lap strap moving up off the pelvis in an impact.
 
-2.35 . " Enclosed buckle release button", a buckle release button such that it shall not be possible to release the buckle using a sphere having a diameter of 40 mm.
+2.33. "Child-restraining strap" means a strap which is a constituent part of the ECRS belt (harness) and restrains only the body of the child.
 
-2.36 . " Non -enclosed buckle release button", a buckle release button such that it shall be possible to release the buckle using a sphere having a diameter of 40 mm.
+2.34. "Buckle" means a quick release device which enables the child to be held by the restraint or the restraint by the structure of the car and can be quickly opened. The buckle may incorporate the adjusting device.
 
-2.37. " Adjusting device" means a device enabling the ECRS belt or its attachments to be adjusted to the physique of the wearer. The adjusting device may either be part of the buckle or be a retractor or any other part of the ECRS belt.
+2.35. "Enclosed buckle release button", a buckle release button such that it shall not be possible to release the buckle using a sphere having a diameter of 40 mm.
 
-2.38 . " Quick adjuster" means an adjusting device which can be operated by one hand in one smooth movement.
+2.36. "Non-enclosed buckle release button", a buckle release button such that it shall be possible to release the buckle using a sphere having a diameter of 40 mm.
 
-2.39 . " Adjuster mounted directly on Enhanced Child Restraint System" means an adjuster for the harness belt which is directly mounted on the Enhanced Child Restraint System, as opposed to being directly supported by the strap that it is designed to adjust.
+2.37. "Adjusting device" means a device enabling the ECRS belt or its attachments to be adjusted to the physique of the wearer. The adjusting device may either be part of the buckle or be a retractor or any other part of the ECRS belt.
 
-2.40 . " Energy absorber" means a device which is designed to dissipate energy independently of or jointly with the strap and forms part of an Enhanced Child Restraint System .
+2.38. "Quick adjuster" means an adjusting device which can be operated by one hand in one smooth movement.
 
-2.41 . " Retractor" means a device designed to accommodate a part or the whole of the strap of an Enhanced Child Restraint System. The term covers the following devices:
+2.39. "Adjuster mounted directly on Enhanced Child Restraint System" means an adjuster for the harness belt which is directly mounted on the Enhanced Child Restraint System, as opposed to being directly supported by the strap that it is designed to adjust.
 
-2.41.1. " Automatically-locking retractor", a retractor which allows extraction of the desired length of a strap and, when the buckle is fastened, automatically adjusts the strap to the wearer's physique, further extraction of the strap without voluntary intervention by the wearer being prevented.
+2.40. "Energy absorber" means a device which is designed to dissipate energy independently of or jointly with the strap and forms part of an Enhanced Child Restraint System.
 
```

</details>

### Page 009 — score=0.1
- Source: `UN-ECE-R94 source p.31 [docling_figure_page,figure_adjacent]`
- Chars: Docling=389, LightOn=601
- Section disagreement: **no** (docling=1, lighton=1)

<details><summary>Diff preview</summary>

```diff
--- docling/page_009.md
+++ lighton/page_009.md
@@ -1,5 +1,15 @@
-The sternum deflection velocity at time t is calculated from the filtered deflection as:
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4 
+Annex 4
 
-Where D(t) is the deflection at time t in metres and ∂t is the time interval in seconds between the measurements of deflection. The maximum value of ∂t shall be 1.25 x 10 -4 seconds. This calculation procedure is shown diagrammatically below:
+The sternum deflection velocity at time $t$ is calculated from the filtered deflection as:
 
-[IMAGE]
+$$
+V_{(t)} = \frac{8\left(D_{(t+1)} - D_{(t-1)}\right) - \left(D_{(t+2)} - D_{(t-2)}\right)}{12\partial t}
+$$
+
+Where $D_{(t)}$ is the deflection at time $t$ in metres and $\partial t$ is the time interval in seconds between the measurements of deflection. The maximum value of $\partial t$ shall be $1.25 \times 10^{-4}$ seconds. This calculation procedure is shown diagrammatically below:
+
+[IMAGE]255,263,775,565
+
+31
```

</details>

### Page 033 — score=0.1
- Source: `UN-ECE-R16 source p.5 [control]`
- Chars: Docling=2768, LightOn=2567
- Section disagreement: **no** (docling=10, lighton=10)

<details><summary>Diff preview</summary>

```diff
--- docling/page_033.md
+++ lighton/page_033.md
@@ -1,22 +1,25 @@
-# 1. Scope
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
 
-This Regulation applies to:
+---
 
-1.1. Vehicles of category M, N, O, L2, L4, L5, L6, L7 and T 1 ), with regard to the installation of safety-belts and restraint systems which are intended for separate use, i.e. as individual fittings, by persons of adult build occupying forwardfacing, rearward-facing and side-facing seats;
+## 1. Scope
 
-1.2. Safety-belts and restraint systems which are intended for separate use, i.e. as individual fittings, by persons of adult build occupying forward-facing, rearward -facing and side-facing seats, and are designed for installation in vehicles of category M, N, O, L2, L4, L5, L6, L7 and T 1 );
+1.1. Vehicles of category M, N, O, L₂, L₄, L₅, L₆, L₇ and T¹), with regard to the installation of safety-belts and restraint systems which are intended for separate use, i.e. as individual fittings, by persons of adult build occupying forward-facing, rearward-facing and side-facing seats;
 
-1.3. Vehicles of category M1 and N1 1 with regard to the installation of child restraint systems and ISOFIX child restraint systems.
+1.2. Safety-belts and restraint systems which are intended for separate use, i.e. as individual fittings, by persons of adult build occupying forward-facing, rearward-facing and side-facing seats, and are designed for installation in vehicles of category M, N, O, L₂, L₄, L₅, L₆, L₇ and T¹);
 
-1.4. Vehicles of categories M1 with regard to safety belt reminder 2 .
+1.3. Vehicles of category M₁ and N₁¹ with regard to the installation of child restraint systems and ISOFIX child restraint systems.
 
-1.5. At the request of the manufacturer, it also applies to the installation of child restraint systems and ISOFIX child restraint systems designated for installation in vehicles of categories M2 and M3 1 .
+1.4. Vehicles of categories M₁ with regard to safety belt reminder².
 
-1.6. At the request of the manufacturer, it also applies to safety-belts designated for installation on side -facing seats in vehicles of category M3 (Class II, III or B 1 ).
+1.5. At the request of the manufacturer, it also applies to the installation of child restraint systems and ISOFIX child restraint systems designated for installation in vehicles of categories M₂ and M₃¹.
 
-# 2. Definitions
+1.6. At the request of the manufacturer, it also applies to safety-belts designated for installation on side-facing seats in vehicles of category M₃ (Class II, III or B¹).
 
-# 2.1. Safety-belt (sea-belt, belt)
+## 2. Definitions
 
-An arrangement of straps with a securing buckle, adjusting devices and attachments which is capable of being anchored to the interior of a powerdriven vehicle and is designed to diminish the risk of injury to its wearer, in the event of collision or of abrupt deceleration of the vehicle, by limiting the mobility of the wearer's body. Such an arrangement is generally referred to as a "belt assembly", which term also embraces any device for absorbing energy or for retracting the belt.
+### 2.1. Safety-belt (sea-belt, belt)
+
+An arrangement of straps with a securing buckle, adjusting devices and attachments which is capable of being anchored to the interior of a power-driven vehicle and is designed to diminish the risk of injury to its wearer, in the event of collision or of abrupt deceleration of the vehicle, by limiting the mobility of the wearer’s body. Such an arrangement is generally referred to as a "belt assembly", which term also embraces any device for absorbing energy or for retracting the belt.
 
@@ -24,7 +27,7 @@
```

</details>

### Page 036 — score=0.1
- Source: `UN-ECE-R16 source p.11 [control]`
- Chars: Docling=3014, LightOn=2821
- Section disagreement: **no** (docling=18, lighton=18)

<details><summary>Diff preview</summary>

```diff
--- docling/page_036.md
+++ lighton/page_036.md
@@ -1,37 +1,52 @@
-2.40. " Visual warning" means a warning by visual signal (lighting, blinking or visual display of symbol or message).
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
 
-2.41. " Audible warning" means a warning by sound signal.
+2.40. "Visual warning" means a warning by visual signal (lighting, blinking or visual display of symbol or message).
 
-2.42. " First level warning" means a visual warning activated when the ignition switch is engaged (engine running or not) and the driver's safety-belt is not fastened. An audible warning can be added as an option.
+2.41. "Audible warning" means a warning by sound signal.
 
-2.43. " Second level warning" means a visual and audible warning activated when a driver operates a vehicle without fastening the safety-belt.
+2.42. "First level warning" means a visual warning activated when the ignition switch is engaged (engine running or not) and the driver's safety-belt is not fastened. An audible warning can be added as an option.
 
-2.44. " Safety-belt is not fastened" means, at the option of the manufacturer, either the driver safety-belt buckle is not engaged or the webbing length pulled out of the retractor is 100 mm or less.
+2.43. "Second level warning" means a visual and audible warning activated when a driver operates a vehicle without fastening the safety-belt.
 
-2.45. " Vehicle is in normal operation" means that vehicle is in forward motion at the speed greater than 10 km/h.
+2.44. "Safety-belt is not fastened" means, at the option of the manufacturer, either the driver safety-belt buckle is not engaged or the webbing length pulled out of the retractor is 100 mm or less.
 
-# 3. Application for approval
+2.45. "Vehicle is in normal operation" means that vehicle is in forward motion at the speed greater than 10 km/h.
 
-# 3.1. Vehicle type
+## 3. Application for approval
 
-3.1.1. The application for approval of a vehicle type with regard to the installation of its safety-belts and restraint systems shall be submitted by the vehicle manufacturer or by his duly accredited representative.
+### 3.1. Vehicle type
 
-3.1.2. It shall be accompanied by the undermentioned documents in triplicate and the following particulars:
+#### 3.1.1.
+The application for approval of a vehicle type with regard to the installation of its safety-belts and restraint systems shall be submitted by the vehicle manufacturer or by his duly accredited representative.
 
-3.1.2.1. Drawings of the general vehicle structure on an appropriate scale, showing the positions of the safety-belts, and detailed drawings of the safety-belts and of the points to which they are attached;
+#### 3.1.2.
+It shall be accompanied by the undermentioned documents in triplicate and the following particulars:
 
-3.1.2.2. A specification of the materials used which may affect the strength of the safety-belts;
```

</details>

### Page 049 — score=0.1
- Source: `UN-ECE-R16 source p.56 [r16_extra]`
- Chars: Docling=826, LightOn=636
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_049.md
+++ lighton/page_049.md
@@ -1,43 +1,26 @@
-## Ageing in air (ASTM Method D 573)
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 6 
 
-hardness:
+---
 
-breaking strength:
+**Ageing in air (ASTM Method D 573)** 
+hardness: 
+- breaking strength: decrease < 15 per cent of RB₀ 
+- elongation: decrease < 10 per cent of AB₀ 
+- volume: swelling < 5 per cent 
 
-decrease < 15 per cent of RB o
+**Immersion in oil (ASTM Method No. 3 Oil):** 
+70 hours at 100 °C 
+- breaking strength: decrease < 15 per cent of RB₀ 
+- elongation: decrease < 15 per cent of AB₀ 
+- volume: swelling < 20 per cent 
 
-elongation:
+**Immersion in distilled water:** 
+1 week at 70 °C 
+- breaking strength: decrease < 35 per cent of RB₀ 
+- elongation: increase < 20 per cent of AB₀ 
 
-decrease < 10 per cent of AB o
+---
 
-volume:
-
-swelling < 5 per cent
-
-# Immersion in oil (ASTM Method No. 3 Oil):
-
```

</details>

### Page 024 — score=0.1
- Source: `UN-ECE-R95 source p.25 [audit_flagged_neighbor,figure_adjacent]`
- Chars: Docling=2854, LightOn=2672
- Section disagreement: **no** (docling=17, lighton=17)

<details><summary>Diff preview</summary>

```diff
--- docling/page_024.md
+++ lighton/page_024.md
@@ -1,4 +1,8 @@
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 4 
+
 It shall be allowed by agreement between manufacturer and Technical Service to modify the fuel system so that an appropriate amount of fuel can be used to run the engine or the electrical energy conversion system.
 
-4.2. Vehicle equipment specification
+### 4.2. Vehicle equipment specification
 
@@ -6,7 +10,11 @@
 
-4.3. Mass of the vehicle
+### 4.3. Mass of the vehicle
 
-4.3.1. The vehicle to be tested shall have the reference mass as defined in paragraph 2.10. of this Regulation. The mass of the vehicle shall be adjusted to ±1 per cent of the reference mass.
+#### 4.3.1.
 
-4.3.2. The fuel tank shall be filled with water to a mass equal to 90 per cent of the mass of a full load of fuel as specified by the manufacturer with a tolerance of ±1 per cent.
+The vehicle to be tested shall have the reference mass as defined in paragraph 2.10. of this Regulation. The mass of the vehicle shall be adjusted to ±1 per cent of the reference mass.
+
+#### 4.3.2.
+
+The fuel tank shall be filled with water to a mass equal to 90 per cent of the mass of a full load of fuel as specified by the manufacturer with a tolerance of ±1 per cent.
 
@@ -14,26 +22,52 @@
 
-4.3.3. All the other systems (brake, cooling, etc.) may be empty; in this case, the mass of the liquids shall be offset.
+#### 4.3.3.
 
-4.3.4. If the mass of the measuring apparatus on board of the vehicle exceeds the 25 kg allowed, it may be offset by reductions which have no noticeable effect on the results of the test.
+All the other systems (brake, cooling, etc.) may be empty; in this case, the mass of the liquids shall be offset.
 
-4.3.5. The mass of the measuring apparatus shall not change each axle reference load by more than 5 per cent, each variation not exceeding 20 kg.
+#### 4.3.4.
 
-Preparation of the vehicle
+If the mass of the measuring apparatus on board of the vehicle exceeds the 25 kg allowed, it may be offset by reductions which have no noticeable effect on the results of the test.
```

</details>

### Page 065 — score=0.1
- Source: `UN-ECE-R129 source p.52 [docling_figure_page,figure_adjacent]`
- Chars: Docling=2875, LightOn=2696
- Section disagreement: **no** (docling=3, lighton=3)

<details><summary>Diff preview</summary>

```diff
--- docling/page_065.md
+++ lighton/page_065.md
@@ -1,5 +1,7 @@
-| | Q0 | Q1 | Q1.5 | Q3 | Q6 | Q10 (design targets) |
-|---------------------------------------------------|------------------|------------------|------------------|------------------|------------------|------------------------|
-| | Dimensions in mm | Dimensions in mm | Dimensions in mm | Dimensions in mm | Dimensions in mm | Dimensions in mm |
-| Height of spacer device for positioning of dummy | 173 ± 2 | 229 ± 2 | 237 ± 2 | 250 ± 2 | 270 ± 2 | 359 ± 2 |
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3
+
+The Enhanced Child Restraint System centre line shall be aligned with the centre line of the test bench.
+
+The dummy shall be placed in the Enhanced Child Restraint System separate from the seat-back of the chair by a flexible spacer. The spacer shall be 2.5 cm thick and 6 cm wide. It shall have length equal to the shoulder height less the thigh height, both in the sitting position and relevant to the dummy size being tested. The resulting height of the spacer is listed in the table below for the different dummy sizes. The board should follow as closely as possible the curvature of the chair and its lower end should be at the height of the dummy’s hip joint.
 
@@ -7,3 +9,31 @@
 
-Adjust the ECRS belt in accordance with the manufacturer's instructions, but to a tension of 250 ± 25 N above the adjuster force, with a deflection angle of the strap at the adjuster of 45 ± 5°, or alternatively, the angle prescribed by the manufacturer.
+<table>
+ <thead>
+ <tr>
+ <th rowspan="2"></th>
+ <th rowspan="2">Q0</th>
+ <th rowspan="2">Q1</th>
+ <th rowspan="2">Q1.5</th>
+ <th rowspan="2">Q3</th>
+ <th rowspan="2">Q6</th>
+ <th rowspan="2">Q10 (design targets)</th>
+ </tr>
+ <tr>
+ <th colspan="6">Dimensions in mm</th>
+ </tr>
+ </thead>
+ <tbody>
+ <tr>
+ <td>Height of spacer device for positioning of dummy</td>
+ <td>173 ± 2</td>
+ <td>229 ± 2</td>
+ <td>237 ± 2</td>
+ <td>250 ± 2</td>
+ <td>270 ± 2</td>
```

</details>

### Page 052 — score=0.1
- Source: `UN-ECE-R16 source p.59 [r16_extra]`
- Chars: Docling=216, LightOn=391
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_052.md
+++ lighton/page_052.md
@@ -1,9 +1,21 @@
-Surface finish of mandrell Interference tolerance ±0.1
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 6 
 
-[IMAGE]
+---
 
-[IMAGE]
+**Figure 4** 
+**Stopping device** 
+*(Olive-shaped knob)* 
 
-[IMAGE]
+[IMAGE]313,168,670,395 
+*This dimension can vary between 43 and 49 mm 
+Dimensions in mm 
 
-Figure 4 Stopping device (Olive-shaped knob)
+[IMAGE]313,465,670,685 
+Dimensions in mm 
+
+Surface finish $\sqrt[0.4]{\text{ }}$ 
+Interference tolerance $\pm 0.1$ 
+
+59
```

</details>

### Page 060 — score=0.1
- Source: `UN-ECE-R129 source p.18 [docling_figure_page,figure_adjacent]`
- Chars: Docling=1938, LightOn=1767
- Section disagreement: **no** (docling=3, lighton=3)

<details><summary>Diff preview</summary>

```diff
--- docling/page_060.md
+++ lighton/page_060.md
@@ -1,4 +1,7 @@
-4.8 .
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3 
 
-Additional markings
+---
+
+### 4.8. Additional markings
 
@@ -12,2 +15,4 @@
 
+[IMAGE]324,327,750,527
+
 (d) The adjustment of ISOFIX latches and the top tether, or other means of limiting Enhanced Child Restraint System rotation, requiring action by the user shall be indicated;
@@ -18,14 +23,12 @@
 
-4.9. An impact shield that is not permanently attached to the seat shall have a permanently attached label to indicate the brand and model of the Enhanced Child Restraint System to which it belongs and the size range. The minimum size of the label shall be 40 x 40 mm or the equivalent area.
+[IMAGE]440,675,570,745
 
-4.10. Enhanced Child Restraint Systems shall have a permanently attached label to inform the user of the appropriate method of restraint of the child over the entire stature range declared by the manufacturer.
+### 4.9.
 
-[IMAGE]
+An impact shield that is not permanently attached to the seat shall have a permanently attached label to indicate the brand and model of the Enhanced Child Restraint System to which it belongs and the size range. The minimum size of the label shall be 40 x 40 mm or the equivalent area.
 
-[IMAGE]
+### 4.10.
 
-[IMAGE]
+Enhanced Child Restraint Systems shall have a permanently attached label to inform the user of the appropriate method of restraint of the child over the entire stature range declared by the manufacturer.
 
-[IMAGE]
-
-[IMAGE]
+18
```

</details>

### Page 013 — score=0.1
- Source: `UN-ECE-R94 source p.47 [docling_figure_page,figure_adjacent]`
- Chars: Docling=2079, LightOn=2241
- Section disagreement: **no** (docling=15, lighton=15)

<details><summary>Diff preview</summary>

```diff
--- docling/page_013.md
+++ lighton/page_013.md
@@ -1,4 +1,12 @@
-The length of the sample shall be measured in three locations, 12.7 mm from each end and in the middle, and recorded as L1, L2 and L3 (Figure 3 of this annex). In the same manner, the width shall be measured and recorded as W 1 , W2 and W3 (Figure 3 of this annex). These measurements shall be taken on the centreline of the thickness. The crush area shall then be calculated as:
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4 
+Annex 9 
 
-# 2.4. Crush rate and distance
+The length of the sample shall be measured in three locations, 12.7 mm from each end and in the middle, and recorded as $L_1$ , $L_2$ and $L_3$ (Figure 3 of this annex). In the same manner, the width shall be measured and recorded as $W_1$ , $W_2$ and $W_3$ (Figure 3 of this annex). These measurements shall be taken on the centreline of the thickness. The crush area shall then be calculated as:
+
+$$
+A = \frac{(L_1 + L_2 + L_3)}{3} \times \frac{(W_1 + W_2 + W_3)}{3}
+$$
+
+### 2.4. Crush rate and distance
 
@@ -6,3 +14,3 @@
 
-# 2.5. Data collection
+### 2.5. Data collection
 
@@ -10,11 +18,9 @@
 
-# 2.6. Crush strength determination
+### 2.6. Crush strength determination
 
-Ignore all data prior to 6.4 mm of crush and after 16.5 mm of crush. Divide the remaining data into three sections or displacement intervals (n = 1, 2, 3) (see Figure 4 of this annex) as follows:
+Ignore all data prior to 6.4 mm of crush and after 16.5 mm of crush. Divide the remaining data into three sections or displacement intervals ( $n = 1, 2, 3$ ) (see Figure 4 of this annex) as follows:
 
-06.4 mm - 09.7 mm inclusive,
-
-09.7 mm - 13.2 mm exclusive,
-
-13.2 mm - 16.5 mm inclusive.
+(1) 06.4 mm - 09.7 mm inclusive, 
+(2) 09.7 mm - 13.2 mm exclusive, 
+(3) 13.2 mm - 16.5 mm inclusive.
 
@@ -22,5 +28,13 @@
```

</details>

### Page 002 — score=0.1
- Source: `UN-ECE-R94 source p.9 [control]`
- Chars: Docling=3016, LightOn=2856
- Section disagreement: **no** (docling=16, lighton=16)

<details><summary>Diff preview</summary>

```diff
--- docling/page_002.md
+++ lighton/page_002.md
@@ -1,35 +1,56 @@
-# 3. Application for approval
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4
 
-3.1. The application for approval of a vehicle type with regard to the protection of the occupants of the front seats in the event of a frontal collision (offset deformable barrier test) shall be submitted by the vehicle manufacturer or by his duly accredited representative.
+## 3. Application for approval
 
-3.2. It shall be accompanied by the undermentioned documents in triplicate and following particulars:
+### 3.1.
+The application for approval of a vehicle type with regard to the protection of the occupants of the front seats in the event of a frontal collision (offset deformable barrier test) shall be submitted by the vehicle manufacturer or by his duly accredited representative.
 
-3.2.1. A detailed description of the vehicle type with respect to its structure, dimensions, lines and constituent materials;
+### 3.2.
+It shall be accompanied by the undermentioned documents in triplicate and following particulars:
 
-3.2.2. Photographs, and/or diagrams and drawings of the vehicle showing the vehicle type in front, side and rear elevation and design details of the forward part of the structure;
+#### 3.2.1.
+A detailed description of the vehicle type with respect to its structure, dimensions, lines and constituent materials;
 
-3.2.3. Particulars of the vehicle's unladen kerb mass;
+#### 3.2.2.
+Photographs, and/or diagrams and drawings of the vehicle showing the vehicle type in front, side and rear elevation and design details of the forward part of the structure;
 
-3.2.4. The lines and inside dimensions of the passenger compartment;
+#### 3.2.3.
+Particulars of the vehicle’s unladen kerb mass;
 
-3.2.5. A description of the interior fittings and protective systems installed in the vehicle;
+#### 3.2.4.
+The lines and inside dimensions of the passenger compartment;
 
-3.2.6. A general description of the electrical power source type, location and the electrical power train (e.g. hybrid, electric).
+#### 3.2.5.
+A description of the interior fittings and protective systems installed in the vehicle;
 
-3.3. The applicant for approval shall be entitled to present any data and results of tests carried out which make it possible to establish that compliance with the requirements can be achieved with a sufficient degree of confidence.
+#### 3.2.6.
```

</details>

### Page 027 — score=0.1
- Source: `UN-ECE-R95 source p.36 [docling_figure_page,figure_adjacent]`
- Chars: Docling=2780, LightOn=2630
- Section disagreement: **no** (docling=16, lighton=16)

<details><summary>Diff preview</summary>

```diff
--- docling/page_027.md
+++ lighton/page_027.md
@@ -1,2 +1,8 @@
-6.2. Propulsion of the mobile deformable barrier
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 5 
+
+---
+
+### 6.2. Propulsion of the mobile deformable barrier
 
@@ -4,9 +10,9 @@
 
-6.3. Measuring instruments
+### 6.3. Measuring instruments
 
-6.3.1. Speed
+#### 6.3.1. Speed
 
-The impact speed shall be 35  0.5 km/h the instrument used to record the speed on impact shall be accurate to within 0.1 per cent .
+The impact speed shall be $35 \pm 0.5$ km/h the instrument used to record the speed on impact shall be accurate to within 0.1 per cent.
 
-6.3.2. Loads
+#### 6.3.2. Loads
 
@@ -14,34 +20,54 @@
 
-| CFC for all blocks: | 60 Hz |
-|----------------------------|----------|
-| CAC for blocks 1 and 3: | 200 kN |
-| CAC for blocks 4, 5 and 6: | 100 kN |
-| CAC for block 2: | 200 kN |
+<table>
+ <tr><td>CFC for all blocks:</td><td>60 Hz</td></tr>
+ <tr><td>CAC for blocks 1 and 3:</td><td>200 kN</td></tr>
+ <tr><td>CAC for blocks 4, 5 and 6:</td><td>100 kN</td></tr>
+ <tr><td>CAC for block 2:</td><td>200 kN</td></tr>
+</table>
 
```

</details>

### Page 050 — score=0.1
- Source: `UN-ECE-R16 source p.57 [r16_extra]`
- Chars: Docling=79, LightOn=217
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_050.md
+++ lighton/page_050.md
@@ -1,3 +1,15 @@
-[IMAGE]
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 6 
 
-Figure 1 Trolley, seat, anchorage
+---
+
+**Figure 1** 
+*Trolley, seat, anchorage*
+
+[IMAGE]116,187,865,337
+
+*Dimensions in millimeters 
+Tolerances: ±5 mm*
+
+57
```

</details>

### Page 023 — score=0.1
- Source: `UN-ECE-R95 source p.24 [annex,audit_flagged,audit_flagged_neighbor,figure_adjacent]`
- Chars: Docling=2702, LightOn=2566
- Section disagreement: **no** (docling=8, lighton=8)

<details><summary>Diff preview</summary>

```diff
--- docling/page_023.md
+++ lighton/page_023.md
@@ -1,10 +1,12 @@
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 4
+
 # Annex 4
 
-# Collision test procedure
+## Collision test procedure
 
-Installations
+### 1. Installations
 
-1.1.
-
-Testing ground
+#### 1.1. Testing ground
 
@@ -12,28 +14,38 @@
 
-Test conditions
+### 2. Test conditions
 
-2.1. The vehicle to be tested shall be stationary.
+#### 2.1.
 
-2.2. The mobile deformable barrier shall have the characteristics set out in Annex 5 to this Regulation. Requirements for the examination are given in the appendices to Annex 5. The mobile deformable barrier shall be equipped with a suitable device to prevent a second impact on the struck vehicle.
+The vehicle to be tested shall be stationary.
 
-2.3. The trajectory of the mobile deformable barrier longitudinal median vertical plane shall be perpendicular to the longitudinal median vertical plane of the impacted vehicle.
+#### 2.2.
 
-2.4. The longitudinal vertical median plane of the mobile deformable barrier shall be coincident within ±25 mm with a transverse vertical plane passing through the R point of the front seat adjacent to the struck side of the tested vehicle. The horizontal median plane limited by the external lateral vertical planes of the front face shall be at the moment of impact within two planes determined before the test and situated 25 mm above and below the previously defined plane.
+The mobile deformable barrier shall have the characteristics set out in Annex 5 to this Regulation. Requirements for the examination are given in the appendices to Annex 5. The mobile deformable barrier shall be equipped with a suitable device to prevent a second impact on the struck vehicle.
 
-2.5. Instrumentation shall comply with ISO 6487:1987 unless otherwise specified in this Regulation.
+#### 2.3.
 
```

</details>

### Page 039 — score=0.1
- Source: `UN-ECE-R16 source p.24 [r16_extra]`
- Chars: Docling=3319, LightOn=3442
- Section disagreement: **no** (docling=17, lighton=17)

<details><summary>Diff preview</summary>

```diff
--- docling/page_039.md
+++ lighton/page_039.md
@@ -1,30 +1,86 @@
-7.4.1.3.2. The strap shall then be kept for one and a half hours on a plane surface in a low -temperature chamber in which the air temperature is -30 + 5 °C. It shall then be folded and the fold shall be loaded with a mass of 2 kg previously cooled to -30 + 5 °C. When the strap has been kept under load for 30 minutes in the same low -temperature chamber, the mass shall be removed and the breaking load shall be measured within 5 minutes after removal of the strap from the low -temperature chamber.
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7
 
-7.4.1.4. Heat -conditioning
+---
 
-7.4.1.4.1. The strap shall be kept for three hours in a heating cabinet in an atmosphere having a temperature of 60 + 5 °C and a relative humidity of 65 + 5 per cent.
+### 7.4.1.3.2.
 
-7.4.1.4.2. The breaking load shall be determined within five minutes after removal of the strap from the heating cabinet.
+The strap shall then be kept for one and a half hours on a plane surface in a low-temperature chamber in which the air temperature is $-30 \pm 5^\circ \text{C}$ . It shall then be folded and the fold shall be loaded with a mass of 2 kg previously cooled to $-30 \pm 5^\circ \text{C}$ . When the strap has been kept under load for 30 minutes in the same low-temperature chamber, the mass shall be removed and the breaking load shall be measured within 5 minutes after removal of the strap from the low-temperature chamber.
 
-7.4.1.5. Exposure to water
+---
 
-7.4.1.5.1. The strap shall be kept fully immersed for three hours in distilled water, at a temperature of 20 + 5 °C, to which a trace of a wetting agent has been added. Any wetting agent suitable for the fibre under test may be used.
+### 7.4.1.4. Heat-conditioning
 
-7.4.1.5.2. The breaking load shall be determined within 10 minutes after removal of the strap from the water.
+#### 7.4.1.4.1.
 
-7.4.1.6. Abrasion conditioning
+The strap shall be kept for three hours in a heating cabinet in an atmosphere having a temperature of $60 \pm 5^\circ \text{C}$ and a relative humidity of $65 \pm 5$ per cent.
 
-7.4.1.6.1. The abrasion conditioning will be performed on every device in which the strap is in contact with a rigid part of the belt, with the exception of all adjusting devices where the micro -slip test (7.3.) shows that the strap slips by less than half the prescribed value, in which case, the procedure 1 abrasion conditioning (7.4.1.6.4.1.) will not be necessary. The setting on the conditioning device will approximately maintain the relative position of strap and contact area.
+#### 7.4.1.4.2.
 
-7.4.1.6.2. The samples shall be conditioned as described under paragraph 7.4.1.1. The ambient temperature during the abrasion procedure shall be between 15 and 30º C.
+The breaking load shall be determined within five minutes after removal of the strap from the heating cabinet.
 
-7.4.1.6.3. In the table below are listed the general conditions for each abrasion procedure.
+---
 
-| | Load daN | Frequency Hz | Cycles Numbers | Shift mm |
-|--------------|------------|----------------|------------------|------------|
-| Procedure 1 | 2.5 | 0.5 | 5,000 | 300 ± 20 |
```

</details>

### Page 006 — score=0.1
- Source: `UN-ECE-R94 source p.17 [docling_figure_page,figure_adjacent]`
- Chars: Docling=2273, LightOn=2153
- Section disagreement: **no** (docling=6, lighton=6)

<details><summary>Diff preview</summary>

```diff
--- docling/page_006.md
+++ lighton/page_006.md
@@ -1,2 +1,3 @@
-[IMAGE]
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4
 
@@ -4,9 +5,9 @@
 
-# 6. Instructions for users of vehicles equipped with airbags
+[IMAGE]130,134,510,287
 
-6.1. For a vehicle fitted with airbag assemblies intended to protect the driver and occupants other than the
+## 6. Instructions for users of vehicles equipped with airbags
 
-driver, compliance with paragraphs 8.1.8. to 8.1.9. of UN Regulation No. 16 as amended by the 08 series of amendments shall be demonstrated as from 1 September 2020 for new vehicle types. Before this date the relevant requirements of the preceding series of amendments apply.
+6.1. For a vehicle fitted with airbag assemblies intended to protect the driver and occupants other than the driver, compliance with paragraphs 8.1.8. to 8.1.9. of UN Regulation No. 16 as amended by the 08 series of amendments shall be demonstrated as from 1 September 2020 for new vehicle types. Before this date the relevant requirements of the preceding series of amendments apply.
 
-# 7. Modification and extension of approval of the vehicle type
+## 7. Modification and extension of approval of the vehicle type
 
@@ -18,3 +19,3 @@
 
-# 7.1.1. Revision
+### 7.1.1. Revision
 
@@ -24,3 +25,3 @@
 
-# 7.1.2. Extension
+### 7.1.2. Extension
 
@@ -33 +34,3 @@
 (c) Approval to a later series of amendments is requested after its entry into force.
+
+17
```

</details>

### Page 012 — score=0.1
- Source: `UN-ECE-R94 source p.46 [docling_figure_page,figure_adjacent]`
- Chars: Docling=2399, LightOn=2289
- Section disagreement: **no** (docling=8, lighton=8)

<details><summary>Diff preview</summary>

```diff
--- docling/page_012.md
+++ lighton/page_012.md
@@ -1,45 +1,46 @@
-Material: Aluminium 5251/5052 (ISO 209, part 1)
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4 
+Annex 9 
 
-1.5. Bumper facing sheet
+---
 
-Dimensions
+**1.5.** 
+Material: Aluminium 5251/5052 (ISO 209, part 1) 
+Bumper facing sheet 
+Dimensions 
+Height: 330 mm ± 2.5 mm 
+Width: 1,000 mm ± 2.5 mm 
+Thickness: 0.81 mm ± 0.07 mm 
+Material: Aluminium 5251/5052 (ISO 209, part 1) 
 
-Height: 330 mm ± 2.5 mm
+**1.6.** 
+Adhesive 
+The adhesive to be used throughout should be a two-part polyurethane (such as Ciba-Geigy XB5090/1 resin with XB5304 hardener, or equivalent). 
 
-Width: 1,000 mm ± 2.5 mm
+**2.** 
+Aluminum honeycomb certification 
+A complete testing procedure for certification of aluminium honeycomb is given in NHTSA TP-214D. The following is a summary of the procedure that should be applied to materials for the frontal impact barrier, these materials having a crush strength of 0.342 MPa and 1.711 MPa respectively. 
 
-Thickness: 0.81 mm ± 0.07 mm
+**2.1.** 
+Sample locations 
+To ensure uniformity of crush strength across the whole of the barrier face, eight samples shall be taken from four locations evenly spaced across the honeycomb block. For a block to pass certification, seven of these eight samples shall meet the crush strength requirements of the following sections. 
 
-Material: Aluminium 5251/5052 (ISO 209, part 1)
+The location of the samples depends on the size of the honeycomb block. First, four samples, each measuring 300 mm x 300 mm x 50 mm thick shall be cut from the block of barrier face material. Please refer to Figure 2 of this annex for an illustration of how to locate these sections within the honeycomb block. Each of these larger samples shall be cut into samples for certification testing (150 mm x 150 mm x 50 mm). Certification shall be based on the testing of two samples from each of these four locations. The other two should be made available to the applicant, upon request. 
 
-# 1.6. Adhesive
```

</details>

### Page 010 — score=0.1
- Source: `UN-ECE-R94 source p.39 [docling_figure_page,figure_adjacent]`
- Chars: Docling=119, LightOn=227
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_010.md
+++ lighton/page_010.md
@@ -1,3 +1,13 @@
-[IMAGE]
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4 
+Annex 7 - Appendix 
 
-Annex 7 - Appendix Equivalence curve - tolerance band for curve V = f(t)
+---
+
+## Annex 7 - Appendix
+
+### Equivalence curve - tolerance band for curve $\Delta V = f(t)$
+
+[IMAGE]203,185,743,720
+
+39
```

</details>

### Page 037 — score=0.1
- Source: `UN-ECE-R16 source p.22 [r16_extra]`
- Chars: Docling=2875, LightOn=2977
- Section disagreement: **no** (docling=11, lighton=11)

<details><summary>Diff preview</summary>

```diff
--- docling/page_037.md
+++ lighton/page_037.md
@@ -1,27 +1,89 @@
-6.4.2.2. The parts of the belt assembly to be subjected to an abrasion procedure are given in the following table and the procedure types which may be appropriate for them are indicated by "x". A new sample shall be used for each procedure.
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7
 
-| | Procedure 1 | Procedure 2 | Procedure 3 |
-|-------------------------|----------------|----------------|---------------|
-| Attachment | - | - | x |
-| Guide or Pulley | - | x | - |
-| Buckle - loop | - | x | x |
-| Adjusting device | x | - | x |
-| Parts sewn to the strap | - | - | x |
+---
 
-# 7. Tests
+### 6.4.2.2.
 
-7.1. Use of samples submitted for approval of a type of belt or restraint system (see Annex 13 to this Regulation)
+The parts of the belt assembly to be subjected to an abrasion procedure are given in the following table and the procedure types which may be appropriate for them are indicated by "x". A new sample shall be used for each procedure.
 
-7.1.1. Two belts or restraint systems are required for the buckle inspection, the lowtemperature buckle test, the low-temperature test described in paragraph 7.5.4. below where necessary, the buckle durability test, the belt corrosion test, the retractor operating tests, the dynamic test and the buckle-opening test after the dynamic test. One of these two samples shall be used for the inspection of the belt or restraint system.
+<table>
+ <thead>
+ <tr>
+ <th></th>
+ <th>Procedure 1</th>
+ <th>Procedure 2</th>
+ <th>Procedure 3</th>
+ </tr>
+ </thead>
+ <tbody>
+ <tr>
+ <td>Attachment</td>
+ <td>-</td>
+ <td>-</td>
+ <td>x</td>
+ </tr>
+ <tr>
```

</details>

### Page 046 — score=0.0
- Source: `UN-ECE-R16 source p.53 [audit_flagged_neighbor,figure_adjacent,r16_extra]`
- Chars: Docling=133, LightOn=220
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_046.md
+++ lighton/page_046.md
@@ -1,4 +1,10 @@
-# Annex 5
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 5 
 
-# Diagram of an apparatus for dust-resistance test
+---
+
+## Annex 5
+
+### Diagram of an apparatus for dust-resistance test
 
@@ -6,2 +12,4 @@
 
-[IMAGE]
+[IMAGE]197,245,805,765
+
+53
```

</details>

### Page 045 — score=0.0
- Source: `UN-ECE-R16 source p.48 [r16_extra]`
- Chars: Docling=679, LightOn=750
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_045.md
+++ lighton/page_045.md
@@ -1,2 +1,6 @@
-[IMAGE]
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 2 
+
+[IMAGE]327,103,520,275
 
@@ -6,4 +10,8 @@
 
-[IMAGE]
+[IMAGE]327,387,520,548
+
+06 24391
 
 The belt bearing the above approval mark is part of a restraint system ("Z"), it is a special type belt ("S") fitted with an energy absorber ("e"). It has been approved in the Netherlands (E4) under the number 0624391, the Regulation already incorporating the 06 series of amendments at the time of approval.
+
+48
```

</details>

### Page 019 — score=0.0
- Source: `UN-ECE-R95 source p.19 [audit_flagged_neighbor,figure_adjacent]`
- Chars: Docling=540, LightOn=610
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_019.md
+++ lighton/page_019.md
@@ -1,3 +1,12 @@
-# 11. Names and addresses of Technical Services responsible for conducting approval tests, and of Type Approval Authorities
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+
+---
+
+## 11. Names and addresses of Technical Services responsible for conducting approval tests, and of Type Approval Authorities
 
 The Contracting Parties to the Agreement applying this Regulation shall communicate to the United Nations secretariat the names and addresses of the Technical Services responsible for conducting approval tests, and of the Type Approval Authority which grant approval and to which forms certifying approval or extension, or refusal or withdrawal of approval, issued in other countries, are to be sent.
+
+---
+
+19
```

</details>

### Page 014 — score=0.0
- Source: `UN-ECE-R94 source p.50 [docling_figure_page,figure_adjacent]`
- Chars: Docling=179, LightOn=245
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_014.md
+++ lighton/page_014.md
@@ -1,6 +1,13 @@
-Figure 1
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4 
+Annex 9 
 
-# Deformable barrier for frontal impact testing
+---
 
-[IMAGE]
+**Figure 1** 
+*Deformable barrier for frontal impact testing*
+
+[IMAGE]230,144,837,498
+
+Ground
 
@@ -9 +16,3 @@
 All dimensions in mm.
+
+50
```

</details>

### Page 022 — score=0.0
- Source: `UN-ECE-R95 source p.23 [audit_flagged_neighbor,figure_adjacent]`
- Chars: Docling=589, LightOn=655
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_022.md
+++ lighton/page_022.md
@@ -1,11 +1,22 @@
-# Annex 3
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 3 
 
-# Procedure for determining the "H" point and the actual torso angle for seating positions in motor vehicles 1
+---
 
-# Appendix 1 - Description of the three dimensional "H" point machine (3-D H machine) 1
+## Annex 3
 
-# Appendix 2 - Three-dimensional reference system 1
+Procedure for determining the "H" point and the actual torso angle for seating positions in motor vehicles¹
 
-Appendix 3 - Reference data concerning seating positions 1
+**Appendix 1** - Description of the three dimensional "H" point machine (3-D H machine)¹
 
-1 The procedure is described in Annex 1 to the Consolidated Resolution on the Construction of Vehicles (RE.3) (document ECE/TRANS/WP.29/78/Rev.3). https://unece.org/transport/standards/transport/vehicle-regulations-wp29/resolutions
+**Appendix 2** - Three-dimensional reference system¹
+
+**Appendix 3** - Reference data concerning seating positions¹
+
+---
+
+¹ The procedure is described in Annex 1 to the Consolidated Resolution on the Construction of Vehicles (RE.3) (document ECE/TRANS/WP.29/78/Rev.3). 
+https://unece.org/transport/standards/transport/vehicle-regulations-wp29/resolutions
+
+23
```

</details>

### Page 063 — score=0.0
- Source: `UN-ECE-R129 source p.33 [docling_figure_page,figure_adjacent]`
- Chars: Docling=749, LightOn=814
- Section disagreement: **no** (docling=2, lighton=2)

<details><summary>Diff preview</summary>

```diff
--- docling/page_063.md
+++ lighton/page_063.md
@@ -1,2 +1,3 @@
-[IMAGE]
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3
 
@@ -4,3 +5,7 @@
 
-# 6.3.5.4. Support-leg foot jig
+[IMAGE]200,122,610,360
+
+10 mm
+
+### 6.3.5.4. Support-leg foot jig
 
@@ -8,6 +13,11 @@
 
-The jig is defined as the ISOFIX CRF corresponding to the size class of the Enhanced Child Restraint System. The jig is expanded with two 6 mm diameter ISOFIX low anchorages. The striped box positioned in front of the jig is positioned and sized according paragraph 6.3.5.2. above. The ECRS shall have its attachments latched when conducting the assessment .
-
-[IMAGE]
+The jig is defined as the ISOFIX CRF corresponding to the size class of the Enhanced Child Restraint System. The jig is expanded with two 6 mm diameter ISOFIX low anchorages. The striped box positioned in front of the jig is positioned and sized according paragraph 6.3.5.2. above. The ECRS shall have its attachments latched when conducting the assessment.
 
 Figure 0(e)
+
+[IMAGE]200,577,665,845
+
+ISOFIX axle 
+6 mm round
+
+33
```

</details>

### Page 017 — score=0.0
- Source: `UN-ECE-R95 source p.9 [control]`
- Chars: Docling=3181, LightOn=3119
- Section disagreement: **no** (docling=11, lighton=11)

<details><summary>Diff preview</summary>

```diff
--- docling/page_017.md
+++ lighton/page_017.md
@@ -1,23 +1,57 @@
-3.4.2. It shall be the responsibility of the applicant for approval to show that the application of paragraph 3.4.1 above is in compliance with the requirements of this Regulation.
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
 
-# 4. Approval
+---
 
-4.1. If the vehicle type submitted for approval pursuant to this Regulation meets the requirements of paragraph 5 . below, approval of that vehicle type shall be granted.
+### 3.4.2.
 
-4.2. In case of doubt, account shall be taken, when verifying the conformity of the vehicle to the requirements of this Regulation, of any data or test results provided by the manufacturer which can be taken into consideration in validating the approval test carried out by the Technical Service.
+It shall be the responsibility of the applicant for approval to show that the application of paragraph 3.4.1 above is in compliance with the requirements of this Regulation.
 
-4.3. An approval number shall be assigned to each type approved. Its first two digits (at present 03 corresponding to the 03 series of amendments) shall indicate the series of amendments incorporating the most recent major technical amendments made to the Regulation at the time of issue of the approval. The same Contracting Party may not assign the same approval number to another vehicle type.
+---
 
-4.4. Notice of approval or of extension or of refusal of approval of a vehicle type pursuant to this Regulation shall be communicated by the Parties to the Agreement applying this Regulation by means of a form conforming to the model in Annex 1 to this Regulation and photographs and/or diagrams and drawings supplied by the applicant for approval, in a format not exceeding A4 (210 x 297 mm) or folded to that format and on an appropriate scale.
+## 4. Approval
 
-4.5. There shall be affixed to every vehicle conforming to a vehicle type approved under this Regulation, conspicuously and in a readily accessible place specified on the approval form, an international approval mark consisting of:
+### 4.1.
 
-4.5.1. A circle surrounding the letter "E" followed by the distinguishing number of the country which has granted approval; 2
+If the vehicle type submitted for approval pursuant to this Regulation meets the requirements of paragraph 5. below, approval of that vehicle type shall be granted.
 
-4.5.2. The number of this Regulation, followed by the letter "R", a dash and the approval number, to the right of the circle prescribed in paragraph 4.5.1. above.
+### 4.2.
 
-4.6. If the vehicle conforms to a vehicle type approved, under one or more other Regulations annexed to the Agreement, in the country which has granted approval under this Regulation, the symbol prescribed in paragraph 4.5.1. above need not be repeated; in this case the Regulation and approval numbers and the additional symbols of all the Regulations under which approval has been granted in the country which has granted approval under this Regulation shall be placed in vertical columns to the right of the symbol prescribed in paragraph 4.5.1. above.
+In case of doubt, account shall be taken, when verifying the conformity of the vehicle to the requirements of this Regulation, of any data or test results provided by the manufacturer which can be taken into consideration in validating the approval test carried out by the Technical Service.
 
-4.7. The approval mark shall be clearly legible and shall be indelible.
+### 4.3.
 
-2 The distinguishing numbers of the Contracting Parties to the 1958 Agreement are reproduced in Annex 3 to the Consolidated Resolution on the Construction of Vehicles (R.E.3), document ECE/TRANS/WP.29/78/Rev. 3, Annex 3 -https://unece.org/transport/standards/transport/vehicle-regulations-wp29/resolutions
+An approval number shall be assigned to each type approved. Its first two digits (at present 03 corresponding to the 03 series of amendments) shall indicate the series of amendments incorporating the most recent major technical amendments made to the Regulation at the time of issue of the approval. The same Contracting Party may not assign the same approval number to another vehicle type.
+
```

</details>

### Page 059 — score=0.0
- Source: `UN-ECE-R129 source p.17 [docling_figure_page,figure_adjacent]`
- Chars: Docling=2355, LightOn=2295
- Section disagreement: **no** (docling=9, lighton=9)

<details><summary>Diff preview</summary>

```diff
--- docling/page_059.md
+++ lighton/page_059.md
@@ -1,2 +1,7 @@
-4.6.2. Specific Vehicle ISOFIX ECRS .
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3
+
+---
+
+### 4.6.2. Specific Vehicle ISOFIX ECRS.
 
@@ -6,5 +11,5 @@
 
-[IMAGE]
+[IMAGE]340,275,500,355
 
-# ISO/F2, ISO/R3 and ISO/L1
+ISO/F2, ISO/R3 and ISO/L1
 
@@ -12,22 +17,40 @@
 
+[IMAGE]470,420,560,500
+
 Specific Vehicle ISOFIX
 
-[IMAGE]
+---
 
-4.6.3. An international approval mark as defined in paragraph 5.4.1. In case the ECRS containing module(s) this marking shall be permanently attached to the part of the ECRS which includes the ISOFIX attachments.
+### 4.6.3.
 
-4.6.4. An international module mark as defined in paragraph 5.4.3. In case the ECRS containing module(s) this marking shall be permanently attached to the module part of the ECRS.
+An international approval mark as defined in paragraph 5.4.1. In case the ECRS containing module(s) this marking shall be permanently attached to the part of the ECRS which includes the ISOFIX attachments.
 
-4.7. Marking for non-integral ECRS
+---
 
-4.7.1. i -Size booster seat Enhanced Child Restraint Systems shall have a permanently attached label with the following information visible to the person installing the Enhanced Child Restraint System in the car:
+### 4.6.4.
 
```

</details>

### Page 021 — score=0.0
- Source: `UN-ECE-R95 source p.21 [audit_flagged_neighbor,figure_adjacent]`
- Chars: Docling=686, LightOn=633
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_021.md
+++ lighton/page_021.md
@@ -1,13 +1,10 @@
-Place:
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 1 
 
-................................ ................................ ................................ .......................
+13. Place: .................................................................................................................... 
+14. Date: .................................................................................................................... 
+15. Signature: .............................................................................................................. 
+16. The list of documents deposited with the Type Approval Authority which has granted approval is annexed to this communication and may be obtained on request. 
 
-14.
-
-Date:.................................................................. ................................ .......................
-
-Signature:
-
-................................ ................................ ................................ ................
-
-The list of documents deposited with the Type Approval Authority which has granted approval is annexed to this communication and may be obtained on request.
+21
```

</details>

### Page 057 — score=0.0
- Source: `UN-ECE-R129 source p.15 [docling_figure_page,figure_adjacent]`
- Chars: Docling=583, LightOn=633
- Section disagreement: **no** (docling=1, lighton=1)

<details><summary>Diff preview</summary>

```diff
--- docling/page_057.md
+++ lighton/page_057.md
@@ -1,2 +1,5 @@
-[IMAGE]
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3
+
+[IMAGE]178,123,835,600
 
@@ -5 +8,3 @@
 The manufacturer shall be permitted to include the word "months" to explain the symbol "M" in the label. The word "months" should be in a language commonly spoken in the country or countries where the product is sold. More than one language is allowed.
+
+15
```

</details>

### Page 029 — score=0.0
- Source: `UN-ECE-R95 source p.39 [docling_figure_page,figure_adjacent]`
- Chars: Docling=345, LightOn=298
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_029.md
+++ lighton/page_029.md
@@ -1,17 +1,17 @@
-[IMAGE]
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 5 
 
-(including the front plate but not the back plate)
+[IMAGE]195,97,695,317
 
-Figure 3
+Figure 3 
+**Aluminium honeycomb orientation**
 
-# Aluminium honeycomb orientation
+[IMAGE]210,379,570,600
 
-[IMAGE]
+Figure 4 
+**Dimension of aluminium honeycomb cells**
 
-Expansion direction of the Aluminium honeycomb
+[IMAGE]200,657,660,787
 
-Figure 4
-
-# Dimension of aluminium honeycomb cells
-
-[IMAGE]
+39
```

</details>

### Page 058 — score=0.0
- Source: `UN-ECE-R129 source p.16 [docling_figure_page,figure_adjacent]`
- Chars: Docling=746, LightOn=787
- Section disagreement: **no** (docling=2, lighton=2)

<details><summary>Diff preview</summary>

```diff
--- docling/page_058.md
+++ lighton/page_058.md
@@ -1,15 +1,19 @@
-# Minimum label size 40 x 40 mm
+E/ECE/324/Rev.2/Add.128/Rev.3 
+E/ECE/TRANS/505/Rev.2/Add.128/Rev.3 
 
-[IMAGE]
+Minimum label size 40 x 40 mm 
 
-4.6. Marking for integral ECRS including ISOFIX connections attachments.
+[IMAGE]178,143,817,495
 
-The marking shall be located on the part of the ECRS which includes the ISOFIX attachments.
+4.6. Marking for integral ECRS including ISOFIX connections attachments. 
+The marking shall be located on the part of the ECRS which includes the ISOFIX attachments. 
 
-One of the following information labels shall be permanently visible to someone installing the Enhanced Child Restraint System in a vehicle:
+One of the following information labels shall be permanently visible to someone installing the Enhanced Child Restraint System in a vehicle: 
 
-# 4.6.1. i -Size ECRS:
+4.6.1. i-Size ECRS: 
 
-i -Size logo. The symbol shown below shall have minimum dimension of 25 x 25 mm and the pictogram shall contrast with the background. The pictogram shall be clearly visible either by means of contrasting colors or by adequate relief if it is moulded or embossed;
+*i-Size logo*. The symbol shown below shall have minimum dimension of 25 x 25 mm and the pictogram shall contrast with the background. The pictogram shall be clearly visible either by means of contrasting colors or by adequate relief if it is moulded or embossed; 
 
-[IMAGE]
+[IMAGE]367,685,630,845
+
+16
```

</details>

### Page 003 — score=0.0
- Source: `UN-ECE-R94 source p.11 [figure_adjacent,r94_figure3_area]`
- Chars: Docling=1834, LightOn=1794
- Section disagreement: **no** (docling=5, lighton=5)

<details><summary>Diff preview</summary>

```diff
--- docling/page_003.md
+++ lighton/page_003.md
@@ -1 +1,4 @@
+E/ECE/324/Rev.1/Add.93/Rev.4 
+E/ECE/TRANS/505/Rev.1/Add.93/Rev.4
+
 requirements of paragraph 5.2.8. below. This can be met by a separate impact test at the request of the manufacturer and after validation by the Technical Service, provided that the electrical components do not influence the occupant protection performance of the vehicle type as defined in paragraphs 5.2.1. to 5.2.5. of this Regulation. In case of this condition the requirements of paragraph 5.2.8. shall be checked in accordance with the methods set out in Annex 3 to this Regulation, except paragraphs, 2., 5. and 6. of Annex 3. But a dummy corresponding to the specifications for Hybrid III (see footnote 1 of Annex 3) fitted with a 45° angle and meeting the specifications for its adjustment shall be installed in each of the front outboard seats.
@@ -6,10 +9,14 @@
 
-5.2.1.2. The Injury Criteria for the neck (NIC) shall not exceed the values shown in Figures 1 and 2 4 ;
+5.2.1.2. The Injury Criteria for the neck (NIC) shall not exceed the values shown in Figures 1 and 2⁴;
 
-Figure 1 Neck tension criterion
+Figure 1 
+Neck tension criterion
 
-[IMAGE]
+[IMAGE]200,425,844,667
 
-Figure 2 Neck shear criterion
+Figure 2 
+Neck shear criterion
 
-4 Until 1 October 1998, the values obtained for the neck shall not be pass/fail criteria for the purposes of granting approval. The results obtained shall be recorded in the test report and be collected by the Type Approval Authority. After this date, the values specified in this paragraph shall apply as pass/fail criteria unless or until alternative values are adopted.
+⁴ Until 1 October 1998, the values obtained for the neck shall not be pass/fail criteria for the purposes of granting approval. The results obtained shall be recorded in the test report and be collected by the Type Approval Authority. After this date, the values specified in this paragraph shall apply as pass/fail criteria unless or until alternative values are adopted.
+
+11
```

</details>

### Page 044 — score=0.0
- Source: `UN-ECE-R16 source p.47 [audit_flagged_neighbor,figure_adjacent,r16_extra]`
- Chars: Docling=1295, LightOn=1263
- Section disagreement: **no** (docling=1, lighton=1)

<details><summary>Diff preview</summary>

```diff
--- docling/page_044.md
+++ lighton/page_044.md
@@ -1,6 +1,8 @@
-2.
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 2 
 
-Arrangements of the safety-belt approval marks (See paragraph 5.3.5. of this Regulation)
+2. Arrangements of the safety-belt approval marks (See paragraph 5.3.5. of this Regulation)
 
-[IMAGE]
+[IMAGE]300,150,600,317
 
@@ -10,7 +12,3 @@
 
-[IMAGE]
-
-B  4 m
-
-06 2489
+[IMAGE]330,400,550,550
 
@@ -18,2 +16,4 @@
 
-Note: The approval number and additional symbol(s) must be placed close to the circle and either above or below the "E" or to left or right of that letter. The digits of the approval number must be on the same side of the "E" and orientated in the same direction. The additional symbol(s) must be diametrically opposite the approval number. The use of roman numerals as approval numbers should be avoided so as to prevent any confusion with other symbols.
+*Note:* The approval number and additional symbol(s) must be placed close to the circle and either above or below the "E" or to left or right of that letter. The digits of the approval number must be on the same side of the "E" and orientated in the same direction. The additional symbol(s) must be diametrically opposite the approval number. The use of roman numerals as approval numbers should be avoided so as to prevent any confusion with other symbols.
+
+47
```

</details>

### Page 028 — score=0.0
- Source: `UN-ECE-R95 source p.38 [docling_figure_page,figure_adjacent]`
- Chars: Docling=1409, LightOn=1431
- Section disagreement: **no** (docling=4, lighton=4)

<details><summary>Diff preview</summary>

```diff
--- docling/page_028.md
+++ lighton/page_028.md
@@ -1,21 +1,38 @@
-t1 is the time where the trolley comes to rest, i.e. where u = 0,
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
+Annex 5
 
-s is the deflection of the trolley deformable element calculated according to paragraph 6.6.3.
+$t_{1}$ is the time where the trolley comes to rest, i.e. where $u = 0$ , 
+$s$ is the deflection of the trolley deformable element calculated according to paragraph 6.6.3.
 
-6.6.5. Verification of dynamic force data
+### 6.6.5. Verification of dynamic force data
 
-6.6.5.1. Compare the total impulse, I, calculated from the integration of the total force over the period of contact, with the momentum change over that period (M*)V).
+#### 6.6.5.1.
+Compare the total impulse, $I$ , calculated from the integration of the total force over the period of contact, with the momentum change over that period $(M^*)V$ .
 
-6.6.5.2. Compare the total energy change to the change in kinetic energy of the MDB, given by:
+#### 6.6.5.2.
+Compare the total energy change to the change in kinetic energy of the MDB, given by:
 
-Where Vi is the impact velocity and M the whole mass of the MDB
+$$
+E_K = \frac{1}{2} MV_i^2
+$$
 
-If the momentum change (M*)V) is not equal to the total impulse (I) ± 5 per cent, or if the total energy absorbed ( E n ) is not equal to the kinetic energy, EK ± 5 per cent, then the test data must be examined to determine the cause of this error.
+Where $V_i$ is the impact velocity and $M$ the whole mass of the MDB
 
-Figure 1 Design of impactor 2
+If the momentum change $(M^*)V$ is not equal to the total impulse $(I) \pm 5$ per cent, or if the total energy absorbed $(E E_n)$ is not equal to the kinetic energy, $E_K \pm 5$ per cent, then the test data must be examined to determine the cause of this error.
 
-[IMAGE]
+---
 
-Figure 2 Impact Top
+**Figure 1** 
+*Design of impactor²*
```

</details>

### Page 043 — score=0.0
- Source: `UN-ECE-R16 source p.46 [audit_flagged,audit_flagged_neighbor,figure_adjacent,r16_extra]`
- Chars: Docling=1195, LightOn=1215
- Section disagreement: **no** (docling=2, lighton=2)

<details><summary>Diff preview</summary>

```diff
--- docling/page_043.md
+++ lighton/page_043.md
@@ -1,12 +1,19 @@
-# Annex 2
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 2
 
-# Arrangements of the approval marks
+---
 
-Arrangements of the vehicle approval marks concerning the installation of safety-belts
+## Annex 2
 
-Model A
+### Arrangements of the approval marks
 
-(See paragraph 5.2.4. of this Regulation)
+1. Arrangements of the vehicle approval marks concerning the installation of safety-belts
 
-[IMAGE]
+#### Model A 
+*(See paragraph 5.2.4. of this Regulation)*
+
+[IMAGE]330,299,630,410
+
+a = 8 mm min.
 
@@ -14,7 +21,6 @@
 
-# Model B
+#### Model B 
+*(See paragraph 5.2.5. of this Regulation)*
 
-(See paragraph 5.2.5. of this Regulation)
-
-[IMAGE]
+[IMAGE]103,570,807,670
 
@@ -22,4 +28,8 @@
```

</details>

### Page 051 — score=0.0
- Source: `UN-ECE-R16 source p.58 [r16_extra]`
- Chars: Docling=350, LightOn=365
- Section disagreement: **no** (docling=0, lighton=0)

<details><summary>Diff preview</summary>

```diff
--- docling/page_051.md
+++ lighton/page_051.md
@@ -1,15 +1,26 @@
-[IMAGE]
+E/ECE/324/Rev.1/Add.15/Rev.7 
+E/ECE/TRANS/505/Rev.1/Add.15/Rev.7 
+Annex 6 
 
-Figure 2 Stopping device (Assembled)
+---
 
-[IMAGE]
+**Figure 2** 
+*Stopping device* 
+*(Assembled)* 
 
-Figure 3 Stopping device (Polyurethane tube)
+[IMAGE]133,185,862,400
 
-[IMAGE]
+---
 
-Interference tolerance ±0.2
+**Figure 3** 
+*Stopping device* 
+*(Polyurethane tube)* 
 
-All dimensions in mm
+[IMAGE]200,545,870,675
 
-Surface finish of mandrell Interference tolerance ±0.1
+Surface finish 
+of mandrell 
+Interference tolerance ±0.2 
+All dimensions in mm 
+
+58
```

</details>

### Page 018 — score=0.0
- Source: `UN-ECE-R95 source p.12 [docling_figure_page,figure_adjacent]`
- Chars: Docling=1791, LightOn=1781
- Section disagreement: **no** (docling=10, lighton=10)

<details><summary>Diff preview</summary>

```diff
--- docling/page_018.md
+++ lighton/page_018.md
@@ -1,17 +1,42 @@
-[IMAGE]
+E/ECE/324/Rev.1/Add.94/Rev.3 
+E/ECE/TRANS/505/Rev.1/Add.94/Rev.3 
 
-Figure
+---
 
-5.3.1.1. In the case of automatically activated door locking systems which are installed optionally and/or which can be de-activated by the driver, this requirement shall be verified by using one of the following two test procedures, at the choice of the manufacturer:
+## Figure
 
-5.3.1.1.1. If testing in accordance with Annex 4, paragraph 5.2.2.1., the manufacturer shall in addition demonstrate to the satisfaction of the Technical Service (e.g. manufacturer's in -house data) that, in the absence of the system or when the system is de-activated, no door will open in case of the impact.
+[IMAGE]185,126,700,470
 
-5.3.1.1.2. If testing in accordance with Annex 4, paragraph 5.2.2.2., the manufacturer shall in addition demonstrate that the inertial load requirements of paragraph 6.1.4. of the 03 series of amendments to Regulation No. 11 are met for the unlocked side doors on the non -struck side.
+---
 
-5.3.2. After the impact, the side doors on the non-struck side shall be unlocked.
+### 5.3.1.1.
 
-5.3.2.1. In the case of vehicles equipped with an automatically activated door locking system, the doors shall be locked before the moment of impact and be unlocked after the impact at least on the non-struck side.
+In the case of automatically activated door locking systems which are installed optionally and/or which can be de-activated by the driver, this requirement shall be verified by using one of the following two test procedures, at the choice of the manufacturer:
 
-5.3.2.2. In the case of automatically activated door locking systems which are installed optionally and/or which can be de-activated by the driver, this requirement shall be verified by using one of the following two test procedures, at the choice of the manufacturer:
+#### 5.3.1.1.1.
 
-5.3.2.2.1. If testing in accordance with Annex 4, paragraph 5.2.2.1, the manufacturer shall in addition demonstrate to the satisfaction of the Technical Service (e.g.
+If testing in accordance with Annex 4, paragraph 5.2.2.1., the manufacturer shall in addition demonstrate to the satisfaction of the Technical Service (e.g. manufacturer’s in-house data) that, in the absence of the system or when the system is de-activated, no door will open in case of the impact.
+
+#### 5.3.1.1.2.
+
+If testing in accordance with Annex 4, paragraph 5.2.2.2., the manufacturer shall in addition demonstrate that the inertial load requirements of paragraph 6.1.4. of the 03 series of amendments to Regulation No. 11 are met for the unlocked side doors on the non-struck side.
+
+### 5.3.2.
+
+After the impact, the side doors on the non-struck side shall be unlocked.
+
+#### 5.3.2.1.
```

</details>

