## Page 1

 
Euro NCAP 
Version 1.1.0 — October 2025 
 
         
   
 
 
Version 1.1 
October 2025 
 
 
 
 
Safe Driving 
Vehicle Assistance 
Protocol 
 
Implementation January 2026  


## Page 2

 
Euro NCAP 
Version 1.1.0 — October 2025 
 
         
   
 
Copyright © Euro NCAP 2025 - This work is the intellectual property of Euro NCAP. Permission is granted 
for this material to be shared for non-commercial, educational purposes, provided that this copyright 
statement appears on the reproduced materials and notice is given that the copying is by permission of 
Euro NCAP. To disseminate otherwise or to republish requires written permission from Euro NCAP. 


## Page 3

 
 
PREFACE 
During the test preparation, Vehicle Manufacturers are encouraged to liaise with the laboratory 
and to check that they are satisfied with the way cars are set up for testing.  Where a Vehicle 
Manufacturer feels that a particular item should be altered, they should ask the laboratory staff to 
make any necessary changes.  Vehicle Manufacturers are forbidden from making changes to any 
parameter that will influence the test, such as dummy positioning, vehicle setting, laboratory 
environment etc. 
It is the responsibility of the test laboratory to ensure that any requested changes satisfy the 
requirements of Euro NCAP.  Where a disagreement exists between the laboratory and Vehicle 
Manufacturer, the Euro NCAP secretariat should be informed immediately to pass final judgment.  
Where the laboratory staff suspect that a Vehicle Manufacturer has interfered with any of the set 
up, the Vehicle Manufacturer's representative should be warned that they are not allowed to do 
so themselves.  They should also be informed that if another incident occurs, they will be asked 
to leave the test site. 
Where there is a recurrence of the problem, the Vehicle Manufacturer’s  ep esent tive will be told 
to leave the test site and the Secretary General should be immediately informed.  Any such 
incident may be reported by the Secretary General to the Vehicle Manufacturer and the person 
concerned may not be allowed to attend further Euro NCAP tests. 
DISCLAIMER: Euro NCAP has taken all reasonable care to ensure that the information published 
in this protocol is accurate and reflects the technical decisions taken by the organisation.  In the 
unlikely event that this protocol contains a typographical error or any other inaccuracy, 
Euro NCAP reserves the right to make corrections and determine the assessment and 
subsequent result of the affected requirement(s). 
 
 
 


## Page 4

 
 
CONTENTS 
DEFINITIONS 
3 
Speed Assistance 
3 
Vehicle Assistance 
4 
SCORING 
6 
1 SPEED ASSISTANCE 
7 
 General Requirements 
7 
 Speed Limit Information Function 
8 
 Speed Control Function 
12 
2 ADAPTIVE CRUISE CONTROL PERFORMANCE 
13 
 Car-to-Car 
15 
 Car-to-PTW 
19 
 Car-to-VRU 
24 
 Road Features 
25 
 Auto-Resume 
26 
3 STEERING ASSISTANCE 
27 
 Steering Assistance 
27 
 Lane Change Assist 
28 
 


## Page 5

 
Euro NCAP 
Version 1.1 — October 2025 
3 
DEFINITIONS 
Throughout this protocol the following terms are used:  
Journey – A journey starts with activation of the master control switch and lasts until the 
de ctiv tion of the m ste  cont ol switch  nd the d ive ’s doo  being opened. 
Vehicle master control switch – Me ns the device by which the vehicle’s on-board electronics 
system is brought from being switched off, as in the case where a vehicle is parked without the 
driver being present, to a normal operation mode. 
Default-ON – A function that is ON by default at the start of every journey. It may be voluntarily 
switched OFF by the driver, but voluntary function deactivation shall not be possible with a single 
momentary push of a button. 
 
Speed Assistance 
Vindicated – The speed at which the vehicle travels as displayed to the driver by the 
speedometer as in ECE R39. 
Vlimit – Maximum allowed legal speed for the vehicle at the location, time and in the 
circumstance the vehicle is driving. 
Speed Limit Information Function (SLIF) – SLIF means a function with which the vehicle 
knows and communicates the speed limit. 
Speed Limit Warning Function (SLWF) – SLWF means a function that alerts the driver that 
the Vindicated is exceeding the perceived speed limit. 
Speed Control Function (SCF) – a system which  ctively limits o  cont ols the vehicle’s speed 
to prevent exceeding the applicable speed limit. 
Adjustable speed (Vadj) – Adjustable speed Vadj means the voluntarily set speed for the 
speed control functions, which is based on Vindicated and includes the offset set by the driver. 
Speed Limitation Function (SLF) – SLF means a system which allows the driver to set a 
vehicle speed Vadj, to which he wishes the speed of his car to be limited and above which he 
wishes to be warned. 
Intelligent Speed Limiter (ISL) – ISL is a SLF combined with SLIF, where the Vadj is set by 
the SLIF with or without driver confirmation.  
Intelligent Adaptive Cruise Control (iACC) – iACC is an ACC combined with SLIF, where the 
speed is set by the SLIF with or without driver confirmation. 
Stabilised speed (Vstab) – Stabilised speed Vstab means the mean actual vehicle speed 
when operating. Vstab is calculated as the average actual vehicle speed over a time interval of 
20 seconds beginning 10 seconds after first reaching Vadj – 10 km/h. 
 
 
 


## Page 6

 
Euro NCAP 
Version 1.1 — October 2025 
4 
Vehicle Assistance  
Vehicle under test (VUT) – means the vehicle tested according to this protocol with a pre- 
crash collision mitigation or avoidance system on board. 
Global Vehicle Target (GVT) – means the vehicle target used in this protocol as defined in ISO 
19206-3:2021.  
Secondary Other Vehicle (SOV) – me ns the “L  ge Obst uction Vehicle”  s defined in the 
latest Crash Avoidance - Frontal Collisions protocol (and not a robot-controlled platform) used in 
the Cut-out test in this protocol. 
Euro NCAP Pedestrian Target (EPTa) – means the articulated adult pedestrian target used in 
this protocol as specified in the ISO 19206-2:2018  
Euro NCAP Bicyclist Target (EBTa) – means the adult bicyclist and bike target used in this 
protocol as specified in ISO 19206-4:2020   
Euro NCAP Motorcyclist Target (EMT) – means the Motorcyclist target used in this protocol 
as specified in the deliverable D2.1 of the MUSE project (Fritz and Wimmer 2019) which at time 
of publication is to be replaced with ISO 19206-5. 
Real Motorcycle – Means a motorcyclist target that can be used in the Blind-Spot Monitoring 
Tests of this protocol, as an alternative to the EMT. The Real Motorcycle shall be a type 
approved two-wheeled motorcycle, with a maximum speed of at least 80km/h by design, without 
front fairing or windshield. It shall closely resemble the EMT (as specified in section 2.1 of 
deliverable D2.1 of the MUSE project), thus staying within the mean dimensions of the most 
registered middleweight naked  motorcycles in Europe (i.e. wheelbase >1405mm. and 
<1445mm.). 
Time To Collision (TTC) – means the remaining time before the VUT strikes the GVT, 
assuming that the VUT and GVT would continue to travel with the speed it is travelling. 
Speed Assist System (SAS) –a system that informs or warns the driver and/or controls the 
vehicle speed 
Speed Limit Information Function (SLIF) – a function with which the vehicle knows and 
communicates the speed limit. 
Speed Limitation Function (SCF) – a system which allows the driver to set a vehicle speed to 
which he wishes the speed of his car to be limited and above which he wishes to be warned. 
Adaptive Cruise Control (ACC) – a system that controls the vehicle speed whilst maintaining a 
set distance to vehicles ahead 
Intelligent Adaptive Cruise Control (iACC) – iACC is an ACC combined with SLIF, where the 
speed is set by the SLIF with or without driver confirmation. 
Autonomous Emergency Braking (AEB) – braking that is applied automatically by the vehicle 
in response to the detection of a likely collision to reduce the vehicle speed and potentially avoid 
the collision. 
Autonomous Emergency Steering (AES) – steering that is applied automatically by the 
vehicle in response to the detection of a likely collision to steer the vehicle around a target in 
front to avoid the collision. 


## Page 7

 
Euro NCAP 
Version 1.1 — October 2025 
5 
Forward Collision Warning (FCW) – an audio-visual warning that is provided automatically by 
the vehicle in response to the detection of a likely collision to alert the driver. 
Lane Support System (LSS) – a set of lateral control features that correct the vehicle heading 
to keep the vehicle within its driving lane and/or warns the driver. 
Lane Centering (LC) – a function which assists the driver in keeping the vehicle within the 
chosen lane, by influencing the lateral movement of the vehicle. 
Lane Change Assist (LCA) – a function which is initiated by the driver OR proposed by the 
system and confirmed by the driver, which can perform a single lateral manoeuvre (e.g. lane 
change). 
Emergency Lane Keeping (ELK) – default ON heading correction that is applied automatically 
by the vehicle in response to the detection of the vehicle that is about to drift beyond a solid lane 
marking, the edge of the road or into oncoming or overtaking traffic in the adjacent lane. 
Lane Keeping Assist (LKA) – heading correction that is applied automatically by the vehicle in 
response to the detection of the vehicle that is about to drift beyond a delineated edge line of 
the current travel lane. 
Lane Departure Warning (LDW) – a warning that is provided automatically by the vehicle in 
response to the vehicle that is about to drift beyond a delineated edge line of the current travel 
lane. 
Driver State Monitoring (DSM) – Driver State Monitoring system that is able to (in)directly 
determine the state of the driver  
Direct Monitoring – Where driver state determination is supported by sensor(s) directly 
observing the driver. 
Car-to-Car – a collision between a vehicle and another car, when no braking and/or steering 
action is applied. 
Car-to-Pedestrian – a collision between a vehicle and an adult or child pedestrian in its path, 
when no braking and/or steering action is applied. 
Car-to-Bicyclist – a collision between a vehicle and an adult bicyclist in its path, when no 
braking and/or steering is applied. 
Car-to-Motorcyclist – a collision between a vehicle and a Motorcyclist in its path, when no 
braking and/or steering is applied. 
 
 


## Page 8

 
Euro NCAP 
Version 1.1 — October 2025 
6 
SCORING  
Vehicle Assistance assessment 
Total points 40 
Speed Assistance 
20 
Speed Limit Information Function  
12 
Speed Control Function  
8 
ACC Performance 
15 
Car-to-Car 
6 
Car-to-PTW  
5 
Car-to-VRU 
2 
Road Features 
1 
Auto-resume 
1 
Steering Assistance 
5 
Steering Assistance 
4 
Lane Change Assist  
1 
 
 
 


## Page 9

 
Euro NCAP 
Version 1.1 — October 2025 
7 
1 SPEED ASSISTANCE 
Speed Assistance assessment 
Total points 20 
Speed Limit Information Function 
12 
SLIF Accuracy 
4 
Advanced Speed Limits 
3 
Local Hazards 
3 
System updates 
2 
Speed Control Function 
8 
ISL not default-ON 
5 
iACC 
8 
 
The Speed Limit Information Function may be a standalone function or an integrated part of a 
Speed Control Function. Any SLIF using all relevant system inputs, e.g., camera input and 
electronic map based or a combination of both, is eligible for scoring points for Advanced 
Functions when meeting the General Requirements.  
The Vehicle Manufacturer shall supply Euro NCAP with a dossier containing background 
information of the SLIF (if applicable to the technology). 
 
 General Requirements 
The SLIF, including the Local Hazard warnings*, shall be default ON at the start of a journey and 
shall be shown at all times (excluding the initialization period and temporary interruption for safety 
reasons).  
The speed limit shall be shown using a traffic sign and shall be clearly seen in the direct field of 
view of the driver, without the need for the head to be moved from the normal driving position, 
e.g. instrument cluster or head-up display. 
In the presence of explicit conditional speed limits the system shall either: 
- 
Identify and show (for example when raining) the applicable speed limit,  
OR  
- 
Indicate the presence of a conditional speed limit which the system is not able to 
compute, in addition to the non-conditional speed limit. 
The SLIF shall incorporate a default ON visual warning informing the driver when Vlimit is 
exceeded. The visual warning shall be a flashing traffic sign used to communicate the speed limit 
or an additional visual signal adjacent to the traffic sign.  
*User consent may apply. 


## Page 10

 
Euro NCAP 
Version 1.1 — October 2025 
8 
 
 Speed Limit Information Function 
Speed Limit Information Function 
Total Points 12 
SLIF Accuracy 
4 
Advanced Speed Limits 
3 
Local Hazards 
3 
System updates 
2 
 
1.2.1 SLIF accuracy 
SLIF Accuracy KPI 
Requirement 
Points 
Distance based (KPIDistance) 
> 80%* 
2 
Event based (KPIEvent) 
> 80 -10%* 
2 
* Assuming perfect ground truth.  
 
SLIF accuracy is defined by two KPIs: Distance-based and Event-based: 
𝐾𝑃𝐼𝐷𝑖𝑠𝑡𝑎𝑛𝑐𝑒= 𝐷𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝐷𝑡𝑜𝑡𝑎𝑙
⁄
 
with: 
Dcorrect = Total distance with correct speed limit displayed (km),applicable to ALL speed limit elements 
Dtotal = Total distance driven (km) 
𝐾𝑃𝐼𝐸𝑣𝑒𝑛𝑡= 𝐸𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝐸𝑡𝑜𝑡𝑎𝑙
⁄
 
with: 
Ecorrect = Total number of correctly identified events of the claimed speed limit elements 
Etotal = Total number of events  of the claimed speed limit elements 
 
Both KPIs shall be determined through an on-road evaluation conducted by the test laboratory 
on public roads. The route shall be defined by Euro NCAP and have a total length of at least 
2000 km, covering between three and five countries within the Euro NCAP Application Area as 
defined in Technical Bulletin G 001. The mileage shall be distributed such that at least 20% of 
the total distance is driven in each country. A minimum of 10% of the total mileage shall be 
conducted during nighttime (i.e., after sunset). The route shall include (per country) between 5% 
and 10% urban roads, between 20% and 30% interurban roads, and at least 60% highway 
driving. 


## Page 11

 
Euro NCAP 
Version 1.1 — October 2025 
9 
1.2.2 Advanced Speed Limits 
Advanced Speed Limits 
Points 
Conditional Speed Limits 
2 
Implicit Speed Limits 
0.5 
Dynamic Speed Limits 
0.5 
 
To be eligible for points in each advanced speed limit, the Vehicle Manufacturer shall demonstrate 
by means of a dossier (following the provisions on SD-301) that the system provides the driver 
with advanced speed limit information  during at least 80% of typical driving on the following 
areas: 
- 
Austria, France, Germany, Italy, Luxemburg, the Netherlands, Spain, Sweden, 
United Kingdom and Norway. 
- 
In at least half of the countries of the Euro NCAP Application Area (as defined in 
G 001). 
The dossier shall contain evidence of the system performance for each advanced speed limit 
resulting from on-road evaluation conducted by the Vehicle Manufacturer of approximately 400km 
of length in above areas, with exceptions allowed for small countries. Alternative validation 
methods may be used when on-road evaluation is not feasible or sufficient e.g., HiL data, test 
track data, etc. 
1.2.2.1 Conditional speed limits 
Conditional Speed Limits 
Requirement 
Points 
Rain/wetness (including implicit) 
Show correct speed limit 
0.4 
Snow/icy 
Warning only / ignore if irrelevant 
0.4 
Time/season 
Show correct speed limit 
0.4 
Distance for/in 
Show correct speed limit 
0.4 
Arrows 
- 
Non lane-relevant 
- 
Lane-relevant 
Show correct speed limit  
 
0.1 
0.2 
Vehicle categories 
Show correct speed limit / ignore if 
irrelevant 
0.2 
 
Systems that can identify and compute conditions and show the applicable speed limit 
accordingly are eligible to score the available points. The speed limit under these conditions shall 
not be shown separately from the speed limit information requested in the general requirements. 


## Page 12

 
Euro NCAP 
Version 1.1 — October 2025 
10 
1.2.2.2 Implicit speed limits 
Implicit Speed Limits 
Requirement 
Points 
Highway / Motorway 
City Entry / Exit 
Residential zones 
Show correct speed limit* 
0.5 
* Applicable to ANY implicit speed limit 
1.2.2.3 Dynamic speed limits 
Dynamic Speed Limits 
Requirement 
Points 
Dynamic speed signs including 
roadworks 
- 
Non lane-relevant 
- 
Lane-relevant 
Show correct speed limit 
 
0.25 
0.5 
 
1.2.3 Local Hazards 
Local Hazards 
Direct OR Cloud 
Communication 
Direct AND Cloud  
Communication 
Sending 
Receiving & 
informing 
Sending 
Receiving & 
informing 
Construction zones 
0.15 
0.15 
0.2 
0.15 
Items on road 
0.15 
0.15 
0.2 
0.15 
Stopped vehicle* 
0.15 
0.15 
0.2 
0.15 
Broken down vehicle* 
0.15 
0.15 
0.2 
0.15 
Post crash* 
0.15 
0.15 
0.2 
0.15 
Poor weather* 
0.15 
0.15 
0.2 
0.15 
Poor road*  
0.15 
0.15 
0.2 
0.15 
Wrong way driver* 
0.15 
0.15 
0.2 
0.15 
Amber + Blue lights 
N/A 
0.15 
N/A 
0.15 
Traffic jam 
N/A 
0.15 
N/A 
0.15 
TOTAL (capped) 
Max 2.5 
Max 3.0 
*When sending information, only information about the condition of the ego vehicle, or 
environmental conditions the ego vehicle is exposed to, is requested  


## Page 13

 
Euro NCAP 
Version 1.1 — October 2025 
11 
Vehicles able to send AND receive local hazard information are eligible to score the available 
points shown in the table above. Points can be scored individually. Local hazards service shall 
be available in all Euro NCAP Application Area (as defined in TB002). 
Vehicles may communicate with a public cloud or via direct communication. Maximum points are 
achieved when both cloud and direct communication is possible. 
“Receiving  nd info ming” is understood as retrieving local hazard data into the vehicle and 
informing the driver about them in due time before reaching the event location.  
“Sending” is understood as sharing local hazard data that is gathered by the vehicle within the 
DFRS cloud ecosystem or via direct communication. Driver-reported traffic events are not eligible 
fo  “sending” points. 
 
1.2.3.1 Cloud communication 
Cloud communication is foreseen to happen via mobile network. The reference cloud for this 
communication channel is the Data For Road Safety (DFRS) ecosystem 
[https://www.dataforroadsafety.eu/]. 
For each Local Hazard covered by the vehicle, the Vehicle Manufacturer shall demonstrate, by 
means of fulfilling the self-declaration forms developed by DFRS, that vehicle data is received 
and/or sent from/to the DFRS ecosystem. 
1.2.3.2 Direct Communication 
Direct Communication is foreseen to happen via direct short range communication standards 
(e.g., Wi-Fi ITS-G5 or C-V2X) 
- 
If ITS-G5 is used, the Vehicle Manufacturer shall self-declare fulfilment of the 
direct short range communication standards and demonstrate interoperability 
with C-ROADS deployment 
- 
If a different approach is followed, the Vehicle Manufacturer shall contact the 
Euro NCAP Secretariat.  
 
1.2.4 System updates 
System Updates 
Points 
Continuous connectivity (Streamed) 
2 
Temporary connectivity (OTA updates) 
1 
 
1.2.4.1 Continuous connectivity 
Vehicles that continuously stream speed limit data while driving. 
1.2.4.2 Temporary connectivity 
Regular updates for speed limit data over the air, at least quarterly.  


## Page 14

 
Euro NCAP 
Version 1.1 — October 2025 
12 
 
Speed Control Function 
Speed Control Function 
Points 
Intelligent Speed Limiter (ISL) 
5 
Intelligent Adaptive Cruise Control (iACC) 
8 
 
The speed control function shall be capable of being activated/de-activated at any time with a 
simple operation. Functionalities above GSR ISA requirements could be configurable by the 
driver, without the need of being default ON. 
To be awarded full score, speedometer accuracy shall be -3/+0 km/h. When the speedometer 
accuracy is -5/+0 km/h the SCF points are halved. 
1.3.1 Setting the speed 
The Speed Control Function (SCF) shall use the speed limit information from the SLIF to set the 
Vadj, with or without driver confirmation (to the choice of the Vehicle Manufacturer). The system 
should adopt, or offer the driver to adopt, an adjusted Vadj within 5s after a change in the speed 
limit.  
A negative and/or positive offset with respect to the known speed limit is allowed but may not be 
larger than 10 km/h (5 mph). This offset is included in Vadj. 
1.3.2 Speed Control 
The vehicle speed shall be limited or controlled to Vadj, but it shall still be possible to exceed Vadj 
by applying a positive action – e.g. pressing the accelerator harder/deeper or kickdown. 
After exceeding Vadj by applying a positive action, the speed control function shall be reactivated 
when the vehicle speed drops to a speed less than or equal to Vadj.  
If the Vadj is set to a speed lower than the current vehicle speed, the SCF shall start reducing the 
vehicle speed to the new Vadj, or shall initiate a warning no later than 30s after Vadj has been 
set.  
If the Vadj is set to a speed higher than the current vehicle speed, the SCF shall start increasing 
the vehicle speed to the new Vadj, when traffic conditions allow (for iACC only, when fitted). 
 
When the speed control function is not able to limit to and/or maintain Vadj and Vadj is exceeded, 
an acoustic warning shall be issued. No warning needs to be given when Vadj is exceeded as a 
result of a positive action. For systems where active braking is applied to maintain and/or limit the 
speed, this warning requirement does not apply. 
 


## Page 15

 
Euro NCAP 
Version 1.1 — October 2025 
13 
2 ADAPTIVE CRUISE CONTROL PERFORMANCE 
ACC Performance 
Total points 15 
Car-to-car 
6 
Longitudinal 
4 
Cut-in / Cut-out 
2 
Car-to-PTW  
5 
Longitudinal 
4 
Cut-in / Cut-out 
1 
Car-to-VRU 
2 
Longitudinal 
2 
Road Features 
1 
Auto-resume 
1 
 
Only the capability of the ACC system is assessed, where acceleration ≥-5 m/s2. 
All ACC tests are performed as per Euro NCAP Crash Avoidance protocols however, where the 
procedure in this protocol deviates from these protocols, this ACC protocol should be followed. 
For each test, the vehicle shall be driven in a fully marked lane with the indicated ACC speed set 
to the required test speed (i.e., not the GPS speed). The ACC shall be initially set to the closest 
following distance for all tests. Where possible, the Steering Assistance shall be engaged and 
used to control the VUT’s position within the l ne. When this system is not available, the vehicle 
will be driven manually. The ACC shall be active before the lower of 10s TTC or 250m relative 
longitudinal distance to target. 
The Vehicle Manufacturer is required to provide the Euro NCAP Secretariat with colour data 
(expected impact speeds are not required) detailing the ACC performance in all scenarios 
included in the assessment, as indicated in the table below: 
 
 


## Page 16

 
Euro NCAP 
Version 1.1 — October 2025 
14 
Colour 
 
Expected ACC performance ( ≥-5 m/s2) 
Green 
Car-to-Car 
Full avoidance 
Car-to-PTW 
Full avoidance 
Car-to-VRU 
Speed reduction > 30 km/h 
Orange 
Car-to-Car 
Speed reduction > 15 km/h 
Car-to-PTW 
Speed reduction > 15 km/h 
Car-to-VRU 
Speed reduction > 15 km/h 
Grey 
Car-to-Car 
Speed reduction ≤ 15 km/h 
Car-to-PTW 
Speed reduction ≤ 15 km/h 
Car-to-VRU 
Speed reduction ≤ 15 km/h  
 
Test selection 
For CCRs, CCRm, CMRs and CMRm, based on the Vehicle Manufacturer colour prediction, the 
following test speeds of each scenario will be tested by the test laboratory (excluding tests speeds 
where the Vehicle Manufacturer predicted Grey in the Safety Backup – Collision Avoidance tests 
of the Assisted Driving protocol):  
- 
Highest test speed with “G een” p ediction 
- 
Highest test speed with “O  nge” p ediction 
- 
One randomly selected test speed  
If the prediction performance is not met in any of these tests, perform a test at the adjacent test 
speed(s) until the predicted performance is confirmed. 
For Cut-in and Cut-out scenarios (both for Car-to-Car and Car-to-PTW), the test laboratory will 
test all test speeds 
In case the Vehicle Manufacturer does not provide performance data, the test laboratory will 
conduct all test speeds in all scenarios. 
 
Impact speed tolerance 
As test results can vary between labs and in-house tests and/or simulations a 2 km/h tolerance 
to the impact speeds of the verification test is applied. The tolerance is applied in both directions, 
meaning that when a tested point scores better than predicted, but within tolerance, the predicted 
result is applied.  
The tolerance only applies to verify whether the predicted colour of the tested verification point is 
correct. When, including tolerance, the colour is not in line with the prediction, the true colour of 
the test point will be determined by comparing the actual measured impact speed reduction with 
the colour band without applying a tolerance to the impact speed reduction.  


## Page 17

 
Euro NCAP 
Version 1.1 — October 2025 
15 
 
 Car-to-Car 
Car-to-car 
Total points 6 
Longitudinal 
4 
CCRs straight 
CCRs curve 
1 
1 
CCRm 
1 
CCRb 
1 
Cut-in / Cut-out 
2 
Cut-in 
1 
Cut-out 
1 
 
ACC Performance Car-to-Car tests consist of the following combination of VUT and target speeds 
with 10 km/h increments where applicable: 
Car-to-car tests 
VUT speed 
Target speed 
CCRs – Stationary Target 
Straight road, 50% impact location 
Curved road, 50% impact location 
 
60-130 km/h 
60-130 km/h 
 
0 km/h 
0 km/h 
CCRm – Moving Target 
Straight road, 50% impact location 
 
 
60-130 km/h 
70-130 km/h 
 
20 km/h 
60 km/h 
CCRb – Braking Target 
Straight road, 50% impact location 
 
55 km/h 
 
50 km/h & -4m/s2 
Cut-in 
Straight road, 50% impact location 
Cut-in @ TTC = 0.00s 
Cut-in @ TTC = 1.50s 
 
 
50 km/h 
120 km/h 
 
 
10 km/h 
70 km/h 
Cut-out 
Straight road, 50% impact location 
Cut-out @ TTC = 3.00s 
Cut-out @ TTC = 3.00s 
 
 
70 km/h 
90 km/h 
 
 
50 km/h 
70 km/h 
 
 
 


## Page 18

 
Euro NCAP 
Version 1.1 — October 2025 
16 
 
For each scenario and test speed, 1 point can be achieved where the ACC fully avoids the 
collision. Where the ACC intervenes and reduces the impact speed by more than 15 km/h before 
the AEB intervenes, 0.5 points are scored. Where the ACC does not reduce the impact speed 
more than 15 km/h, no points are awarded. 
2.1.1 CCRs 
CCRs tests are conducted on both straight and curved roads from 60 to 130 km/h in 10 km/h 
speed increments. Tests on straight roads are conducted with 50% impact location. 
For tests on a curved section of road, the first turn of the S-Bend as required for the Steering 
Assistance assessment is used where the GVT shall be positioned such that it is central in lane 
around the first bend so that the rear corner is touching the extrapolated line as if the straight 
were continue (as shown in the picture below). 
For vehicles not equipped with Lane Centering or where the VUT cannot follow the S-bend path, 
conduct the test manually following the S-Bend path and ensuring as much as possible a 50% 
impact location.  
Automatic speed reduction/adaptation prior to entering the S-Bend is allowed if the speed 
reduction/adaptation strategy is always active. 
 
 
2.1.2 CCRm 
CCRm tests are conducted on straight road with a VUT speed from 60 to 130 km/h in 10 km/h 
speed increments, and with combinations of target speeds of 20 and 60 km/h. All tests are 
conducted with 50% impact location. 
In the case of CCRm test cases where the GVT travels at 60km/h it is permissible to use a real 
vehicle of B-Segment fitted with data recording instrumentation. 
A physical vehicle shall only be used when full avoidance from the ACC system is predicted, I.e. 
deceleration levels do not exceed 5m/s2. The test shall be aborted safely if the VUT does not 
initiate ACC braking when TTC = [3.0s], at which point the test is repeated with the Soft Car 
GVT & platform. 
2.1.3 CCRb 
CCRb test is conducted with a VUT speed of 55 km/h and target speed of 50 km/h, with ACC set 
to closest distance, and with 50% impact location. The target shall decelerate at a rate of 4m/s2 
 
 
0 km/h 
60-130 km/h 


## Page 19

 
Euro NCAP 
Version 1.1 — October 2025 
17 
2.1.4 Cut-in 
In the Cut-in tests, the GVT on the adjacent lane shall perform a full lane change (3.5m lateral 
offset) into the lane of the VUT. The indicated TTC is defined as the TTC at the point in time when 
the GVT has finished the lane change manoeuvre, where the rear centre of the GVT is in the 
middle of the VUT driving lane. 
 
Cut-in  
VUT 
GVT 
Lane Change Manoeuvre GVT 
Lateral 
Acceleration 
Change 
Length 
Radius of turning 
segments 
Cut-in @ TTC = 0.00 
Cut-in @ TTC = 1.50 
 
50 km/h 
120 km/h 
 
10 km/h 
70 km/h 
 
0.5 m/s2 
1.5 m/s2 
 
14.5 m 
60.0 m 
 
15 m 
250 m 
 
 
 
2.1.5 Cut-out 
The Cut-out test shall be performed using the SOV. The vehicle cutting out (SOV) shall perform 
a full lane change (3.5m lateral offset) into the adjacent lane to avoid the stationary GVT. With the 
measurement behind the stationary GVT indicting that start of the lane change, and the 
measurement in front of the stationary GVT indicating the end of the lane change. The indicated 
TTC is defined as the TTC of the lead vehicle to the GVT when the lead vehicle shall start the 
lane change. Indicators are not to be used by the SOV during the manoeuvre. It is permissible 
for the test laboratory to place physical markers, that shall not affect vehicle performance, of the 
different cut-out paths. SOV path deviation = [±0.2m].  
 
Cut-out 
VUT 
Lead 
Vehicle 
Lane Change Manoeuvre of lead vehicle 
Lateral 
Acceleration 
Change  
Length 
Radius of 
turning 
segments 
Cut-out @ TTC = 3.00 
Cut-out @ TTC = 3.00 
 
70 km/h 
90 km/h 
 
50 km/h 
70 km/h 
 
1.5 m/s2 
1.5 m/s2 
 
44.0 m 
60.0 m 
 
130 m 
250 m 


## Page 20

 
Euro NCAP 
Version 1.1 — October 2025 
18 
 
 
 
 
 
 
 
 
 
 
 


## Page 21

 
Euro NCAP 
Version 1.1 — October 2025 
19 
 
Car-to-PTW 
Car-to-PTW 
Total points 5 
Longitudinal 
4 
CMRs straight 
CMRs curve 
1 
1 
CMRm 
1 
CMRb 
1 
Cut-in / Cut-out 
1 
Cut-in 
0.5 
Cut-out 
0.5 
 
ACC Performance Car-to-PTW tests consist of the following combination of VUT and target 
speeds with 10 km/h increments where applicable: 
Car-to-PTW tests 
VUT speed 
Target speed 
CMRs – Stationary Target 
Straight road, 25% impact location, GVT on side 
Straight road, 25% impact location, GVT in front 
Curved road, 50% impact location 
 
60-90 km/h 
60-90 km/h 
60-90 km/h 
 
0 km/h 
0 km/h 
0 km/h 
CMRm – Moving Target 
Straight road, 25% impact location 
 
 
60-130 km/h 
70-130 km/h 
 
20 km/h 
60 km/h 
CMRb – Braking Target 
Straight road, 25% impact location 
 
55 km/h 
 
50 km/h & -4m/s2 
Cut-in 
Straight road, 25% impact location 
Cut-in @ TTC = 0.50s 
Cut-in @ TTC = 1.50s 
 
 
50 km/h 
120 km/h 
 
 
10 km/h 
70 km/h 
Cut-out 
Straight road, 25% impact location 
Cut-out @ TTC = 3.00s 
Cut-out @ TTC = 3.00s 
 
 
70 km/h 
90 km/h 
 
 
50 km/h 
70 km/h 
 
 


## Page 22

 
Euro NCAP 
Version 1.1 — October 2025 
20 
 
For each scenario and test speed, 1 point can be achieved where the ACC fully avoids the 
collision. Where the ACC intervenes and reduces the impact speed by more than 15 km/h before 
the AEB intervenes, 0.5 points are scored. Where the ACC does not reduce the impact speed 
more than 15 km/h, no points are awarded. 
 
2.2.1 CMRs 
CMRs tests are conducted on both straight and curved roads from 60 to 90 km/h in 10 km/h 
speed increments.  
For tests on a straight road, the stationary EMT shall be positioned in a 25% impact location 
position. The test laboratory shall randomly select one of the following scenario layouts.  
a) A stationary GVT positioned in the adjacent lane such that the left side is 20 cm from the 
centre of the dashed lane marking of the VUT lane, and the rear side coincides with the 
rear wheel of the stationary EMT:  
 
 
 
 
b) A stationary GVT positioned centred in-lane and 1m in front of the EMT: 
 
 
 


## Page 23

 
Euro NCAP 
Version 1.1 — October 2025 
21 
For tests on a curved section of road, the first turn of the S-Bend as required for the Steering 
Assistance assessment is used where the EMT shall be positioned such that it is central in lane 
around the first bend, with the most rear part of the rear wheel is touching the extrapolated line 
as if the straight were continue (as shown in the picture below). 
For vehicles not equipped with Lane Centering or where the VUT cannot follow the S-bend path, 
conduct the test manually following the S-Bend path and ensuring as much as possible a 50% 
impact location.  
Automatic speed reduction/adaptation prior to entering the S-Bend is allowed if the speed 
reduction/adaptation strategy is always active. 
 
  
 
2.2.2 CMRm 
In the case of CMRm test cases where the EMT travels at 60km/h it is permissible to use a real 
motorcycle with data recording instrumentation.  
A real motorcycle shall only be used when full avoidance from the ACC system is predicted, i.e. 
deceleration levels do not exceed 5m/s2. The test shall be aborted safely if the VUT does not 
initiate ACC braking when TTC = [3.0s], at which point the test is repeated with the EMT. 
2.2.3 CMRb 
For CMRb, the test is conducted in the same way as CCRb, but with an EMT positioned at a 25% 
impact location.  
 
 
60-90 km/h 
0 km/h  


## Page 24

 
Euro NCAP 
Version 1.1 — October 2025 
22 
2.2.4 Cut-in 
In the Cut-in tests, the EMT on the adjacent lane shall perform a partial lane change (2.5m lateral 
offset) into the lane of the VUT. The indicated TTC is defined as the TTC at the point in  time that 
the EMT has finished the lane change manoeuvre, where the rear wheel of the EMT is in the 25% 
impact location  of the VUT. 
 
Cut-in 
VUT 
EMT 
Lane Change Manoeuvre EMT 
Lateral 
Acceleration 
Change 
Length 
Radius of 
turning 
segments 
Cut-in @ TTC = 0.50 
Cut-in @ TTC = 1.50 
 
50 km/h 
120 km/h 
 
10 km/h 
70 km/h 
 
0.5 m/s2 
1.5 m/s2 
 
14.5 m 
60.0 m 
 
15 m 
250 m 
 
 
 
To ensure a realistic trajectory and sufficient repeatability/reproducibility across different EMT 
platforms, the following EMT boundary conditions shall be met during the Lane Change length:  
• 
Path error/Lateral deviation [m]: +/- 0.15 
• 
Heading/Yaw angle deviation [°]: +/- 2.00 
• 
Speed deviation [km/h]: +/- 0.50 
 
 
 


## Page 25

 
Euro NCAP 
Version 1.1 — October 2025 
23 
2.2.5 Cut-out 
The Cut-out test shall be performed using the SOV. The vehicle cutting out (SOV) shall perform 
a full lane change (3.5m lateral offset) into the adjacent lane to avoid a stationary EMT positioned 
at a 25% impact location. With the measurement behind the stationary EMT indicting the start of 
the lane change, and the measurement in front of the stationary EMT indicating the end of the 
lane change. The indicated TTC is defined as the TTC of the lead vehicle to the EMT when the 
lead vehicle shall start the lane change. Indicators are not to be used by the SOV during the 
manoeuvre. It is permissible for the test laboratory to place physical markers, that shall not affect 
vehicle performance, of the different cut-out paths. SOV path deviation = [±0.2m].  
 
 
Cut-out 
VUT 
 
Lead 
Vehicle 
Lane Change Manoeuvre of lead vehicle 
Lateral 
Acceleration 
Change 
Length 
Radius of 
turning 
segments 
Cut-out @ TTC = 3.00 
Cut-out @ TTC = 3.00 
 
70 km/h 
90 km/h 
 
50 km/h 
70 km/h 
 
1.5 m/s2 
1.5 m/s2 
 
44.0 m 
60.0 m 
 
130 m 
250 m 
 
 
 
 
 
 
 
 


## Page 26

 
Euro NCAP 
Version 1.1 — October 2025 
24 
 
Car-to-VRU 
Car-to-VRU 
Total points 2 
Longitudinal 
2 
CPLA 
1 
CBLA 
1 
 
Car-to-VRU tests evaluate the ACC performance to pedestrians and bicyclists ahead travelling 
in the same direction, with an impact location of 0%. The CPLA-0 and CBLA-0 consist of the 
following combination of VUT and target speeds with 10 km/h increments: 
Car-to-VRU tests 
VUT speed 
Target speed 
CPLA – Moving Target 
Straight road, 0% impact location 
 
60-90 km/h 
 
5 km/h 
CBLA – Moving Target 
Straight road, 0% impact location 
 
60-90 km/h 
 
20 km/h 
 
For each scenario and test speed, 1 point can be achieved where the ACC reduces the impact 
speed by more than 30 km/h. Where the ACC intervenes and reduces the impact speed by more 
than 15 km/h before the AEB intervenes, 0.5 points are scored. Where the ACC does not reduce 
the impact speed more than 15 km/h, no points are awarded. 
 
 
A valid test run shall be considered when the 0% impact location  is achieved with a lateral offset 
accuracy of + 10cm – 0cm. 
The Vehicle Manufacturer may implement an early ACC speed reduction strategy linked to 
transient and/or non-transient driver states detected by a DSM. 
The Vehicle Manufacturer may implement avoidance by steering strategy when preceded by a 
[15] km/h speed reduction.  


## Page 27

 
Euro NCAP 
Version 1.1 — October 2025 
25 
 
Road Features 
Road Features 
Required action 
Total Points 1 
Curves 
Show and adjust the vehicle's speed to ensure that lateral 
acceleration does not exceed 3.5 + 0.5 m/s²  
0.2 
Roundabouts 
Show and start reducing speed so that [25] m before the 
roundabout, the vehicle’s speed is reduced to [50] km/h or 
lower 
0.2 
Intersection, 
no right-of-way 
Show and reduce speed to 30 km/h or lower if there is no 
driver response 
0.2 
Traffic lights 
For red lights, show and reduce speed to 30 km/h or lower 
if there is no driver response 
For orange lights, show and reduce speed to 30 km/h or 
lower 
if 
there 
is 
no 
driver 
response, 
provided 
emergency/harsh braking is not required e.g., the 
maximum deceleration does not exceed 5 m/s². 
0.2 
Stop signs 
Show and reduce speed to 30 km/h or lower if there is no 
driver response.  
0.2 
 
The reaction to road features is not required to be default ON. 
To avoid overreliance, Euro NCAP recommends that ACC speed adaptation to road features 
Curves, Roundabouts and Intersections may only be available for roads where the posted speed 
limit is 60 km/h or higher. It is assumed that traffic lights and stop signs are never placed at 
locations where the posted speed is more than 80 km/h. 
The road features functions shall be verified in the default ACC mode with activated road feature 
reaction during on-road driving to  confirm that the VUT responds as indicated by the Vehicle 
Manufacturer. 
 
 


## Page 28

 
Euro NCAP 
Version 1.1 — October 2025 
26 
 
Auto-Resume 
This assessment looks at the strategy to resume the ACC after the vehicle has come to a full 
stop. To be eligible for assessment, the VUT shall be capable of coming to a complete stop under 
ACC control when the traffic in front stops. 
ACC Auto-Resume 
Requirements 
Total Points 1 
Automatic resume 
All below requirements should be met: 
- Confirm surrounding with external sensors 
- Eyes on-road 
1 
Driver input 
Resume only after driver confirmation 
0.5 
 
With ACC active and following the GVT or other surrogate vehicle, decelerate the leading vehicle 
to a complete stop avoiding harsh decelerations. 
2.5.1.1 Confirm surrounding with external sensors 
After 5 seconds hold time, position a pedestrian dummy between the VUT and lead vehicle which 
after the lead vehicle shall drive off to confirm the VUT remains stopped. 
When confirmed, the pedestrian dummy should be removed and the VUT may resume driving. 
2.5.1.2 Eyes on-road 
After 5 seconds hold time, the driver shall look away from the forward road to after which the lead 
vehicle shall drive off to confirm the VUT remains stopped. 
When confirmed, the VUT may only resume driving after at least 0.5s of the driver looking back 
towards the forward road view. 
 
 
 


## Page 29

 
Euro NCAP 
Version 1.1 — October 2025 
27 
3 STEERING ASSISTANCE 
Steering Assistance Performance 
Total points 5 
Steering Assistance 
4 
Lane Change Assist  
1 
 
 
Steering Assistance 
Steering Assistance 
 
 
 
 
S-bend 
60 km/h 
80 km/h 
100 km/h 
130 km/h 
VUT stays in lane in both turns 
1 
1 
1 
1 
VUT stays in lane in 1st turn and 
redirects in 2nd turn 
0.5 
0.5 
0.5 
0.5 
 
A steering assistance function should support the driver to keep the vehicle in lane, not only on 
straight roads. If a car departs from its lane there is an increased risk of collision. Euro NCAP   
does not expect vehicles to stay in the centre of the lane in all corners, but expects the vehicle  
to always support the driver by directing the vehicle to the correct heading. Euro NCAP tests the 
steering assistance in a so called S-Bend. 
All tests shall be performed with longitudinal and lateral assistance activated. For test vehicles 
without longitudinal assistance available, the vehicle shall be controlled with driver input or using 
alternative control systems that can modulate the vehicle controls as necessary to perform the 
tests. 
 
3.1.1 S-Bend dimensions 
 
 
 


## Page 30

 
Euro NCAP 
Version 1.1 — October 2025 
28 
S-Bend 
Clothoid parameter 
Radius 
Length 
1st turn 
153.7 
 
30.0 
 
787 m 
57.1 
105.0 
 
14.0 
2nd turn 
98.6 
 
26.0 
 
374 m 
5.1 
120.8 
 
39.0 
 
It is permissible for an S-Bend to be used with the turn directions mirrored as long as the same 
geometry is maintained. 
 
3.1.2 Test Method 
The capability of the steering assist system is tested at ACC indicated vehicle speeds of 60, 80km/h, 
100km/h and 130km/h. Where possible, all other lane support systems shall be switched off for 
the duration of the test. 
The vehicle shall be driven along the straight section of the fully marked lane at a constant speed 
with the steering assist system on for enough time for the steering assist system to take up a 
constant position within the lane, prior to the start of the S-Bend. 
The driver shall make every effort not to add any input into the steering system which can affect 
the path of the vehicle once it has entered the S-Bend section. It is permissible for the test driver 
to remove their hands from the steering wheel. However, the driver may need to keep their hands 
on the wheel or provide a different input to prevent the actions of the vehicle being dictated by the 
systems recognition of an inattentive driver. 
The driver shall allow the vehicle to maintain a continuous maximum ACC speed as set 
throughout each test run. It is permissible for the vehicle system to reduce the driven speed in 
response to the road geometry, and this reduction in speed shall not be overridden by the test 
driver. It may also be the case that the curvature tested would cause the vehicle to slow 
sufficiently to remain within lane if it were on a mapped location (real world driving); if this is 
predicted to be the case the Vehicle Manufacturer shall advise the laboratory carrying out the test 
and confirm a suitable location to prove that the vehicle can slow and remain in lane. 
 
 
Lane Change Assist 
If the VUT is equipped with Lane Change Assist, i.e. a function which is initiated by the driver OR 
proposed by the system and confirmed by the driver, which can perform a single lateral 
manoeuvre (e.g. lane change), 1 point shall be awarded. 
 


