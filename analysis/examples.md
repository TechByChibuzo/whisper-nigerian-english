# Examples: zero-shot Whisper-base vs. Whisper-small, full FT, 13 h


## Big fixes

- [yoruba, general]
  - REF: Meanwhile, stay away from social media.
  - ZERO-SHOT: Me and Wai, come and stay here from, so shall we do a full stop?
  - FINE-TUNED: Meanwhile, stay away from social media.

- [igbo, general]
  - REF: But new acquisitions will also be required.
  - ZERO-SHOT: We will do a position before solving the query. First of all, I want to buy a one ASM.
  - FINE-TUNED: For new acquisitions with personal required.

- [igbo, clinical]
  - REF: Wash thermometer in lukewarm, soapy water.
  - ZERO-SHOT: Watch the moment in Lyu Kong, com a super water first up.
  - FINE-TUNED: Wash thermometer in lukewarm, soapy water.

- [yoruba, clinical]
  - REF: Elevated CK: trop <0.01 and CK 187.
  - ZERO-SHOT: Elivated Seaky chop, less than 0, 0, 1 and Seaky 1, it's 7.
  - FINE-TUNED: Elevated CK TROP LESTON 001 and CK 187.

- [yoruba, clinical]
  - REF: CA: a cancer journal for clinicians.
  - ZERO-SHOT: see a colon. It can't start a journal for clinicians first of all.
  - FINE-TUNED: CA colon a cancer journal for clinicians.

- [yoruba, general]
  - REF: But this year, 63 percent were black and 37 percent were white.
  - ZERO-SHOT: Body is so common, if you are 3% with black, then 7% white will stop.
  - FINE-TUNED: But this year, 63% were black and 37% were white.

- [hausa, clinical]
  - REF: It has a superior and inferior ramus and a triangular body.
  - ZERO-SHOT: It has a video and it is a video that I will send it via Google and I will be the first to
  - FINE-TUNED: It has an inferior and inferior armors and a triangular body.

- [igbo, clinical]
  - REF: No acute distress  Eyes / Conjunctiva: PERRL
  - ZERO-SHOT: No IQ distress ice slash conjunctiva. Send me column P-E-R-R-L.
  - FINE-TUNED: No acute distress eyes s.c.semi-perrl.

- [yoruba, clinical]
  - REF: TABLET, ORAL BACLOFEN, BACLOFEN, 10MG
  - ZERO-SHOT: tablet or a bottle of fenn, bottle of fenn 10mg.
  - FINE-TUNED: TABLET, ORAL BACLOFEN, BACLOFEN, 10MG

- [igbo, clinical]
  - REF: Battery, rechargeable from a solar panel.
  - ZERO-SHOT: battery, come on, return your book from his solar panel, first of all.
  - FINE-TUNED: Battery, rechargable from a solar panel.


## Hallucinations cured

- [yoruba, general]
  - REF: As an adult, Simon has become a successful solicitor, taking class actions in relation to Catholic Church and institutional sexual abuse in Australia.
  - ZERO-SHOT: The United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the
  - FINE-TUNED: As an adult, Simon has become a successful solicitor, taking class actions in relation to Catholic Church and institutional sexual abuse in Australia.

- [yoruba, general]
  - REF: Next line. Plan: next line. Tablets Coartem, 4 tabs twice daily for 3 days and paracetamol 1gm 3x daily for 3 days. Next line. Signed: Dr Ugochinyere Zikoranaudodimma
  - ZERO-SHOT: Next time, first of all, plan colon next time, first of all, tablet, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, quad-tem, q
  - FINE-TUNED: Next line. Plan  next line. Tablet  4  time, 4 tabs  twice daily for 3 days and paracetamol 1gm 3 times daily for 3 days. Next line. Signed  Dr Ogochiniri Zikoran Wadolima

- [yoruba, general]
  - REF: Margif, in his acceptance speech after being declared winner, said he is convinced that he will be the next Governor of Plateau State after Lalong, saying the party will work together to ensure the success of the election come 2023.
  - ZERO-SHOT: My GIF-coma in the S.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.T.A.R.
  - FINE-TUNED: Magif, in his acceptance speech after being declared winner, said he is convinced that he will be the next governor of the state after Lallung, saying the party will work together to ensure the success of the election come 2023.

- [hausa, general]
  - REF: On Tuesday evening, Modi announced an economic package which along with previous packages and RBI reprieves will amount to Rs 20 lakh crore, which he claimed will address the issues pertaining to land, labour, liquidity and loss.
  - ZERO-SHOT: On Tuesday evening, Coma Modi announced an economic package which along with previous packages and our BI reprisks will amounts to RS20 Lakh Kauri, Coma, which he claimed will address the issue pertaining to Lakh, Lakh, Lakh, Lakh, Lakh, Lakh, Lakh, 
  - FINE-TUNED: On Tuesday evening, Modi announced an economic package which along with previous packages and RBI reprieves will amnst to RS20 lakh crore, which he claimed will address the issue pertaining to land, labour, liquidity and loss.

- [hausa, general]
  - REF: He claimed that survey results were accurate since the margin of error was between three and four percent, adding that any statistical deviation within that range was accepted internationally for such surveys.
  - ZERO-SHOT: It claimed that several results were accurate since the margin of error was between 3 and 4% from a adding that any statistical division within the range was accepted internationally for soft soft soft soft soft soft soft soft soft soft soft soft sof
  - FINE-TUNED: He claimed that several results were accurate since the margin of error was between 3 and 4%, adding that any statistical deviation within that range was accepted internationally for subase.


## Still wrong after fine-tuning

- [hausa, clinical]
  - REF: Vanco q 12hrs iv.
  - ZERO-SHOT: So 1 for Q. 12 hours again. 12 for Q. 12 hours I use.
  - FINE-TUNED: Ubanco Q 12 hours a day.Ubanco Q 12 hours a day.

- [igbo, clinical]
  - REF: erefore,thediagnostictestofchoiceis stillthe sweat test.
  - ZERO-SHOT: Therefore, comma, the diagnostic test of choice is still the sweat test full stop.
  - FINE-TUNED: Therefore, the diagnostic test of choice is still the sweat test.

- [yoruba, clinical]
  - REF: : Extended transthoracicresectioncomparedwithlimitedtranshiatal resection for adenocarcinoma of the esophagus.
  - ZERO-SHOT: Cologne extended trans-doracic recession compared with limited trans-hiter recession for adenocal cinema of the suffagos.
  - FINE-TUNED: Colon Extended trans thoracic recession compared with limited trans-hital recession for adenocal synomers of the esophagus.

- [igbo, general]
  - REF: Wale Adebanwi on Mama HID (63)
  - ZERO-SHOT: Wale Adébanwe, Omar Mahid, ID Open brackets, 63, close brackets.
  - FINE-TUNED: Wale Adebanwe on Mama HID open bracket 63 closed bracket.

- [yoruba, clinical]
  - REF: Received hydralazine x1 for sbp > 180.
  - ZERO-SHOT: received hydrologs in one time for systolic blood pressure that had a larger than 8.
  - FINE-TUNED: Received Hx 1x for systolic blood pressure greater than 180.

- [yoruba, clinical]
  - REF: GI: ABD SOFT NT/ND W/+HYPOACTIVE BS.
  - ZERO-SHOT: gastrointestinal, abdomen, soaps, non-tender, non-distended with positive, hyperactive, buuil, sanat.
  - FINE-TUNED: Gastrointestinal abd soft non tender non distended with positive hyperactive borosound.

- [yoruba, clinical]
  - REF: 2019-02-01 ( 4 of 4 bottles ) E.coli
  - ZERO-SHOT: 2019-2012 for all four bottles equally.
  - FINE-TUNED: 2019-02-01 4 of 4 bottles equally

- [yoruba, clinical]
  - REF: NOSPITS MAX ASPIRATE3.6CC.
  - ZERO-SHOT: No speeds, mass, aspirates.ccc
  - FINE-TUNED: No spits, mass aspirate.ccc

- [igbo, general]
  - REF: You would need these ingredients:
  - ZERO-SHOT: You will need to increase that volume.
  - FINE-TUNED: You will need this in breathing column.

- [hausa, clinical]
  - REF: A: aga P: cont to support development4 cont on antibiotics bld cx pnd.
  - ZERO-SHOT: Accessments are prepared for gestation and plan. Continue to support developed plans for continued unantibertics, blood, culture, PND.
  - FINE-TUNED: Assessment Appropriate for gestational aid Plan


## Fine-tuning made it worse

- [yoruba, general]
  - REF: Judgment against NBC, vindication of our position IPC
  - ZERO-SHOT: judgment against NBC, communication of our position IPC.
  - FINE-TUNED: CHOGMENT AGAINNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN

- [igbo, clinical]
  - REF: Neck hematoma secondary to pulmonary artery line placement requiring transfusion, catheter tip infection .
  - ZERO-SHOT: Make hematoma secondary to pulmonary artery, line placement requiring transfusion, catheter, tip infection.
  - FINE-TUNED: NEC HEMA TOMA, SECOND RETUPUMONAR ARTUE, LINE PLACEMENT REQUIRING TRANSFUTION, CATHETIC TIPT INFECTION

- [yoruba, general]
  - REF: While performing Yope song, the newly crowned 2019 African Player of the Year Sadio Mane could not help but joined Diamond on the dance floor a situation which highly sparked the entire occasion.
  - ZERO-SHOT: While performing Yubesong Coma, the newly crowned 2019 Africa player of the year's Sadiw Mani could not help but join the diamond on the stance floor, a situation which I list back the international occasion for stock.
  - FINE-TUNED: While performing

- [igbo, clinical]
  - REF: However, other associated focal signs like diplopia, ipsilateral ataxia, facial paralysis and homonymous hemianopia indicate that the lesion is in the vascular system rather than in the inner ear.
  - ZERO-SHOT: However, other associated focal signs like the Eplopia, Epsilatra, Attexia, coma, facial paralysis, and homonymos, hemelnuper, indicate that the lesion is in the vasculous system rather than in the inner ear, full stop.
  - FINE-TUNED: However, other associated focal signs like theiplopia,                                                                                                         However, other associated focal signs like theiplopia,

- [yoruba, clinical]
  - REF: Etiology: Possibly autoimmune association. Extrahepatic biliary Possibly autoimmune association with other autoimmune diseases obstruction, biliary atresia with inflammatory  bowel disease 2.
  - ZERO-SHOT: etiology possibly autoimmune association, extra-apactypiary, possibly autoimmune association with other autoimmune diseases, obstruction of the area, atreasia with impromedribal disease, cheerful stuff.
  - FINE-TUNED: Iteology possibly autoimmune aso
