# Accepted-in-Principle (AIP) quality-control guideline

**Shared EMBO Press rules** for The EMBO Journal, EMBO Reports, EMBO Molecular Medicine and Molecular Systems Biology.

Version 1.1 (draft) · 30 September 2026 · report template: `aip-qc-report-template.md`
Journal rules verified against the EMBOJ, EMBOR, EMM, MSB and LSA author guidelines (September 2026).

This file is sent for every manuscript. The journal-specific file sent with it adds or overrides rules. If the two files conflict, the journal-specific file wins. Life Science Alliance rules are not in this file; they are in `life_science_alliance.md`.

---

## 1. Purpose and scope

A manuscript that is accepted in principle has already passed peer review on its science. The AIP check is the last pass before the manuscript goes to production. It asks one question: **does the submission package follow the journal's rules for files, metadata, structure, declarations, data availability and reporting?**

This guideline covers:

- the shared EMBO Press checklist (Section 5);
- how to assess each check and assign a status (Section 3);
- the exact report the reviewer (editor or model) must produce (Section 4).

Checks that exist only for one journal are in that journal's file, not here. For EMBO Press manuscripts those are C8 (EMBO Molecular Medicine only). Do not invent Life Science Alliance checks (A7, B2, B7) for an EMBO Press manuscript.

**In scope:** file format and packaging; submission-system metadata; manuscript structure and section naming; declarations and ethics statements; data, code and source-data availability; methods-reporting points editors routinely raise at this stage; figure callouts and legend housekeeping; references.

**Out of scope:**

- Scientific merit. This was settled in peer review.
- In-depth image and data integrity: manipulation, beautification, statistical correctness. These belong to the integrity check. Apparent duplications noticed during AIP QC are still reported (check G9).
- Commissioned and non-research article types. These follow their own article-type rules.

**Figure-level checks are owned by mmQC.** The benchmarked figure checklist ([source-data/mmQC](https://github.com/source-data/mmQC), `fig-checklist`) covers statistics, error bars, scale bars, axes and annotations in figures and legends. This guideline does not re-implement those checks. It takes their results as input and reports them under checks G4–G7 and G12 (Section 6).

---

## 2. Workflow

Follow these steps in order for every manuscript.

1. **Use the journal already selected for this review.** Apply this file plus the journal-specific file. Do not apply another journal's rules.
2. **Inventory the inputs** (list below). Record which ones are available in the report header.
3. **Run every check in Section 5, in order, plus every check in the journal-specific file.** Assign exactly one status per check (Section 3).
4. **Ingest the mmQC results**, if provided, into G4–G7 and G12 (Section 6).
5. **Write the report** exactly as specified in Section 4, using `aip-qc-report-template.md`.
6. **If this is a follow-up round**, take the previous report as input. Re-check every item that was not closed with `==>RESOLVED`, and keep its ID so the history can be followed.

### 2.1 Journal

The manuscript tracking number (MSID) has the form `<PREFIX>-<YEAR>-<NUMBER>[<REVISION SUFFIX>]`, e.g. `EMBOR-2025-62426V2`. Record the revision suffix exactly as it appears. Do not interpret it.

| MSID prefix | Journal | Rules |
|---|---|---|
| `EMBOJ` | The EMBO Journal | This file only |
| `EMBOR` | EMBO Reports | This file, plus the Reports profile note in the journal file |
| `EMM` | EMBO Molecular Medicine | This file, plus C8 and the C1 addition in the journal file |
| `MSB` | Molecular Systems Biology | This file only |
| `LSA` | Life Science Alliance | This file, overridden by `life_science_alliance.md` |

The pipeline has already chosen the journal-specific file. If that file says a check here does not apply, mark it `N/A`. If that file restates a check, use the journal file.

### 2.2 Inputs

| Input | Used for | If not available |
|---|---|---|
| Manuscript file (.docx; PDF only if nothing else) | Most checks in C, D, E, F, G, H | Report cannot be produced |
| Figure files, supplementary files, movies, tables as uploaded (file names and formats) | A-series, G1–G3, G8–G11 | Affected checks → `MANUAL` |
| Submission-system metadata (eJP / system export: title, running title, blurb, keywords, category, authors, corresponding authors, ORCID, funding, social handles, CRediT) | B-series, D1, D2 | Affected checks → `MANUAL` |
| mmQC output (JSON, per figure and check) | G4–G7, G12 | Affected checks → `MANUAL`, noted "mmQC not run" |
| Author checklist, point-by-point response (EMBO Press) | F7 | F7 → `MANUAL` |
| Previous AIP report (follow-up rounds) | Carrying over open items | Treat as first round |

---

## 3. How to assess a check

### 3.1 Status values

Every check gets exactly one of five statuses. Use these tokens verbatim; the report is parsed on them.

| Status | Meaning | Appears in Part 1 (action list)? |
|---|---|---|
| `PASS` | The check is met. | No |
| `ACTION` | A **required** rule is not met; the authors must fix it. | Yes |
| `ADVISE` | A **recommended** rule is not met; the authors are encouraged to fix it. | Yes |
| `N/A` | The check does not apply to this manuscript or journal (e.g. no animal work, no movies, or a check the journal file says does not apply). | No |
| `MANUAL` | The check cannot be completed from the available inputs; an editor must check it by hand. | Listed under "Check manually" |

Each check in Section 5 has a **Level**, and the level decides which status a failure gets:

| Level | Condition not met | Status |
|---|---|---|
| Required | Rule not met | `ACTION` |
| Required if applicable | Item absent | `N/A` |
| Required if applicable | Item present but rule not met | `ACTION` |
| Recommended | Rule not met | `ADVISE` |

The journal-specific file may change the level for that journal. For example, source data are required here, and only encouraged at Life Science Alliance.

### 3.2 Assessment rules

1. **Evidence, not assumption.** A `PASS` must rest on something actually seen. Every `ACTION` or `ADVISE` must say *where* the problem is: section heading, page or line, figure or panel, or file name.
2. **Never pass by default.** The checks in the watch list (Section 7) fail often. Inspect them explicitly.
3. **Partial checks.** Some checks have a part that can be verified and a part that cannot (e.g. title length is visible in the manuscript, but system consistency is not). Resolve them in this order:
   - If a problem is found in the verifiable part → `ACTION`. Add to the note what could not be verified.
   - If no problem is found but part of the check remains unverifiable → `MANUAL`. Note what was verified.
4. **One status per check, one row per check.** If a check fails in several places, list all locations in one finding (e.g. "Figs 2C, 4B, S3A"). Do not split it into several rows.
5. **Author wording.** For `ACTION` and `ADVISE` items, start from the standard wording given for the check and fill in every `[PLACEHOLDER]`. Adapt the wording only as far as needed to be accurate.
   - Required items start with "Please …".
   - Recommended items start with "We encourage you to …" or "We recommend that you …".
   - Where something is partly in place, acknowledge it first, e.g. "Thank you for providing a Data Availability section. Please …".
6. **Apply only this file and the journal-specific file.** A rule from another journal is never a reason to fail a manuscript.
7. **Do not edit the author files.** The check reads the files and reports on them; it never changes them.

---

## 4. The report

One Markdown file per manuscript and round, based on `aip-qc-report-template.md`. File name: `<MSID>_AIP-QC_round<N>.md`.

The report has a machine-readable header and four parts. Keep the heading texts exactly as in the template so the report can be parsed.

### 4.1 Header (YAML front matter)

```yaml
---
report: aip-qc
guideline_version: "1.0"
msid: EMBOR-2026-03700V2
journal: EMBOR                    # EMBOJ | EMBOR | EMM | MSB | LSA
journal_name: EMBO Reports
revision: TR                      # as in the MSID
round: 1
checked_by: <editor initials or agent id>
date: 2026-09-30
inputs:
  manuscript: ms_revised.docx
  system_metadata: available      # available | not available
  mmqc: ingested                  # ingested | not run
  previous_report: none           # file name or none
profile:                          # informational, never a status
  main_figures: 6
  supplementary_figures: 5        # EV + Appendix figures for EMBO Press
  tables: 1
  movies: 0
  datasets: 2
  character_count: 58400          # record for EMBO Press; optional at LSA
  results_discussion_combined: no # yes | no
counts:
  action: 7
  advise: 2
  manual: 2
  pass: 41
  na: 9
---
```

Below the YAML comes a one-line human summary: `**7 required actions · 2 recommendations · 2 to check manually**`.

### 4.2 Part 1: Action required

This is the only part most readers need. It is a single table containing **every `ACTION` and `ADVISE` check**:

- `ACTION` rows first, then `ADVISE` rows.
- Within each status, rows follow checklist order (A1 → H2).

| Column | Content |
|---|---|
| ID | Check ID (e.g. `D3`). |
| Check | Short check name, as in Section 5. |
| Status | `ACTION` or `ADVISE`. |
| Finding | For the editor: what is wrong and where. One or two short sentences, with locations. |
| Action for authors | Ready-to-send wording (standard wording, placeholders filled). This column feeds Part 3. |
| Resolution | Left empty by the checker. Editors append `>>> reply` (e.g. `>>> Done`, `>>> Editor, please check`) and close with `==>RESOLVED <date> <initials>`. |

If there are no `ACTION` or `ADVISE` items, replace the table with the line `No action required.`

Directly under the table come two lists:

- **Check manually.** One bullet per `MANUAL` check, formatted as `ID Check — reason (what could not be verified)`.
- **Notes for the editor.** Free-form observations that are not author actions, e.g. an apparent image duplication routed to the integrity check, or a request that needs an editorial decision. Write `None.` if empty.

### 4.3 Part 2: Full checklist

All checks, in checklist order, grouped under the area headings (A–H), one table per area:

| Column | Content |
|---|---|
| ID | Check ID |
| Check | Short name |
| Status | One of the five statuses |
| Note | ≤ 15 words |

What goes in the Note column depends on the status:

| Status | Note |
|---|---|
| `PASS` | Optional short evidence (e.g. "Headed 'Data Availability', after Methods"). |
| `ACTION` / `ADVISE` | Gist of the finding (the full text is in Part 1). |
| `N/A` | The reason (e.g. "No animal work", "Not a requirement of this journal"). |
| `MANUAL` | What is missing. |

Every check ID in Section 5 appears exactly once in Part 2, including journal-exclusive checks, which get `N/A`.

### 4.4 Part 3: Draft author letter

A plain list, one action per line, each line starting with `- `. It is made from the "Action for authors" column of Part 1, but **reordered into the letter groups below**. Within a group, follow checklist order.

| # | Letter group | Check IDs |
|---|---|---|
| 1 | Science and ethics | D5, D6, F1, F2, F3, F4, F5, F6 |
| 2 | Figure content | G9, G8, G7, G6, G5, G12, D8 |
| 3 | Data, code and source data | E2, E3, E4, E5, E6 |
| 4 | Files | A1, A2, A3, A4, A5, A6, A7, B4, F7, G11 |
| 5 | System metadata | B1, B2, B3, B5, B6, B7, B9, B10 |
| 6 | Structure and declarations | C1, C2, C3, C4, C5, C6, C7, C8, C9, D1, D2, D3, D4, D7, E1, H1, H2 |
| 7 | Legends and callouts | G3, G4, G10, G1, G2 |
| 8 | Closing line (always) | B8 |

The letter always ends with *"Please be sure that the authorship listing and order are correct and match between the system and the manuscript file."* This line appears even when B8 is `PASS`.

`MANUAL` items and editor notes never go into the letter.

### 4.5 Example (abridged)

```markdown
**4 required actions · 1 recommendation · 2 to check manually**

## Part 1 — Action required

| ID | Check | Status | Finding | Action for authors | Resolution |
|---|---|---|---|---|---|
| A3 | Figures as individual files | ACTION | Figs S1–S5 supplied as one merged PDF | Please upload all figure files as individual ones, including the supplementary figure files; all figure legends should only appear in the main manuscript file. | |
| D3 | Competing interests statement | ACTION | Headed "Competing interests" (p. 21) | Please rename "Competing interests" to "Conflict of Interest." | |
| F1 | Imaging details | ACTION | Confocal: objective N.A. and filters missing; live imaging (Fig 5) without temperature | In the Methods, please expand on the imaging details, including the microscope used, the objectives (type, magnification, N.A.), excitation and emission wavelengths/filters and, for live/time-lapse imaging, the temperature during acquisition. | |
| G7 | Scale bars | ACTION | mmQC: no scale bar in Fig 3D zoom-in; size not stated for Fig 2A | Please add scale bars to Figure 3D (including zoomed-in images), make them clearly visible and define their size in the legend for Figures 2A and 3D. | |
| E4 | Source data | ADVISE | No source data provided | We encourage you to provide source data (one file per main figure, with uncropped blots and numerical values) and to state their availability in the Data Availability section. | |

**Check manually**
- B7 Social media handles — system metadata not available
- G9 Image reuse / apparent duplication — possible duplication Fig 4C / S2B, see note

**Notes for the editor**
- Fig 4C (actin) and Fig S2B (actin) appear to show the same blot, undisclosed. Not yet raised with the authors (G9 left `MANUAL`) — please route to the integrity check first.
```

---

## 5. The checklist

Every check below has an ID, a name and five lines:

- **Applies:** which journals it covers.
- **Level:** Required, Required if applicable, or Recommended (Section 3.1).
- **Pass:** what a passing manuscript looks like. These are the EMBO Press rules. A journal file may override them.
- **Fails when:** the most common ways the check fails.
- **Wording:** the standard author wording. Placeholders are in `[BRACKETS]`.

Checks marked **⚑** are on the watch list (Section 7).

### A · Files and package

#### A1 · Editable, clean manuscript file ⚑
- **Applies:** All journals · **Level:** Required
- **Pass:** Main text is an editable .doc/.docx file with no tracked changes, highlighting or comments.
- **Fails when:** Only a PDF is supplied. Tracked changes or highlighted text remain, which is typical for revisions.
- **Wording:** "Please upload a clean, editable .docx manuscript file without tracked changes or highlighted text."

#### A2 · No figures embedded in the manuscript
- **Applies:** All journals · **Level:** Required
- **Pass:** No figures (main, supplementary or graphical abstract) inside the manuscript text; they exist only as separate uploads.
- **Fails when:** Figures are pasted into the text, often at the end or next to their legends.
- **Wording:** "Please remove the figures from the manuscript text and upload them only as separate files."

#### A3 · Each figure uploaded as its own file ⚑
- **Applies:** All journals · **Level:** Required
- **Pass:** Every main and EV figure is an individual, single-figure, production-quality file (TIFF, EPS or PDF; ≥300 ppi, ≥600 ppi for fine line art). No merged multi-figure PDF.
- **Fails when:** Figures are merged into one PDF.
- **Wording:** "Please upload all main and EV figures as individual, production-quality files (one figure per file); a merged PDF of several figures cannot be used for production."

#### A4 · Supplementary material packaged correctly
- **Applies:** All journals · **Level:** Required
- **Pass:** At most 5 Expanded View figures (Figure EV1–EV5). Remaining supplementary figures, text and small tables are in one Appendix PDF (see G11). Large tables, datasets and code are individual files (Dataset EV1, Table EV1 …). Excel files carry their legend on a separate tab; non-Excel datasets and code are zipped with a README.
- **Fails when:** More than 5 EV figures, or everything is in one supplementary document.
- **Wording:** "Please supply the supplementary material as Expanded View figures (max. 5), a single Appendix PDF and individual Dataset EV/Table EV files, following the Author Guidelines."

#### A5 · Tables editable and numbered
- **Applies:** All journals · **Level:** Required if applicable
- **Pass:** Tables are editable (.doc/.docx or .xls/.xlsx, not images), numbered consecutively with Arabic numerals, each with a brief title, and placed at the end of the manuscript or supplied as separate files. No shading or coloured font.
- **Fails when:** Tables are pasted as images, supplied as PDF, or numbered with Roman numerals.
- **Wording:** "Please upload your Tables in editable .doc or Excel format, numbered consecutively with Arabic numerals (1, 2, 3, 4); they can be included at the bottom of the main manuscript file or be sent as separate files."

#### A6 · Movies / videos
- **Applies:** All journals · **Level:** Required if applicable
- **Pass:** Each movie is zipped individually together with its legend as a .txt file, one ZIP per movie. Named Movie EV1, EV2 … in ZIP names and callouts. Movie legends removed from the manuscript.
- **Fails when:** Movies are bundled. Legends are still in the manuscript. Callouts are missing. Nomenclature is wrong.
- **Wording:** "Please upload each movie as an individual ZIP file containing the movie and its legend as a .txt file, named Movie EV1, EV2 …, and remove the movie legends from the manuscript."

### B · Submission system and metadata

*Most B checks need the submission-system metadata. Without it, check what the manuscript shows and set `MANUAL` for the rest (Section 3.2, rule 3).*

#### B1 · Title: length and consistency
- **Applies:** All journals · **Level:** Required
- **Pass:** Title ≤100 characters including spaces, with no non-standard abbreviations and no serial titles (e.g. "Part II"). Identical in the system and the manuscript.
- **Fails when:** The title was edited during revision in only one place, or is too long.
- **Wording:** "The titles in both the system and the manuscript file must be consistent with each other." / "Please shorten the title to a maximum of 100 characters including spaces."

#### B3 · Synopsis text / summary blurb
- **Applies:** All journals · **Level:** Required
- **Pass:** Synopsis supplied as separate text: a 2-sentence blurb (≤250 characters) plus 3–4 single-sentence bullet points on the key findings.
- **Fails when:** The text is missing or too long. Bullets are missing or there are more than 4.
- **Wording:** "Please provide the synopsis text: a 2-sentence blurb (max. 250 characters) and 3–4 bullet points summarising the key findings."

#### B4 · Synopsis image
- **Applies:** EMBO Press only · **Level:** Required
- **Pass:** PNG or JPG file (not PDF or TIFF) of exactly 550 × 300 pixels (width × height). The rule is identical for EMBOJ, EMBOR, EMM and MSB.
- **Fails when:** The image is a PDF or TIFF, or has the wrong dimensions (most often too tall or too large).
- **Wording:** "Please provide the synopsis image as a PNG or JPG file of exactly 550 × 300 pixels (width × height)."


#### B5 · Keywords
- **Applies:** All journals · **Level:** Required
- **Pass:** 4–5 general keywords on the abstract page.
- **Fails when:** Keywords are missing, too many, or not on the abstract page.
- **Wording:** "Please add 4–5 general keywords on the abstract page."

#### B6 · Subject category
- **Applies:** All journals · **Level:** Required
- **Pass:** At least one subject category selected in the system. This is visible only in the system or JATS XML, never in the manuscript, so without system metadata the status is `MANUAL`.
- **Fails when:** No category is selected.
- **Wording:** "Please add a Category for your manuscript in our system."

#### B8 · Author list and order match ⚑
- **Applies:** All journals · **Level:** Required
- **Pass:** Author names (spelling, diacritics, initials) and order identical in the manuscript and the system.
- **Fails when:** An author was added or removed, or names are spelled differently.
- **Wording:** "Please be sure that the authorship listing and order are correct and match between the system and the manuscript file." This line also closes every author letter (Section 4.4).

#### B9 · Corresponding authors and ORCID
- **Applies:** All journals · **Level:** Required
- **Pass:** Corresponding authors (including co-/secondary corresponding authors) are identical in the system and the manuscript. Each has an ORCID iD linked in the system and an institutional e-mail on the title page.
- **Fails when:** A co-corresponding author was added only in the manuscript. An ORCID iD is not linked. An e-mail is missing.
- **Wording:** "Please add the ORCID iD for all corresponding authors (including secondary corresponding authors) – they should have received instructions on how to do so." / "Please note that the corresponding authors must match between the system and the manuscript file."

#### B10 · Funding entries
- **Applies:** All journals · **Level:** Required
- **Pass:** Every funder and grant entered as a separate entry in the system (not in the Comments box). The system list matches the funders and grant numbers in the Acknowledgements, with no missing or extra entries.
- **Fails when:** Funders are listed in the Comments box. Lists are mismatched. Grant numbers differ.
- **Wording:** "Please enter each funder and grant as a separate entry in the submission system and make sure the list matches the funders and grant numbers acknowledged in the manuscript."

### C · Manuscript structure

#### C1 · Sections present and in correct order
- **Applies:** All journals · **Level:** Required
- **Pass:** Title page → Abstract (+ Keywords) → Introduction → Results → Discussion → Methods → Data Availability → Acknowledgements → Disclosure and Competing Interests Statement → References → Figure Legends → (Tables) → Expanded View Figure Legends. EMBO Molecular Medicine inserts The Paper Explained between the Abstract and the Introduction (see that journal's file, check C8).
- **Fails when:** Methods come before Results. Declarations are in the wrong order. Legends are not at the end. Sections are missing.
- **Wording:** "Please arrange the manuscript sections in the order given in the Author Guidelines."

#### C2 · Title page
- **Applies:** All journals · **Level:** Required
- **Pass:** Starts with the title, then authors, then affiliations. Affiliations are on the title page (not in footnotes) and matched to authors by superscript numbers. No duplicated title/author page. No "character count" line.
- **Fails when:** Affiliations are in footnotes, a cover page is duplicated, or a character-count line remains.
- **Wording:** "Please reorder the title page so that it starts with the title, followed by the list of authors and then their affiliations, and remove any duplicated title/author information." / "Please list the affiliations on the title page rather than in footnotes." / "Please remove the character count from the title page."

#### C3 · Abstract
- **Applies:** All journals · **Level:** Required
- **Pass:** Section headed "Abstract" (rename "Summary"). A single paragraph of ≤175 words, with no citations and no statements such as data availability.
- **Fails when:** The abstract is too long, has several paragraphs, or contains references or a data statement.
- **Wording:** "The Abstract should be a single paragraph not exceeding 175 words, without references or other statements (e.g., data availability)." *If headed "Summary", add:* "Please rename 'Summary' to 'Abstract'."

#### C4 · No abbreviations list
- **Applies:** All journals · **Level:** Required
- **Pass:** No separate abbreviations section; abbreviations defined in brackets at first mention.
- **Fails when:** An "Abbreviations" section is present.
- **Wording:** "Please remove the list of abbreviations and ensure that abbreviations are defined in brackets after their first mention in the text."

#### C5 · Discussion
- **Applies:** EMBO Press · **Level:** Recommended
- **Pass:** The Discussion does not repeat the Results, and speculation is clearly labelled. A combined "Results and Discussion" section is recorded in the header profile (`results_discussion_combined`) and is not a failure.
- **Fails when:** The Discussion largely restates the Results (`ADVISE`).
- **Wording:** "We encourage you to shorten the Discussion so that it does not repeat the Results and to label speculation clearly."

#### C6 · Results subheadings
- **Applies:** All journals · **Level:** Recommended
- **Pass:** Results divided by informative subheadings, in the order the data are presented.
- **Fails when:** Results have no subheadings, or they are purely generic (e.g. "Experiment 1").
- **Wording:** "We encourage you to make the sub-headings of the Results section more informative."

#### C7 · Methods heading
- **Applies:** All journals · **Level:** Required
- **Pass:** Headed "Methods".
- **Fails when:** The heading is "Materials and Methods" or another name.
- **Wording:** "Please rename 'Materials and Methods' to 'Methods'."

#### C9 · Language and consistency
- **Applies:** All journals · **Level:** Recommended
- **Pass:** No obvious spelling or grammar problems. Consistent number style and italics (*in vivo*, *in vitro*). Gene names italic, proteins roman.
- **Fails when:** Errors or inconsistencies are frequent. Report only clear, repeated problems, not isolated typos.
- **Wording:** "We recommend that you conduct a thorough spell and grammar check and be consistent in describing numbers and in the use of italics (e.g., in vivo, in vitro) throughout the text."

### D · Declarations and ethics

#### D1 · Author contributions ⚑
- **Applies:** All journals · **Level:** Required
- **Pass:** No free-text Author Contributions section in the manuscript; CRediT roles entered per author in the system.
- **Fails when:** The Author Contributions section is still in the manuscript.
- **Wording:** "Please remove the Author Contributions section from the manuscript; contributions are captured through the CRediT entries in the submission system."

#### D2 · Contributions qualify for authorship
- **Applies:** All journals · **Level:** Required
- **Pass:** Each author's system CRediT roles meet authorship criteria.
- **Fails when:** An author has only a minor role, e.g. a single "investigation" or "validation" role and no part in writing or reviewing. Query such authors; do not decide for them.
- **Wording:** "The contributions selected for [AUTHOR NAME(S)] do not qualify them for authorship. Please either update the contributions in our system, or let us know if the author needs to be removed (and added eventually to the Acknowledgements section)."

#### D3 · Competing interests statement ⚑
- **Applies:** All journals · **Level:** Required
- **Pass:** Headed "Disclosure and Competing Interests Statement" and placed after the Acknowledgements.
- **Fails when:**
  - The statement is missing.
  - The heading is wrong. Old or other names to rename include "Conflict of Interest", "Competing interests", "Ethics declarations" and "Author Declaration".
  - The statement is correctly named but in the wrong place.
- **Wording:** "Please rename "[CURRENT HEADING]" to "Disclosure and Competing Interests Statement" and place it after the Acknowledgements."

#### D4 · Acknowledgements and funding in the text
- **Applies:** All journals · **Level:** Required
- **Pass:** An Acknowledgements section at the end of the text (not footnotes) that names funders in full with grant numbers. No separate "Funding" heading. No dedications.
- **Fails when:** The section is missing, funding is in a separate section or footnote, or grant numbers are missing.
- **Wording:** "Please add an Acknowledgements section to the main manuscript text, including all funders and grant numbers." / "Please move the funding information into the Acknowledgements section."

#### D5 · Animal ethics statement
- **Applies:** All journals · **Level:** Required if applicable (any work with vertebrates or regulated invertebrates)
- **Pass:** A statement that all animal experiments were performed in accordance with relevant guidelines and regulations. It names the approving institutional and/or licensing committee (with its institution), sits in the relevant Methods section (e.g. at the start of "Animals"), and covers all animal work.
- **Fails when:** No committee is named, the institution is missing, or the statement covers only some experiments.
- **Wording:** "Please confirm that all experiments with animals were performed in accordance with relevant guidelines and regulations. Please also include a statement identifying the institutional and/or licensing committee approving the experiments."

#### D6 · Human subjects ethics statement
- **Applies:** All journals · **Level:** Required if applicable (human subjects or human samples)
- **Pass:**
  - The approving committee and its institution are named.
  - Informed consent was obtained from all subjects.
  - The work conforms to the WMA Declaration of Helsinki and the Department of Health and Human Services Belmont Report.
  - The statement covers all subjects, including healthy donors.
- **Fails when:** The institution is missing, consent is not stated, Helsinki/Belmont is not mentioned, or healthy donors are not covered.
- **Wording:** "We recommend that you also include the organisation/university this committee is affiliated with. Please also include a statement that informed consent was obtained from all subjects and that the experiments conformed to the principles set out in the WMA Declaration of Helsinki and the Department of Health and Human Services Belmont Report. Please confirm that the statement covers all human subjects, including healthy donors."

#### D7 · BioRender / AI-assisted graphics
- **Applies:** All journals · **Level:** Required if applicable
- **Pass:** No BioRender disclaimers in individual legends; a "Graphics" line in the Methods, e.g. "Figures X, Y were created with BioRender.com".
- **Fails when:** Disclaimers are scattered in legends.
- **Wording:** "Please remove the BioRender statements from the figure legends and add a 'Graphics' line to the Methods (e.g., 'Figures X, Y were created with BioRender.com')."

#### D8 · Third-party images credited
- **Applies:** All journals · **Level:** Required if applicable
- **Pass:** The source (and permission, where relevant) is stated for every photograph or image not produced by the authors.
- **Fails when:** Stock or reused photographs, maps or schematics are shown without a source.
- **Wording:** "In the legends, please provide a source for all photographs used in Figure [X]."

### E · Data availability

#### E1 · Data Availability section named and placed ⚑
- **Applies:** All journals · **Level:** Required
- **Pass:** A single section headed exactly "Data Availability", placed directly after the Methods and before the Acknowledgements. There are no separate "Lead Contact" or "Materials Availability" blocks and no data statement in the Abstract.
- **Fails when:**
  - The section is placed before the Methods or at the end.
  - It is headed "Data sharing", "Availability of data and materials" or similar.
  - It is split into several blocks.
- **Wording:** "Please provide a single 'Data Availability' section directly after the Methods and before the Acknowledgements." / "Please rename "[CURRENT HEADING]" to "Data Availability.""

#### E2 · Datasets deposited with accession, URL and reviewer access ⚑
- **Applies:** All journals · **Level:** Required if applicable (any large-scale data: omics, sequences, structures, proteomics, computational models)
- **Pass:**
  - Every dataset is in a recommended repository: GEO/ArrayExpress, PRIDE/PeptideAtlas, GenBank/ENA/DDBJ, PDB/EMDB; BioStudies, Dryad, Zenodo or Figshare for unstructured data.
  - For each dataset, the DAS gives the repository name, the accession/DOI and an explicit URL, plus a reviewer access token while the data are not public.
  - With several datasets, it is explicit which accession covers which dataset.
  - Accessions match those given elsewhere (e.g. Methods, reagents table).
  - "Available upon request" is accepted only with a description of the data, the reason, a contact and re-use conditions.
- **Fails when:** A URL or token is missing, a dataset is mentioned in the Methods but not deposited, or accessions are inconsistent. Scan the full text for accession patterns such as `PXD…`, `GSE…`, `PRJNA…`, `S-BSST…`, `E-MTAB-…`, `EMD-…` and PDB IDs.
- **Wording:** "Please deposit [DATASET] in a recommended public repository and include the repository name, accession ID and a direct URL (plus a reviewer access token if the data are not yet public) in the Data Availability section. Please state explicitly which accession covers each dataset."

#### E3 · Code and computational models available
- **Applies:** All journals · **Level:** Required if applicable
- **Pass:** Code needed to reproduce the analyses is listed in the DAS with repository name and identifier (e.g. GitHub plus Zenodo DOI). This includes machine-learning models, image-analysis pipelines such as CellProfiler, analysis scripts and models. Web tools are hosted at a stable location.
- **Fails when:** Custom code or pipelines are mentioned in the Methods but not deposited, or only "available on request".
- **Wording:** "Please extend the Data Availability section to include information (repository name and accession number) on relevant source code(s) such that readers can understand and replicate your findings."

#### E4 · Source data
- **Applies:** EMBO Press · **Level:** Required
- **Pass:** One zipped folder per figure, one file per panel, and a completed Source Data checklist. For large deposited datasets, accessibility is verified through the Data Availability section.
- **Fails when:** Source data are missing or not organised per figure.
- **Wording:** Main figures: one zipped folder per figure, one file per panel, plus the completed Source Data checklist. Expanded View figures, Appendix content and EV files: a single ZIP labelled "Source Data for Expanded View and Appendix", with one folder per figure or table. For large deposited datasets, accessibility is verified through the Data Availability section.

#### E5 · Data and resources cited
- **Applies:** All journals · **Level:** Required if applicable
- **Pass:** Public datasets and studies used (e.g. GWAS summary statistics, ChIP-seq peak sets, reanalysed GEO series) are cited in the Methods and the reference list. They are tagged `[DATASET]` in the reference list and cited as "Data ref:" in the text.
- **Fails when:** Reused datasets are mentioned only by accession, without a citation.
- **Wording:** "In the Methods, please cite all datasets and studies used (e.g., [DATASET]), including them in the reference list."

#### E6 · No "data not shown"
- **Applies:** All journals · **Level:** Required
- **Pass:** Neither "data not shown", "manuscript in preparation" nor "manuscript submitted" appears anywhere. Personal communications have written authorisation. Each instance is replaced by "unpublished observations (Name)" or by the data (e.g. an Appendix figure).
- **Fails when:** Any of these phrases is present. Search the full text, including legends and the Methods.
- **Wording:** "Please replace every 'data not shown' with either the data (e.g., as an Appendix figure) or 'unpublished observations (Name)'."

### F · Methods reporting

*The science-and-ethics checks (D5, D6, F1–F6, G8, G9) can be routed to the figure/integrity role if preferred. The report format is unchanged either way.*

#### F1 · Imaging details ⚑
- **Applies:** All journals · **Level:** Required if applicable
- **Pass:** For every imaging modality used, the Methods give:
  - the microscope;
  - the objectives (type, magnification, numerical aperture);
  - excitation/emission wavelengths or filters;
  - the camera/detector and acquisition software;
  - temperature (and medium) for live or time-lapse imaging;
  - how co-localisation and other image quantifications were calculated.
- **Fails when:** N.A. or filters are missing, the live-imaging temperature is missing, or quantification is not described.
- **Wording:** "In the Methods, please expand on the imaging details, including the microscope used, the objectives (type, magnification, N.A.), excitation and emission wavelengths/filters and, for live/time-lapse imaging, the temperature during acquisition."

#### F2 · Antibodies fully described ⚑
- **Applies:** All journals · **Level:** Required if applicable
- **Pass:** Every primary and secondary antibody is listed with source and catalogue number (or preparation, if non-commercial) and dilution or concentration **per application** (WB, IF, IHC, FACS …).
- **Fails when:** Secondary antibodies or dilutions are missing, a "home-made" antibody has no reference, or only WB dilutions are given.
- **Wording:** "Please provide details (source, catalogue number, dilution/concentration) for all primary and secondary antibodies used, for each application." / "Kindly provide antibody concentrations used for various approaches."

#### F3 · All procedures described or cited ⚑
- **Applies:** All journals · **Level:** Required
- **Pass:** Every experiment and resource is described in enough detail to be repeated, or cited. This includes:
  - cell, tissue and animal sources and preparation;
  - generation and validation of KO/KI lines (with primers and antibodies);
  - constructs, protein purification and RNA extraction/RNA-seq;
  - differentiation protocols and quantification methods.
- **Fails when:** Experiments shown in figures have no Methods entry, or cell sources are missing.
- **Wording:** "Please provide details on [METHOD] or provide a relevant citation."

#### F4 · Software and web resources named and cited
- **Applies:** All journals · **Level:** Required
- **Pass:** All software, servers and algorithms are named (with version) and cited.
- **Fails when:** An analysis is described without naming the tool, or tools are named without a citation.
- **Wording:** "Please name and cite the software/server used for [ANALYSIS] (line [X])."

#### F5 · Reagents and resources table
- **Applies:** All journals · **Level:** Required
- **Pass:** A Reagents and Tools table is supplied as a separate file using the Author Guidelines template and is removed from the manuscript body. Revised research papers use structured methods.
- **Fails when:** The table is still in the manuscript body.
- **Wording:** "Please supply the Reagents and Tools table as a separate file using the template and remove it from the manuscript body."

#### F6 · Statistics described in the Methods
- **Applies:** All journals · **Level:** Required
- **Pass:**
  - Statistical tests are named consistently in the Methods and legends, with a rationale for the choice.
  - Quantification and normalisation are described (e.g. blot densitometry).
  - Biological and technical replicates are distinguished.
- **Fails when:** The Methods and legends name different tests, there is no statistics section, or replicate type is unclear.
- **Wording:** "Please name the statistical tests consistently in the legends and Methods, give a rationale for the choice of tests and describe how quantification was performed."

#### F7 · Author checklist and point-by-point response
- **Applies:** EMBO Press only · **Level:** Required
- **Pass:**
  - The author checklist is uploaded. The manuscript ID is filled in, every corresponding author is listed, and every question is answered Yes or Not Applicable, with cross-references to manuscript sections rather than duplicated content.
  - For revisions, a point-by-point response is uploaded.
- **Fails when:** The checklist is missing or partly completed, or the point-by-point response is missing.
- **Wording:** "Please upload the completed author checklist (all questions answered, manuscript sections cross-referenced) and the point-by-point response."

### G · Figures and legends

*G4–G7 and G12 take their results from mmQC. See Section 6 for how to read mmQC output. The other G checks are done by the AIP check itself.*

#### G1 · Every figure and panel called out ⚑
- **Applies:** All journals · **Level:** Required
- **Pass:** Every main and supplementary panel, table, dataset and movie is cited in the text. Callouts appear in sequential order (Fig 1 before Fig 2; S6 before S7).
- **Fails when:** Supplementary panels are never cited, callouts are out of order, or tables or movies are uncited.
- **Wording:** "Please add callouts for Figures [LIST] to your main manuscript text." / "Please make sure the figures are called out in sequential order."

#### G2 · Callouts match actual panels
- **Applies:** All journals · **Level:** Required
- **Pass:** No callouts to panels or figures that do not exist. Single-panel figures carry no "A" label (in figure, legend or callout).
- **Fails when:** Callouts point to missing panels, often after panels were removed in revision. A single-panel figure is labelled "A".
- **Wording:** "There is a callout for Figure [X], and this figure doesn't have panel [Y]; please correct." / "Figure [X] has only one panel; therefore, please remove the label A from the current figure, its legend, and call-outs in the manuscript text."

#### G3 · Legends placed correctly ⚑
- **Applies:** All journals · **Level:** Required
- **Pass:** Every figure has a legend, and there is no separate legend file. Figure Legends follow the References, then Expanded View Figure Legends. Dataset EV / Table EV legends are on a tab inside each Excel file. Movie legends are in the .txt inside each movie ZIP. Appendix legends are in the Appendix.
- **Fails when:** Supplementary legends are in a separate document, or a legend is missing.
- **Wording:** "Please add the figure and Expanded View figure legends to the main manuscript text after the References, and place the Dataset EV legends on a separate tab of each Excel file."

#### G4 · Legend structure (mmQC-assisted)
- **Applies:** All journals · **Level:** Required
- **Pass:** Each legend has a heading (title). Panels are described in alphabetical order (no F before E) and match what the figure shows. The legend does not simply repeat the Results.
- **Source:** AIP check for heading and order. mmQC `panel-image-matches-caption` for the match between panels and caption.
- **Fails when:** A legend has no title, panels are out of order, or panels are described but absent (or present but not described).
- **Wording:** "Please revise the legends for Figures [X] so that each has a heading and the panels are introduced in alphabetical order and match the figure."

#### G5 · Statistics in legends (mmQC)
- **Applies:** All journals · **Level:** Required if applicable (any quantitative panel)
- **Pass:** For every quantitative panel:
  - n is defined, including what it represents;
  - the statistical test is named (consistent with F6);
  - exact P values are given, or asterisk thresholds are defined;
  - error bars are defined (SD, SEM, 95% CI);
  - individual data points are shown where n is small.
- **Source:** mmQC `stat-test`, `stat-significance-level`, `error-bars-defined`, `individual-data-points`, `replication-reporting`. **Checked by AIP directly**, because mmQC does not cover them: box plots defined (centre, box bounds, whiskers, minima/maxima, percentiles), and a flag for any panel with n = 2.
- **Fails when:** Any of the above is missing for any quantitative panel.
- **Wording:** "In the legends of Figures [X], please define n, the statistical test, exact P values and what the error bars represent." Adjust the list to what is actually missing, per panel.

#### G6 · Axes, symbols and annotations defined (mmQC)
- **Applies:** All journals · **Level:** Required
- **Pass:**
  - All plot axes are labelled with units.
  - Axis breaks/gaps are marked.
  - Non-obvious axis titles are explained (e.g. %input).
  - Arrows, arrowheads, asterisks, dashed lines, colours, abbreviations and symbols (e.g. N/T) are defined in the legend.
- **Source:** mmQC `plot-axis-units`, `plot-gap-labeling`, `image-annotation-defined`.
- **Fails when:** Axes are unlabelled or have no units, or arrows or abbreviations are undefined.
- **Wording:** "Kindly label axes in Figure [X]." / "Please define [SYMBOL/ABBREVIATION] in the legend of Figure [X]."

#### G7 · Scale bars (mmQC) ⚑
- **Applies:** All journals · **Level:** Required if applicable (any micrograph)
- **Pass:** Every micrograph, including zoom-ins and insets, has a clearly visible scale bar. Its size is stated in the legend, and the legend says when one bar applies to several panels.
- **Source:** mmQC `micrograph-scale-bar`.
- **Fails when:** A zoom-in has no bar, or the size is not stated.
- **Wording:** "Please add scale bars to Figure [X] (including zoomed-in images), make them clearly visible and provide scale bar information in the legend."

#### G8 · Blots: markers and controls
- **Applies:** All journals · **Level:** Required if applicable (any blot or gel)
- **Pass:** Molecular-weight markers are shown on every blot and gel. Loading controls are on the same blot, or their origin is stated. Vertical splices are marked.
- **Fails when:** Markers are missing on cropped blots, or unmarked splices are visible.
- **Wording:** "Please include molecular weight markers for blots shown in Figures [X]."

#### G9 · Image reuse / apparent duplication
- **Applies:** All journals · **Level:** Required if applicable
- **Pass:** Any image reused in several panels or figures (e.g. the same loading control, a shared untreated condition) is disclosed in every relevant legend. No undisclosed apparent duplications are visible at QC level.
- **Fails when:** The same image appears in two places without disclosure. Report it to the authors as a query, not an accusation, and add a line to "Notes for the editor" so the integrity check picks it up.
- **Wording:** "Please confirm whether the images in Figure [X] and Figure [Y] are the same; if so, please indicate this in the respective legends, otherwise correct the figure."

#### G10 · Supplementary nomenclature
- **Applies:** All journals · **Level:** Required
- **Pass:** Supplementary items are named consistently in file names, in-figure titles, callouts and legends: "Figure EV1…" (never "Figure S1" for EV figures), "Appendix Figure S1" / "Appendix Table S1", "Dataset EV1" (data tables) vs "Table EV1" (small tables), "Movie EV1". EV numbering is contiguous from 1 (e.g. EV5, EV6, EV13 → EV1, EV2, EV3).
- **Fails when:** Names drift, e.g. "Figure S1" in file names but "Figure EV1" in callouts, or numbering has gaps.
- **Wording:** "Please use the EV nomenclature ('Figure EV1', not 'Figure S1') consistently in file names, figure titles, callouts and legends, numbered consecutively from 1."

#### G11 · Appendix file
- **Applies:** EMBO Press only · **Level:** Required if applicable
- **Pass:**
  - A single Appendix PDF.
  - The first page reads "Appendix for *<manuscript title>*", has a table of contents with page numbers, and has no author list.
  - Items are named "Appendix Figure Sn" / "Appendix Table Sn" throughout the Appendix and the callouts.
  - "Supplementary Materials and Methods" is relabelled "Appendix Materials and Methods", or moved into the main Methods.
- **Fails when:** There is no table of contents, the Appendix is supplied as Word, the author list is on page 1, or "Supplementary Figure" naming is used.
- **Wording:** "Please provide the Appendix as a single PDF titled 'Appendix for …' with a table of contents on the first page and 'Appendix Figure S1' nomenclature throughout."

#### G12 · Other mmQC figure checks (mmQC)
- **Applies:** All journals · **Level:** Required
- **Pass:** No failures in any mmQC check not mapped to G4–G7. This currently means `single-channel-for-overlay`: individual channels are shown for merged/overlay images where interpretation depends on them. Any mmQC check added in future also lands here until it is mapped.
- **Fails when:** mmQC reports a failure in one of these checks.
- **Wording:** "Please show the individual channels for the merged images in Figure [X]." For other checks, derive the wording from the mmQC finding.

### H · References

#### H1 · Reference format ⚑
- **Applies:** All journals · **Level:** Required
- **Pass:** The first 10 authors are listed, then "et al."; journal names are abbreviated and italic. Citations in the text are author–year. The list is alphabetical. Journal abbreviations follow Index Medicus.
- **Fails when:** More than 10 authors are listed before "et al." The list is not alphabetical. Numbered citations are used.
- **Wording:** "Please use the [10 author names, et al.] format in your references (i.e., limit the author names to the first 10)." If needed: "Please list the references in alphabetical order and cite them in author–year format in the text."

#### H2 · Single "References" list
- **Applies:** All journals · **Level:** Required
- **Pass:** One list headed "References" (not "Bibliography"). Supplementary references are merged into it. Preprints are labelled as such. Only published, accepted or preprint items are listed.
- **Fails when:** The list is headed "Bibliography", there is a separate supplementary reference list, or it contains "submitted" or "in preparation" entries.
- **Wording:** "Please rename "Bibliography" to "References." Please incorporate the supplementary references into the main references list."

---

## 6. Figure checks from mmQC

mmQC runs one benchmarked check per figure (image plus caption) and returns JSON that follows each check's `schema.json` (in `soda_mmqc/data/checklist/fig-checklist/<check>/`). Check names use hyphens in the repository; treat hyphens and underscores as equivalent.

### 6.1 Mapping

| mmQC check | AIP check |
|---|---|
| `panel-image-matches-caption` | G4 |
| `stat-test` | G5 |
| `stat-significance-level` | G5 |
| `error-bars-defined` | G5 |
| `individual-data-points` | G5 |
| `replication-reporting` | G5 |
| `plot-axis-units` | G6 |
| `plot-gap-labeling` | G6 |
| `image-annotation-defined` | G6 |
| `micrograph-scale-bar` | G7 |
| `single-channel-for-overlay` | G12 |
| any check not listed | G12 |

### 6.2 Turning mmQC output into a status

1. Read each check's output per figure and panel. Interpret the fields using the check's `schema.json`. A panel **fails** when an element that is required is present in the figure but not defined or not adequate. For example, in `error-bars-defined`, `error_bar_on_figure = yes` together with `error_bar_defined_in_caption = no` is a failure.
2. Aggregate per AIP check:
   - any failing panel in any mapped mmQC check → `ACTION`;
   - otherwise, all mapped checks returned and nothing failed → `PASS`;
   - no applicable panels (e.g. no micrographs for G7) → `N/A`.
3. In the finding, list the failing figures and panels and prefix with "mmQC:" (e.g. "mmQC: error bars undefined in Figs 2B, 3D").
4. Do not re-judge mmQC results. If a result looks clearly wrong, report it as mmQC says and add a line to "Notes for the editor".
5. **If mmQC was not run**, set G5, G6, G7 and G12 to `MANUAL` with the note "mmQC not run". For G4, check only the heading and panel order; if that passes, set `MANUAL` with the note "panel match needs mmQC". Do not substitute a partial, un-benchmarked version of the mmQC checks.

*Design note:* keeping mmQC separate avoids two versions of the same rule drifting apart and preserves mmQC's benchmarking. If the team later decides to fold the figure checks into AIP completely, only this section and the "Source" lines of G4–G7 and G12 need to change; the IDs and report format stay the same.

---

## 7. Watch list: frequent failures

Inspect these explicitly on every EMBO Press manuscript. Life Science Alliance frequencies are in `life_science_alliance.md`.

| ID | What typically goes wrong |
|---|---|
| E1 / E2 | Data Availability misplaced or misnamed; dataset URL or reviewer token missing. Most frequent EMBO Press failure |
| A3 | Figures merged into one PDF |
| D1 | Author Contributions section not removed from the manuscript |
| D3 | Competing-interests statement missing, wrongly named, or wrongly placed |
| G5 | Exact P values, test, n, or error bars not defined |
| G10 | "Figure S1" used instead of "Figure EV1" |
| B4 | Synopsis image as PDF/TIFF, or not exactly 550 × 300 px |
| B10 | Funders in the Comments box, or the system list does not match the Acknowledgements |
| B8 | Author list or order differs between the system and the manuscript |
| F1 | Objective N.A., filters, or live-imaging temperature missing |
| G7 | Scale bar missing on a zoom-in, or its size not stated |
| H1 | More than 10 authors before "et al." |

---

## 8. What the journal files change

The checks in Section 5 apply to all four EMBO Press journals. The journal file changes only the following.

| Journal | File | Change |
|---|---|---|
| The EMBO Journal | `the_embo_journal.md` | None. C8 is `N/A` |
| EMBO Reports | `embo_reports.md` | None, except a profile note for Report length and combined Results and Discussion. Not an `ACTION` in v1.1. C8 is `N/A` |
| Molecular Systems Biology | `molecular_systems_biology.md` | None. C8 is `N/A` |
| EMBO Molecular Medicine | `embo_molecular_medicine.md` | C8 is required. C1 inserts The Paper Explained between the Abstract and the Introduction |
| Life Science Alliance | `life_science_alliance.md` | Overrides the checks listed in that file. Not used for an EMBO Press manuscript |

---

## 9. Open points to confirm

These are editorial notes, not extra checks. Do not fail a manuscript on an open point, and do not apply a Life Science Alliance note to an EMBO Press manuscript. Until an item is decided, v1.1 applies the rule stated.
| # | Topic | Issue | v1.1 rule |
|---|---|---|---|
| 1 | EMBOJ and EMM guidelines | **Resolved.** Verified 30 Sep 2026; all four EMBO Press guideline pages share the same text | EMBO Press lines apply to all four; EMM lines only where the AIP pilot adds a rule (C8) |
| 2 | EMBO Press section order | All four journal pages: … Data availability → (Author contributions, replaced by CRediT) → Disclosure and competing interests statement → Acknowledgements → References. AIP pilot: Acknowledgements → Disclosure statement | Pilot order (Acknowledgements first) until decided |
| 3 | Synopsis bullets | **Resolved.** All journal pages: 3–4 | 3–4 (B3) |
| 4 | Synopsis image | **Resolved.** All journal pages: PNG/JPG, 550 × 300; the pilot's 300–600 px height and TIFF are no longer listed | 550 × 300, PNG/JPG, all EMBO Press (B4) |
| 5 | Dataset EV / Table EV legends | Journal pages: each file in its own ZIP with a plain-text README (title and description). AIP pilot: legend on a separate tab inside each Excel file | Pilot rule (A4, G3) until decided |
| 6 | The Paper Explained (EMM) | Required by the AIP pilot; not mentioned on the EMM guidelines page | Required (C8) |
| 7 | Keywords (EMBO Press) | 4–5 keywords on the abstract page per AIP pilot; not mentioned on any EMBO Press guidelines page | Pilot rule (B5) |
| 8 | "Data not shown" (EMBO Press) | Journal pages: data must be shown in main or EV figures. AIP pilot also allows "unpublished observations (Name)" | Pilot rule (E6) |
| 9 | LSA summary blurb | One source lists it in the manuscript section order, another only in the system | System only (B3); not part of C1 |
| 10 | Science & ethics ownership | D5, D6, F1–F6, G8, G9: AIP editor or figure/integrity role? | Checked in AIP; routable |
| 11 | mmQC delegation | G4–G7, G12 taken from mmQC rather than checked in AIP | Delegated (Section 6) |
| 12 | Character count / combined R&D | Journal pages: Reports have Results and Discussion combined, ≤25,000 characters (excluding spaces, Methods, legends, references), ≤5 figures or tables. No Report type listed for EMBOJ | Recorded in header profile only; decide whether to make it a check |
| 13 | Introduction subheadings (new) | All EMBO Press pages: Introduction without subheadings. Not yet a check | Not checked; decide whether to add to C6 |
| 14 | Layered figure files (new) | All EMBO Press pages: TIFF/Photoshop figures with text and arrows on separate layers. Not yet part of A3 | Not checked; decide whether to add to A3 |
| 15 | Running title, social handles at EMBO Press | Only documented for LSA | LSA only |
| 16 | EMBO Press author letter | Letter format and closing line documented only for LSA | Same format and closing line for all journals |
| 17 | DOIs in references (EMBO Press) | AIP pilot mentions removing DOIs "where journal style requires"; journal pages ask for the DOI (or "in press") for accepted papers | Not checked |
| 18 | LSA frequencies | 3 LSA example reports (03669-TRR, 03673-TRR, 03812-T) not yet analysed | Update watch list when added |