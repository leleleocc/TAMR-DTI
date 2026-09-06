TAMR-DTI Supporting Information

Primary upload file
-------------------
TAMR-DTI_Supporting_Information.pdf

Contents
--------
1. Equation and notation guide for Section 3 (Methodology), including every
   labelled equation from the drug, protein, alignment, Mamba, BiIntention,
   prediction, and loss modules.
2. Standalone selectable-text versions of the manuscript tables.
3. High-resolution vector PDF versions of the seven manuscript figures.

LaTeX source
------------
supporting_information.tex is the entry point. It inputs
si_formula_guide.tex, si_tables.tex, and si_figures.tex. The figure sources
are read from figures/*.pdf.

ACS upload note
---------------
Upload the PDF as the Supporting Information file. If the submission system
requests separate graphics at revision, upload the PDF figure files in the
figures directory as graphics. The main manuscript should retain its figures,
tables, and equations at the points where they are discussed.

Local compilation
-----------------
Run `pdflatex supporting_information.tex` twice (or use latexmk/tectonic)
from this directory. A TeX engine with graphicx, booktabs, longtable,
ragged2e, xurl, and hyperref is required.
