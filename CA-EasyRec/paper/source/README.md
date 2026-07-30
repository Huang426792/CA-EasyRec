# CA-EasyRec ACL Paper Draft

This package contains an English ACL-format course-paper draft:

- `paper.tex`: manuscript source
- `references.bib`: BibTeX references
- `acl.sty`: official ACL style file, unmodified
- `acl_natbib.bst`: official ACL bibliography style, unmodified

Compile with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

The manuscript intentionally uses an em dash (`--` in LaTeX) for every new
experimental value that has not yet been measured. The row labeled
`EasyRec-S (reported)` contains values reported by the EasyRec paper and must
not be described as a local reproduction.

Before submission:

1. Replace `email to be inserted` with the student's real email.
2. Run the EasyRec and CA-EasyRec experiments with five seeds.
3. Fill the pending cells with mean and standard deviation.
4. Add significance marks only after the stated paired tests.
5. Remove the boxed draft-status note.
6. Rename the final PDF to the required `studentID_name_NLPassignment.pdf`
   pattern.
