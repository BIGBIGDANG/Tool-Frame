# Official RA-L Template Notes

This directory contains files downloaded from the official RA-L author
instructions and PaperCept support page.

Official RA-L author instructions consulted:

- https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/

Important formatting distinction from the RA-L website:

- **Initial submission / revised submission**: RA-L asks authors to use the RAS
  conference manuscript format:

  ```latex
  \documentclass[letterpaper, 10 pt, conference]{ieeeconf}
  ```

- **Final accepted RA-L version**: RA-L asks authors to reformat the accepted
  paper into IEEE journal style:

  ```latex
  \documentclass[letterpaper, 10 pt, journal, twoside]{IEEEtran}
  ```

The main official-template outline for advisor review is:

- `initial_submission/official_initial_toolframe_outline.tex`
- `initial_submission/build/official_initial_toolframe_outline.pdf`

The downloaded official files are:

- `downloads/ieeeconf.zip`
- `downloads/IEEEtranBST.zip`
- `downloads/IEEEtran_HOWTO.pdf`

The extracted initial-submission files include:

- `initial_submission/ieeeconf.cls`
- `initial_submission/root.tex`
- `initial_submission/root.pdf`
- `initial_submission/IEEEtran.bst`

The bilingual outline remains useful for advisor discussion, but it is not a
submission template because RA-L submissions should not contain Chinese
explanatory notes.
