# Paper source — Forensic Science International: Digital Investigation

`main.tex` uses Elsevier's `elsarticle` class with the numbered citation style.

## Compiling

**Overleaf (no setup):** upload this directory. `elsarticle.cls` ships with
Overleaf, so nothing else is needed.

**Locally:** you need a TeX distribution with `elsarticle` (Debian/Ubuntu:
`texlive-publishers`), then

    latexmk -pdf main.tex

`elsarticle/` holds the distribution as downloaded from CTAN, including
`elsarticle.dtx`/`.ins` (the class is generated from these by `latex
elsarticle.ins`), the official templates and the `.bst` files. `elsarticle.cls`
itself is not checked in because every TeX distribution already provides it.

## Layout

    main.tex           front matter, abstract, keywords, \input of each section
    sections/          one file per section, 01--08
    references.bib     bibliography
    figures/           figures referenced by the sections
    elsarticle/        upstream CTAN distribution (reference only)

## Switching to the submission layout

The preprint option gives a readable single-column draft. For the journal's
own layout, change the class options to `\documentclass[final,5p,times,twocolumn]{elsarticle}`.
