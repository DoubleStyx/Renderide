TEX = renderide_whitepaper

.PHONY: all clean distclean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex references.bib figures/process_topology.tex figures/frame_pipeline.tex figures/material_pipeline.tex
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX).tex
	bibtex $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX).tex

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).toc $(TEX).fdb_latexmk $(TEX).fls

distclean: clean
	rm -f $(TEX).pdf
