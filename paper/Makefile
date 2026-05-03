# Build the PLA paper.
#   make          -> compile main.pdf with full bibliography pass
#   make quick    -> single pdflatex pass (no bib refresh)
#   make clean    -> remove build artifacts
#   make watch    -> rebuild on source change (requires latexmk)

MAIN = main
BIB  = references

.PHONY: all quick clean watch wordcount

all:
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex || true
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex || true
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	@echo
	@echo "Built $(MAIN).pdf. Pages: $$(pdfinfo $(MAIN).pdf | awk '/^Pages:/ {print $$2}' 2>/dev/null || echo '?')"

quick:
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	latexmk -pdf -pvc -view=none $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz \
	      $(MAIN).pdf

wordcount:
	@command -v texcount >/dev/null 2>&1 && \
	  texcount -1 -sum $(MAIN).tex || \
	  (echo "texcount not installed; falling back to wc -w"; \
	   detex $(MAIN).tex 2>/dev/null | wc -w)
