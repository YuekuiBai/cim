# Makefile for LaTeX Beamer presentation
TEX = xelatex
TARGET = slides
SOURCES = $(TARGET).tex

.PHONY: all clean pdf view

all: pdf

pdf: $(TARGET).pdf

$(TARGET).pdf: $(SOURCES)
	$(TEX) $(SOURCES)
	bibtex $(TARGET) 2>/dev/null || true
	$(TEX) $(SOURCES)
	$(TEX) $(SOURCES)

view: $(TARGET).pdf
	xdg-open $(TARGET).pdf

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.bbl *.blg *.synctex.gz
	rm -f $(TARGET).pdf
