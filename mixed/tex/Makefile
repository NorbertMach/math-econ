JOB = mixed
SRC = main.tex preamble.tex $(wildcard chapters/*.tex)

$(JOB).pdf: $(SRC)
	pdflatex -interaction=nonstopmode -file-line-error -jobname=$(JOB) main.tex
	pdflatex -interaction=nonstopmode -file-line-error -jobname=$(JOB) main.tex

clean:
	rm -f $(JOB).aux $(JOB).log $(JOB).out $(JOB).toc chapters/*.aux

.PHONY: clean
