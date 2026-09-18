"""
translational_triangle

This package implements the "translational triangle" framework
(Weber et al.) to measure how translational a body of biomedical
research is — that is, how far it has moved away from purely
preclinical work (animal models, cell/molecular work) toward research
that directly involves human subjects.

The full pipeline, run from main.py, does the following:

  1. Starts from a list of Scopus identifiers (a researcher's list of
     publications) and resolves each one to its PubMed identifier
     (PMID), along with its MeSH descriptors, by querying the Scopus
     and PubMed/NCBI APIs (see scopus_to_pmid.py).
  2. Classifies each article into one or more of three categories — A
     (Animal), C (Cell), H (Human) — depending on which branches of the
     official MeSH descriptor tree its MeSH terms fall under (see
     mesh_classification.py). A single article can belong to several
     categories at once (e.g. "AH" for a study that combines an animal
     model with human patients).
  3. Places each article at a geometric position inside an equilateral
     triangle whose three vertices are A, C and H, and computes from
     that position a single summary number, the Translational Index
     (TI): how far the article moves, along that axis, from "purely
     preclinical" (Animal+Cell) toward "purely human" (see geometry.py).
  4. Traverses each article's forward citation network (using citation
     data from Scopus/PubMed) with a breadth-first search (BFS) to find
     out whether, and how quickly, that article ends up being cited by
     an article in the Human category — this captures translational
     impact over time, rather than looking only at the article's own
     MeSH content (see forward_citations.py). This stage is mandatory
     within the pipeline (it always runs right after classification,
     every time new articles are processed): the only way to skip it is
     to run the package in its dedicated "author summary" mode, which
     assumes this stage has already completed in a previous run and
     that its results are already saved in the CSV. This stage produces
     three further per-article metrics:
       - Translational Distance (TD): the minimum number of citation
         "generations" (hops) needed to reach an article in the H
         category starting from this one. It is left undefined (no
         path found) if the article is never cited, directly or
         indirectly, by any H-category article.
       - Translational Years (TY): the difference in years between this
         article's publication year and that of the H article it
         reached with the fewest hops (the same one that determines its
         TD).
       - Translational Closeness (TC): simply 1/TD, so that a shorter
         citation path gives a value closer to 1, and an article that
         never reaches H gets a TC of 0.
  5. Aggregates these per-article numbers up to the researcher level,
     highlighting above all the Translational Fraction (TF): the
     proportion of a researcher's articles that eventually reach some
     H-category article, no matter how long that citation path turned
     out to be (see metrics.py).

Taken together, TI describes an individual article's own translational
content, while TD/TY/TC/TF describe translational impact as it actually
propagates through the citation network and across a researcher's whole
body of work.
"""

__version__ = "0.1.0"
