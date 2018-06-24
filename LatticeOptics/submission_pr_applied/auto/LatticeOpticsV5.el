(TeX-add-style-hook
 "LatticeOpticsV5"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("revtex4-1" "twocolumn" "aps" "pra" "showpacs" "preprintnumbers" "bibnotes")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontenc" "T1")))
   (TeX-run-style-hooks
    "latex2e"
    "revtex4-1"
    "revtex4-110"
    "fontenc"
    "natbib"
    "graphicx"
    "bm"
    "color"
    "amsmath")
   (TeX-add-symbols
    '("unit" 2))
   (LaTeX-add-labels
    "fig:high_level"
    "fig:noises"
    "fig:optical_layout"
    "fig:berek"
    "fig:berek_step_response"
    "fig:circuits"
    "fig:pll"
    "fig:cut_pd"
    "fig:da_rin"
    "fig:bandwidth"
    "fig:low_pass_noises"
    "fig:averaged_mott"
    "fig:stability"
    "fig:long_term_stability")
   (LaTeX-add-bibliographies))
 :latex)

