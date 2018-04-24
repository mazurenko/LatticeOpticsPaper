__author__ = 'sine2k'

linestyles = {
        "-":    (None, None),
        "--":   (3., 2.),
        "--.":   (3., 2., 3., 2., 1., 2.,),
        "---.":   (3., 2., 3., 2., 3., 2., 1., 2.,),
        "----.":   (3., 2., 3., 2., 3., 2., 3., 2., 1., 2.,),
        "-.":   (3., 2., 1., 2.,),
        "-..":   (3., 2., 1., 2., 1., 2.,),
        "-...":   (3., 2., 1., 2., 1., 2., 1., 2.),
        "-....":   (3., 2., 1., 2., 1., 2., 1., 2., 1., 2.),
        ":":    (1., 2.),
        }

# colors
color_winered = '#a40000'
color_red_m = "#BA0000"
color_red_d = '#660821'#"#720000"
#color_red_m = "#EE2B2D"
#color_red_d = "#A31D21"
color_red_l = '#B38190'#'#9C4C4C'

color2 = '#5c3566'
color_grey = "#7A7A7A"
color_grey_d = '#353535'
color_grey_l = "#bfbfbf"
color_grey_ll = "#d4d4d4"
color_black = '#000000'

color_violet_d = '#5c3566'
color_violet_m = '#75507b'
color_violet_l = '#cd7fa8'

color_blue_d = '#174074'#'#204a87'
color_blue_m = '#3465a4'
color_blue_l = '#859DBA'#'#729fef'

color_teal_d = '#01665e'
color_teal_m = '#5ab4ac'
color_teal_l = '#80cdc1'

color_green_d = "#4e9a06"
color_green_m = "#73d216"

#color_brown_l = "#e9b96e"
color_brown_d = '#8c510a'#"#512E00" #
color_brown_m = '#d8b365'#"#774100" #
color_brown_l = '#dfc27d'#"#c4a000" #

color_set_r = '#9a6666'
color_set_r_l = '#ccb2b2'
color_set_g = '#669a66'
color_set_g_l = '#b2ccb2'
color_set_b = '#ae8553'
color_set_v = '#c1b1d2'
color_set_v_l = '#decbe4'

rcpars = {


    "font.serif": ["Helvetica Neue", "Bitstream Vera Sans"],
    #"font.serif": ["Helvetica Neue", "Bitstream Vera Sans"],
    #"font.serif": "Bembo Std",

    #"font.sans-serif": ["Helvetica", "URW Nimbus Sans", "Bitstream Vera Sans"],
    "font.monospace": ["Courier", "Bitstream Vera Sans"],
    "font.family": "serif",
    #"font.family": "Gentium",

    "pdf.fonttype": 42,  # correct cid subset, but problems with
                              # glyphs
    #"text.usetex": True, # uses better type1 fonts but but blows up
                              # file size by 10

    # activate this to use sans serif fonts in math mode in combination with text.usetex=true
    "text.latex.preamble": [r"\usepackage{sfmath}"],
    "text.usetex": True,

    #"mathtext.default": "regular",

    "mathtext.fontset": "custom",
    "mathtext.cal": "cursive",
    "mathtext.rm": "serif",
    "mathtext.tt": "monospace",
    "mathtext.it": "serif:oblique", #"serif:italic",
    "mathtext.bf": "serif:bold",
    "mathtext.sf": "serif",
    "mathtext.fallback_to_cm": True,

    "patch.linewidth": 0.5,

    #"figure.figsize": (fig_width*.99,fig_width*.7),
    # "figure.figsize": (8.3,4),

    "figure.subplot.left": .08,
    "figure.subplot.right": .98,
    "figure.subplot.bottom": .1,
    "figure.subplot.top": .98,
    "figure.subplot.wspace": .2,
    "figure.subplot.hspace": .2,


    #          'backend': 'ps',
    'axes.labelsize': 7,
    # 'axes.elinewidth': 0.5,
    'lines.markersize': 4,
    'axes.linewidth' : 0.5,
    'text.fontsize': 7,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,

    'lines.linewidth': 0.5,

    "grid.linewidth"   :   0.5,

    "xtick.major.size"     : 2,      # major tick size in points
    "xtick.minor.size"     : 1,      # minor tick size in points

    "ytick.major.size"     : 2,      # major tick size in points
    "ytick.minor.size"     : 1,      # minor tick size in points

    #'text.usetex': True,

#
#
# 	"font.size": 8,
#
# 	"font.family": "cmr10",
# #	"font.family": "Gentium",
# 	"pdf.fonttype": 42,  # correct cid subset, but problems with
#                               # glyphs
# #	"text.usetex": True, # uses better type1 fonts but but blows up
#                               # file size by 10
# 	"mathtext.fontset": "cm",
# #	"mathtext.cal": "cursive",
# #	"mathtext.rm": "serif",
# #	"mathtext.tt": "monospace",
# #	"mathtext.it": "serif:italic",
# #	"mathtext.bf": "serif:bold",
# #	"mathtext.sf": "sans",
# 	"mathtext.fallback_to_cm": True,
#     "patch.linewidth": 0.5,
# 	"lines.linewidth": 0.5,
#     "lines.markeredgewidth": 0.5,
#     "lines.markersize": 3,
# 	"axes.labelsize": 8,
# 	"axes.titlesize": 8,
# 	"axes.linewidth": 0.5,
# 	"text.fontsize": 8,
# 	"legend.fontsize": 8,
# 	"xtick.labelsize": 8,
# 	"ytick.labelsize": 8,
# 	"figure.subplot.left": .08,
# 	"figure.subplot.right": .98,
# 	"figure.subplot.bottom": .1,
# 	"figure.subplot.top": .98,
# 	"figure.subplot.wspace": .2,
# 	"figure.subplot.hspace": .2,
# 	"legend.borderpad": .2,
# 	"legend.handlelength": 2.5,
# 	"legend.handletextpad": .01,
# 	"legend.borderaxespad": .3,
# 	# "legend.labelsep": .002,
# 	"legend.labelspacing": .1,
#     "axes3d.grid": False,
}

