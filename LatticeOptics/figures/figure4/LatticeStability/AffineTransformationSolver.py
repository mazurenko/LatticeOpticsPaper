def affine_fit(from_pts, to_pts):
    """
    :param from_pts: initial points
    :param to_pts: transformed points
    :return:
    Fit an affine transformation to given point sets.
      More precisely: solve (least squares fit) matrix 'A'and 't' from
      'p ~= A*q+t', given vectors 'p' and 'q'.
      Works with arbitrary dimensional vectors (2d, 3d, 4d...).
    """


    q = from_pts
    p = to_pts
    if len(q) != len(p) or len(q)<1:
        print "from_pts and to_pts must be of same size."
        return False

    dim = len(q[0]) # num of dimensions
    if len(q) < dim:
        print "Too few points => under-determined system."
        return False

    # Make an empty (dim) x (dim+1) matrix and fill it
    c = [[0.0 for a in range(dim)] for i in range(dim+1)]
    for j in range(dim):
        for k in range(dim+1):
            for i in range(len(q)):
                qt = list(q[i]) + [1]
                c[k][j] += qt[k] * p[i][j]

    # Make an empty (dim+1) x (dim+1) matrix and fill it
    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim+1)]
    for qi in q:
        qt = list(qi) + [1]
        for i in range(dim+1):
            for j in range(dim+1):
                Q[i][j] += qt[i] * qt[j]

    # Ultra simple linear system solver. Replace this if you need speed.
    def gauss_jordan(m, eps = 1.0/(10**10)):
      """Puts given matrix (2D array) into the Reduced Row Echelon Form.
         Returns True if successful, False if 'm' is singular.
         NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
         Written by Jarno Elonen in April 2005, released into Public Domain"""
      (h, w) = (len(m), len(m[0]))
      for y in range(0,h):
        maxrow = y
        for y2 in range(y+1, h):    # Find max pivot
          if abs(m[y2][y]) > abs(m[maxrow][y]):
            maxrow = y2
        (m[y], m[maxrow]) = (m[maxrow], m[y])
        if abs(m[y][y]) <= eps:     # Singular?
          return False
        for y2 in range(y+1, h):    # Eliminate column y
          c = m[y2][y] / m[y][y]
          for x in range(y, w):
            m[y2][x] -= m[y][x] * c
      for y in range(h-1, 0-1, -1): # Backsubstitute
        c  = m[y][y]
        for y2 in range(0,y):
          for x in range(w-1, y-1, -1):
            m[y2][x] -=  m[y][x] * m[y2][y] / c
        m[y][y] /= c
        for x in range(h, w):       # Normalize row y
          m[y][x] /= c
      return True

    # Augement Q with c and solve Q * a' = c by Gauss-Jordan
    M = [Q[i] + c[i] for i in range(dim+1)]
    if not gauss_jordan(M):
        print "Error: singular matrix. Points are probably coplanar."
        return False

    # Make a result object
    class Transformation(object):
        """Result object that represents the transformation
           from affine fitter."""

        def __str__(self):
            res = ""
            for j in range(dim):
                str = "x%d' = " % j
                for i in range(dim):
                    str +="x%d * %f + " % (i, M[i][j+dim+1])
                str += "%f" % M[dim][j+dim+1]
                res += str + "\n"
            return res

        def Transform(self, pt):
            res = [0.0 for a in range(dim)]
            for j in range(dim):
                for i in range(dim):
                    res[j] += pt[i] * M[i][j+dim+1]
                res[j] += M[dim][j+dim+1]
            return res

    return Transformation()


if __name__ == "__main__":
    from_pt = ((1, 1), (1, 2), (2, 2), (2, 1))  # a 1x1 rectangle
    to_pt = ((4, 4), (6, 6), (8, 4), (6, 2))  # scaled x 2, rotated 45 degrees and translated

    trn = affine_fit(from_pt, to_pt)

    print "Transformation is:"
    print trn.To_Str()

    err = 0.0
    for i in range(len(from_pt)):
        fp = from_pt[i]
        tp = to_pt[i]
        t = trn.Transform(fp)
        print ("%s => %s ~= %s" % (fp, tuple(t), tp))
        err += ((tp[0] - t[0]) ** 2 + (tp[1] - t[1]) ** 2) ** 0.5

    print "Fitting error = %f" % err