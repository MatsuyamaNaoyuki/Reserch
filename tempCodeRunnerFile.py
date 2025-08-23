bigminlen = get_minimum(bigx)
remove_vals = [573810, 72194, 72450, 573556, 573850, 552812, 552543,9678, 9399, 510778, 511055,51529, 51218]
bigminlen = bigminlen[~np.isin(bigminlen, remove_vals)]
add_vals = [72608, 72271,573972, 573636, 552611, 552946, 9451, 9790, 510832, 511166, 51570, 51236]
bigminlen = np.append(bigminlen, add_vals)
bigminlen = np.sort(bigminlen)
diffs = np.diff(bigminlen)
