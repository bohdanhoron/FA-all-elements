def highpass_threshold(file_length, lmin, lmax, fmin1, fmin2):
    if lmin == lmax:
        f_min = fmin1  # If all files are the same length
    else:
        # Linear interpolation between fmin1 and fmin2
        f_min = fmin1 + (fmin2 - fmin1) * (file_length - lmin) / (lmax - lmin)
        f_min = round(f_min)  # Round to nearest integer"""

    return f_min
