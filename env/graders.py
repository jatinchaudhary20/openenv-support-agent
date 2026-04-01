def grade(score):
    if score <= 0:
        return 0.0
    if score >= 3:
        return 1.0
    return round(score / 3, 2)