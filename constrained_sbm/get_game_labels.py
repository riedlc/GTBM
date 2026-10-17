"""Game taxonomy and role labels for 2x2 payoff orderings (Riedl & Smead 2026)."""


def game_label(a, b, c, d, with_roles=False):
    """Return the game-theory label for payoff ordering (a, b, c, d).

    Rows (a vs b, c vs d):
        Coordination: a>b, c<d
        Conflict:     a<b, c>d
        Dilemma:      a<b, c<d
        Delight:      a>b, c>d
    Columns (a vs c, d vs b):
        Assortative:    a>c, d>b
        Disassortative: a<c, d<b
        X-Absorptive:   a>c, d<b
        Y-Absorptive:   a<c, d>b
    """
    # Row
    if   a > b and c < d: row = "Coordination"
    elif a < b and c > d: row = "Conflict"
    elif a < b and c < d: row = "Dilemma"
    elif a > b and c > d: row = "Delight"
    else:
        return "Undefined (tie in a/b or c/d)"

    # Column
    if   a > c and d > b: col = "Assortative"
    elif a < c and d < b: col = "Disassortative"
    elif a > c and d < b: col = "X-Absorptive"
    elif a < c and d > b: col = "Y-Absorptive"
    else:
        return "Undefined (tie in a/c or d/b)"

    labels = {
        ("Coordination", "Assortative"):    ("Coordination", "Standard / Alt"),
        ("Coordination", "X-Absorptive"):   ("Stag Hunt",     "Team / Solo"),
        ("Conflict",     "Disassortative"): ("Anti-Coord",    "Provider / Consumer"),
        ("Conflict",     "X-Absorptive"):   ("Hawk-Dove",     "Dove / Hawk"),
        ("Dilemma",      "X-Absorptive"):   ("Dilemma",       "Producer / Exploiter"),
        ("Delight",      "Assortative"):    ("Synergy",       "Mainstream / Niche"),
        ("Delight",      "Disassortative"): ("Mutualism",     "Host / Symbiont"),
        ("Delight",      "X-Absorptive"):   ("Harmony",       "Attractor / Satellite"),
        ("Delight",      "Y-Absorptive"):   ("Dissonance",    "Invasive / Native"),
    }

    entry = labels.get((row, col))
    if entry is None:
        return "Undefined (no game defined for this cell)"

    name, roles = entry
    return f"{name} ({roles})" if with_roles else name
