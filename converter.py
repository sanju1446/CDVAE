import json
from generate import generate

def predict_and_serialize(elements, n_samples=1, device="cpu"):
    results = generate(
        model,
        n_samples   = n_samples,
        device      = device,
        composition = [SYMBOL_TO_Z[e.capitalize()] for e in elements],
        temperature = 0.8,
    )

    output = []
    for r in results:
        output.append({
            "n_atoms"   : r["n_atoms"],
            "symbols"   : r["symbols"],
            "cartesian" : r["cartesian"].tolist(),   # numpy → list (JSON serializable)
            "lattice"   : r["lattice"].tolist(),      # numpy → list
            "properties": {
                "formation_energy" : round(r["energy"], 4),
                "e_above_hull"     : round(r["ehull"],  4),
                "band_gap"         : round(r["band_gap"], 4),
                "cell_a"           : r["abc"][0],
                "cell_b"           : r["abc"][1],
                "cell_c"           : r["abc"][2],
            }
        })

    return json.dumps(output)