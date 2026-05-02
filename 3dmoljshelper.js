// assume `data` is one structure from your JSON array

function renderMolecule(data) {

    // 1. Build XYZ string from symbols + cartesian coords
    let xyz = `${data.n_atoms}\n\n`;
    data.symbols.forEach((sym, i) => {
        const [x, y, z] = data.cartesian[i];
        xyz += `${sym}  ${x.toFixed(4)}  ${y.toFixed(4)}  ${z.toFixed(4)}\n`;
    });

    // 2. Init 3Dmol viewer
    const viewer = $3Dmol.createViewer("viewer-div", { backgroundColor: "black" });

    // 3. Load the XYZ
    viewer.addModel(xyz, "xyz");

    // 4. Style atoms
    viewer.setStyle({}, { sphere: { scale: 0.4 }, stick: { radius: 0.15 } });

    // 5. Add unit cell box from lattice vectors
    const [a, b, c] = data.lattice;   // each is [x, y, z]
    const origin = { x: 0, y: 0, z: 0 };
    drawUnitCell(viewer, origin, a, b, c);

    viewer.zoomTo();
    viewer.render();

    // 6. Show properties
    displayProperties(data.properties);
}

function drawUnitCell(viewer, o, a, b, c) {
    // 8 corners of the parallelepiped
    const corners = [
        o,
        { x: a[0],           y: a[1],           z: a[2]           },
        { x: b[0],           y: b[1],           z: b[2]           },
        { x: c[0],           y: c[1],           z: c[2]           },
        { x: a[0]+b[0],      y: a[1]+b[1],      z: a[2]+b[2]      },
        { x: a[0]+c[0],      y: a[1]+c[1],      z: a[2]+c[2]      },
        { x: b[0]+c[0],      y: b[1]+c[1],      z: b[2]+c[2]      },
        { x: a[0]+b[0]+c[0], y: a[1]+b[1]+c[1], z: a[2]+b[2]+c[2] },
    ];
    // 12 edges
    const edges = [
        [0,1],[0,2],[0,3],[1,4],[1,5],[2,4],
        [2,6],[3,5],[3,6],[4,7],[5,7],[6,7]
    ];
    edges.forEach(([i, j]) => {
        viewer.addLine({ start: corners[i], end: corners[j],
                         color: "white", linewidth: 1.5 });
    });
}

function displayProperties(props) {
    document.getElementById("formation-energy").textContent =
        `${props.formation_energy} eV/atom`;
    document.getElementById("e-above-hull").textContent =
        `${props.e_above_hull} eV/atom`;
    document.getElementById("band-gap").textContent =
        `${props.band_gap} eV`;

    // Stability badge
    const stab = props.e_above_hull < 0.025 ? "Stable"
               : props.e_above_hull < 0.1   ? "Metastable"
               :                              "Unstable";
    document.getElementById("stability").textContent = stab;
}