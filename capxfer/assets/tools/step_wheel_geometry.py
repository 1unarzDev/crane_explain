"""Extract mecanum wheel geometry from the vendor STEP (WS-H).

Run:  ~/micromamba/envs/crane-cad/bin/python -I assets/tools/step_wheel_geometry.py \
        "Mecanum Wheel/Mecanum.STEP" > assets/wheel_step_geometry.json
Uses pythonocc-core (OCC) only to read geometry; prints JSON (units: mm, mm^3, deg).
"""
import json, sys, math
import numpy as np
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.TDF import TDF_LabelSequence
from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TDataStd import TDataStd_Name

path = sys.argv[1]
doc = TDocStd_Document(TCollection_ExtendedString("doc"))
r = STEPCAFControl_Reader(); r.SetNameMode(True)
assert r.ReadFile(path) == 1
r.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool(doc.Main())

def name(lab):
    try:
        return lab.GetLabelName()
    except Exception:
        return "?"

leaves = []
def walk(lab, loc, path):
    if st.IsReference(lab):
        from OCC.Core.TDF import TDF_Label
        ref = TDF_Label(); st.GetReferredShape(lab, ref)
        walk(ref, loc.Multiplied(st.GetLocation(lab)), path + [name(lab)])
        return
    if st.IsAssembly(lab):
        comps = TDF_LabelSequence(); st.GetComponents(lab, comps)
        for i in range(comps.Length()):
            walk(comps.Value(i + 1), loc, path + [name(lab)])
        return
    shp = st.GetShape(lab).Moved(loc)
    leaves.append((name(lab), path, shp))

roots = TDF_LabelSequence(); st.GetFreeShapes(roots)
for i in range(roots.Length()):
    walk(roots.Value(i + 1), TopLoc_Location(), [])

parts = []
for nm, pth, shp in leaves:
    g = GProp_GProps(); brepgprop.VolumeProperties(shp, g)
    c = g.CentreOfMass(); m = g.MatrixOfInertia()
    I = np.array([[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    b = Bnd_Box(); brepbndlib.Add(shp, b, False); bb = b.Get()
    parts.append(dict(name=nm, path=pth, vol=g.Mass(), com=[c.X(), c.Y(), c.Z()], I=I.tolist(), bbox=list(bb)))

# wheel axis: assembly bbox / plate principal axis
allpts = np.array([p["bbox"] for p in parts])
lo = allpts[:, :3].min(0); hi = allpts[:, 3:].max(0)
ext = hi - lo
plates = [p for p in parts if "plate" in p["name"].lower()]
Ip = np.array(plates[0]["I"]); w, v = np.linalg.eigh(Ip)
# symmetric plate: the unique (largest) moment axis = wheel axis
axis = v[:, np.argmax(w)]; axis = axis / np.linalg.norm(axis)
centre = np.mean([p["com"] for p in plates], axis=0)

rollers = [p for p in parts if "satellite" in p["name"].lower()]
out_r = []
for p in rollers:
    I = np.array(p["I"]); w, v = np.linalg.eigh(I)
    # roller of revolution: unique (smallest) moment axis = roller axis
    d = np.abs(w[:, None] - w[None, :]); uniq = int(np.argmax(d.sum(1)))
    ra = v[:, uniq]
    c = np.array(p["com"]) - centre
    radial = c - axis * (c @ axis); rr = np.linalg.norm(radial); rhat = radial / rr
    # roller angle vs wheel axis
    ang = math.degrees(math.acos(min(1, abs(ra @ axis))))
    # tangential direction t = axis x rhat; sign of helix: (ra.axis)*(ra.t)
    t = np.cross(axis, rhat)
    out_r.append(dict(com_radius=rr, angle_to_axle_deg=ang, helix_sign=float(np.sign((ra @ axis) * (ra @ t))),
                      vol=p["vol"], I_eig=sorted(w.tolist()), axis_dir=ra.tolist()))

# roller dimensions in its own frame (use first roller, project bbox via its own axis using vertex sampling)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
def mesh_pts(shp):
    BRepMesh_IncrementalMesh(shp, 0.1)
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    pts = []
    ex = TopExp_Explorer(shp, TopAbs_FACE)
    while ex.More():
        f = topods.Face(ex.Current()); loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(f, loc)
        if tri is not None:
            tr = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                q = tri.Node(i).Transformed(tr); pts.append((q.X(), q.Y(), q.Z()))
        ex.Next()
    return np.array(pts)

leafmap = {id(l[2]): l for l in leaves}
rl = [l for l in leaves if "satellite" in l[0].lower()]
P = mesh_pts(rl[0][2]); ra = np.array(out_r[0]["axis_dir"]); c0 = np.array(rollers[0]["com"])
s = (P - c0) @ ra; perp = np.linalg.norm((P - c0) - np.outer(s, ra), axis=1)
roller_len = s.max() - s.min(); roller_dmax = 2 * perp.max()
# outer envelope radius about wheel axis (all rollers)
allR = []
for l in rl:
    Q = mesh_pts(l[2]) - centre
    allR.append(np.linalg.norm(Q - np.outer(Q @ axis, axis), axis=1).max())
wheel_od = 2 * max(allR); wheel_od_min = 2 * min(allR)
# width along axis
allP = np.vstack([mesh_pts(l[2]) for l in leaves]) - centre
ax_s = allP @ axis

res = dict(
    source=path, units="mm",
    n_leaf_parts=len(parts),
    part_counts={n: sum(1 for p in parts if p["name"] == n) for n in sorted({p["name"] for p in parts})},
    part_volumes_mm3={n: sum(p["vol"] for p in parts if p["name"] == n) for n in sorted({p["name"] for p in parts})},
    assembly_bbox_extent=ext.tolist(), wheel_axis=axis.tolist(), wheel_centre=centre.tolist(),
    wheel_outer_diameter_mm=wheel_od, wheel_outer_diameter_min_over_rollers_mm=wheel_od_min,
    wheel_width_mm=float(ax_s.max() - ax_s.min()),
    roller_count=len(rollers), roller_length_mm=float(roller_len), roller_max_diameter_mm=float(roller_dmax),
    rollers=out_r,
)
print(json.dumps(res, indent=1))
