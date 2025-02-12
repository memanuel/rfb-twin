import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

NumpyFloatArray = npt.NDArray[np.float64]

# *************************************************************************************************
# Colormap for the state of charge
cmap = mpl.colormaps['jet']

# Dimensions
Lx = 1280.0
Ly = 640.0
Lz = 160.0

# *************************************************************************************************
@dataclass 
class PovTextOpt:
    """Wrap the options for a POV-Ray text object"""
    font: str
    thickness: float
    offset: float
    rotate: str
    scale: float
    texture: str
    
# *************************************************************************************************
def make_text(text: str, loc: NumpyFloatArray, font: str, thickness: float, 
                  offset: float, rotate: str, scale: float, texture: str) -> str:
    """
    Create a POV-Ray text object
    INPUTS:
        text: the text to display
        loc: the location of the text as an array (x, y, z) in POV-Ray units
        font: the name of a true type font file
        thickness: the thickness of the text
        offset: the additional spacing between characters
        rotate: the rotation of the text as a povray string, e.g. "x*20"; default is the XY plane
        scale: the scaling factor for the text; default size is 1.0 unit
        texture: the texture of the text, e.g. "{pigment {color White} finish {reflection 0.0}}"
    """
    # Unpack the location
    x, y, z = loc
    # Build the string to make a POV-Ray text object
    pov: str = \
    f"""
    text {{
    ttf "{font}" "{text}" {thickness:0.6f} {offset:0.6f}
    rotate {rotate}
    scale {scale:0.6f}
    translate <{x:0.6f}, {y:0.6f}, {z:0.6f}>
    <texture>
    }}
    """
    # Format the texture block
    texture_sub: str = f"texture {texture}" if texture else ""
    pov = pov.replace("<texture>", texture_sub)
    parts = pov.split()
    return " ".join(parts)

# *************************************************************************************************
def make_text_union(texts: list[str], locs: list[NumpyFloatArray], text_opt: PovTextOpt) -> str:
    """
    Create a POV-Ray union of text objects
    INPUTS:
        texts: list of texts to display
        locs: the locations of the text with locs[i] an array (x, y, z) in POV-Ray units
        text_opt: the options for the text objects e.g. font, scale, texture
    """
    # Unpack the text options
    font: str = text_opt.font
    thickness: float = text_opt.thickness
    offset: float = text_opt.offset
    rotate: str = text_opt.rotate
    scale: float = text_opt.scale
    texture: str = text_opt.texture

    # Process list of POV-Ray text objects
    # use an empty string for texture because this is set below in the union
    pov_strs = []
    for text, loc in zip(texts, locs):
        pov_str = make_text(text=text, loc=loc, font=font, thickness=thickness, offset=offset, 
                            rotate=rotate, scale=scale, texture="")
        pov_strs.append(pov_str)

    # Join the text objects into a union
    text_joined: str = "    " + "\n    ".join(pov_strs)
    pov: str = \
    f"""
union {{
{text_joined}
    texture {texture}
}}
"""
    return pov

# *************************************************************************************************
def make_cylinder(q0: NumpyFloatArray, q1: NumpyFloatArray, r: float, v: float, 
                  rgb_func: callable, finish: str) -> str:
    """
    Make one cylinder
    INPUTS:
        q0: the starting point of the cylinder; array (x, y, z)
        q1: the ending point of the cylinder; array (x, y, z)
        r: the radius of the cylinder
        v: the value used to generate the color of the cylinder
        rgb_func: a function that returns the color of the cylinder based value v
        finish: a string with the finish of the cylinder
    """
    # Calculate the color of the cylinder based on the value
    cr, cg, cb = rgb_func(v)
    # Start and end points as povray string fragments
    p0: str = f"<{q0[0]:0.6f}, {q0[1]:0.6f}, {q0[2]:0.6f}>"
    p1: str = f"<{q1[0]:0.6f}, {q1[1]:0.6f}, {q1[2]:0.6f}>"
    # Texture of the cylinder
    texture: str = f"texture {{pigment {{ color <{cr:0.6f}, {cg:0.6f}, {cb:0.6f}>}} finish {finish} }}"
    # The completed cylinder
    return f"cylinder {{ {p0:s}, {p1:s}, {r:0.6f} {texture:s} }}"

# *************************************************************************************************
def make_cylinder_union(q0: NumpyFloatArray, q1: NumpyFloatArray, r: float, vs: NumpyFloatArray, 
                        rgb_func: callable, finish: str) -> str:
    """
    Make a sequence of cylinders
    INPUTS:
        q0: the starting point of the cylinder sequence; array (x, y, z)
        q1: the ending point of the cylinder sequence; array (x, y, z)
        r: the radius of the cylinder
        vs: array of values used to generate the color of the cylinder
        rgb_func: a function that returns the color of the cylinder based value v
        finish: a string with the finish of the cylinder
    """
    # Number of cylinders to build
    n: int = len(vs)
    # Array of shared start / end points
    qq: NumpyFloatArray = np.linspace(q0, q1, n+1)

    cyls: list[str] = []
    for j in range(n):
        cyl = make_cylinder(q0=qq[j], q1=qq[j+1], r=r, v=vs[j], rgb_func=rgb_func, finish=finish)
        cyls.append(cyl)

    cyls_joined: str = "    " + "\n    ".join(cyls)
    pov: str = \
    f"""
union {{
{cyls_joined}
}}
"""
    return pov

# *************************************************************************************************
def make_scalebar(vmin: float, vmax: float, rgb_func: callable, vals: NumpyFloatArray, 
                  labels: list[str], q0: NumpyFloatArray, q1: NumpyFloatArray, dq: float,
                  r: float, text_opt: PovTextOpt) -> None:
    """
    Generate a POV-Ray scalebar for a plot.
    INPUTS:
        vmin: the minimum value of the color scale
        vmax: the maximum value of the color scale
        rgb_func: a function that returns the color of the cylinder based value v
        vals: values used to place the labels
        labels: labels for the scalebar
        q0: the starting point of the scalebar; array (x, y, z)
        q1: the ending point of the scalebar; array (x, y, z)
        dq: offset from the scalebar to the labels
        r: the radius of the cylinder
        text_opt: the options for the text objects e.g. font, scale, texture
    """
    # The number of labels
    n: int = len(vals)
    # The number of cylinders - always 16 per label
    m: int = n * 16
    # The values for the cylinders
    vs: NumpyFloatArray = np.linspace(vmin, vmax, m)
    # Finish for the scalebar
    finish: str = "{reflection 0.0 specular 0.0 emission 1.0}"
    # The scalebar as a cylinder collection
    cyls_pov: str = make_cylinder_union(q0=q0, q1=q1, r=r, vs=vs, rgb_func=rgb_func, finish=finish)

    # The scalebar labels
    texts: list[str] = labels
    # Location of the scalebar labels
    ss: NumpyFloatArray = (vmax - vals) / (vmax - vmin)
    tt: NumpyFloatArray = (vals - vmin) / (vmax - vmin)
    locs: NumpyFloatArray = [ss[i] * q0 + tt[i] * q1 + dq for i in range(n)]

    # The scalebar labels as a union
    labels_pov: str = make_text_union(texts=texts, locs=locs, text_opt=text_opt)

    pov = f"{cyls_pov}\n{labels_pov}"
    return pov

# *************************************************************************************************
def make_scalebar_streamline():
    """Make scalebar for streamlines"""

    # Range of data values; run streamline_pov.py for the maximum speed
    vmin = 0.000
    vmax = 2.844

    # Size of scalebar
    r = 12.0
    # Offset from the scalebar to the labels
    dq = np.array([-16, 20, 0])

    # Set text options
    scale: float = 24.0
    font: str = "Micross.ttf"
    thickness: float = 0.02
    offset: float = 0.0
    rotate: str = "x*20"
    pigment: str = "{color White}"
    finish = "{reflection 0.0 specular 0.0 emission 1.0}"

    # Build a color function using the colormap set for the module
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    def rgb_func(v: float) -> tuple[float, float, float]:
        r, g, b, a = cmap(norm(v))
        return (r, g, b,)
    
    # Start and end points for scalebar
    qx0 = Lx * 0.25
    qx1 = Lx * 0.75
    qy = -Ly * 0.12
    qz =  Lz * 0.00
    q0 = np.array([qx0, qy, qz])
    q1 = np.array([qx1, qy, qz])

    # Sample values for scalebar
    vals = np.linspace(vmin, vmax, 5)
    labels = [f"{val:0.2f}" for val in vals]

    # Wrap the text options
    texture: str = f"{{ pigment {pigment:s} finish {finish:s} }}"
    text_opt: PovTextOpt = PovTextOpt(
        font=font, thickness=thickness, offset=offset, rotate=rotate, scale=scale, texture=texture)
   
    # Make the scalebar
    scale_pov = make_scalebar(
        vmin=vmin, vmax=vmax, rgb_func=rgb_func, vals=vals, labels=labels, q0=q0, q1=q1, 
        dq=dq, r=r, text_opt=text_opt)

    # Save the scalebar to a file
    path_out = Path('povray') / "stream_scalebar.pov"
    with open(path_out, "w") as fh:
        fh.write(scale_pov)

# *************************************************************************************************
def make_scalebar_soc():
    """Make scalebar for SOC plottend on streamlines"""

    # Range of data values; run soc_pov.py for the data range
    vmin = 0.000
    vmax = 0.500

    # Size of scalebar
    r = 12.0
    # Offset from the scalebar to the labels
    dq = np.array([-16, 20, 0])

    # Set text options
    scale: float = 24.0
    font: str = "Micross.ttf"
    thickness: float = 0.02
    offset: float = 0.0
    rotate: str = "x*20"
    pigment: str = "{color White}"
    finish = "{reflection 0.0 specular 0.0 emission 1.0}"

    # Build a color function using the colormap set for the module
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    def rgb_func(v: float) -> tuple[float, float, float]:
        r, g, b, a = cmap(norm(v))
        return (r, g, b,)
    
    # Start and end points for scalebar
    qx0 = Lx * 0.25
    qx1 = Lx * 0.75
    qy = -Ly * 0.12
    qz =  Lz * 0.00
    q0 = np.array([qx0, qy, qz])
    q1 = np.array([qx1, qy, qz])

    # Sample values for scalebar
    vals = np.linspace(vmin, vmax, 6)
    labels = [f"{val:0.2f}" for val in vals]

    # Wrap the text options
    texture: str = f"{{ pigment {pigment:s} finish {finish:s} }}"
    text_opt: PovTextOpt = PovTextOpt(
        font=font, thickness=thickness, offset=offset, rotate=rotate, scale=scale, texture=texture)
   
    # Make the scalebar
    scale_pov = make_scalebar(
        vmin=vmin, vmax=vmax, rgb_func=rgb_func, vals=vals, labels=labels, q0=q0, q1=q1, 
        dq=dq, r=r, text_opt=text_opt)

    # Save the scalebar to a file
    path_out = Path('povray') / "soc_scalebar.pov"
    with open(path_out, "w") as fh:
        fh.write(scale_pov)

# *************************************************************************************************
def make_scalebar_overpot():
    """Make scalebar for overpotential plottend on streamlines"""

    # Range of data values; run soc_pov.py for the data range
    vmin =  0.00
    vmax = 28.90

    # Size of scalebar
    r = 12.0
    # Offset from the scalebar to the labels
    dq = np.array([-24, 24, 0])

    # Set text options
    scale: float = 24.0
    font: str = "Micross.ttf"
    thickness: float = 0.02
    offset: float = 0.0
    rotate: str = "<20.0, 0.0, 2.0>"
    pigment: str = "{color White}"
    finish = "{reflection 0.0 specular 0.0 emission 1.0}"

    # Build a color function using the colormap set for the module
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    def rgb_func(v: float) -> tuple[float, float, float]:
        r, g, b, a = cmap(norm(v))
        return (r, g, b,)
    
    # Start and end points for scalebar
    qx0 = Lx * 0.25
    qx1 = Lx * 0.75
    qy  = -Ly * 0.12 
    qy0 = qy - 40
    qy1 = qy - 10 
    qz =  Lz * 0.00
    q0 = np.array([qx0, qy0, qz])
    q1 = np.array([qx1, qy1, qz])

    # Sample values for scalebar
    vals = np.linspace(vmin, vmax, 5)
    labels = [f"{val:0.2f}" for val in vals]

    # Wrap the text options
    texture: str = f"{{ pigment {pigment:s} finish {finish:s} }}"
    text_opt: PovTextOpt = PovTextOpt(
        font=font, thickness=thickness, offset=offset, rotate=rotate, scale=scale, texture=texture)
   
    # Make the scalebar
    scale_pov = make_scalebar(
        vmin=vmin, vmax=vmax, rgb_func=rgb_func, vals=vals, labels=labels, q0=q0, q1=q1, 
        dq=dq, r=r, text_opt=text_opt)

    # Save the scalebar to a file
    path_out = Path('povray') / "overpot_scalebar.pov"
    with open(path_out, "w") as fh:
        fh.write(scale_pov)

# *************************************************************************************************
def make_scalebar_curr_dens():
    """Make scalebar for current density plottend on streamlines"""

    # Range of data values; run soc_pov.py for the data range
    vmin =  0.00
    vmax = 22.50

    # Size of scalebar
    r = 12.0
    # Offset from the scalebar to the labels
    dq = np.array([-24, 24, 0])

    # Set text options
    scale: float = 24.0
    font: str = "Micross.ttf"
    thickness: float = 0.02
    offset: float = 0.0
    rotate: str = "<20.0, 0.0, 2.0>"
    pigment: str = "{color White}"
    finish = "{reflection 0.0 specular 0.0 emission 1.0}"

    # Build a color function using the colormap set for the module
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    def rgb_func(v: float) -> tuple[float, float, float]:
        r, g, b, a = cmap(norm(v))
        return (r, g, b,)
    
    # Start and end points for scalebar
    qx0 = Lx * 0.25
    qx1 = Lx * 0.75
    qy  = -Ly * 0.12 
    qy0 = qy - 40
    qy1 = qy - 10 
    qz =  Lz * 0.00
    q0 = np.array([qx0, qy0, qz])
    q1 = np.array([qx1, qy1, qz])

    # Sample values for scalebar
    vals = np.linspace(vmin, vmax, 5)
    labels = [f"{val:0.2f}" for val in vals]

    # Wrap the text options
    texture: str = f"{{ pigment {pigment:s} finish {finish:s} }}"
    text_opt: PovTextOpt = PovTextOpt(
        font=font, thickness=thickness, offset=offset, rotate=rotate, scale=scale, texture=texture)
   
    # Make the scalebar
    scale_pov = make_scalebar(
        vmin=vmin, vmax=vmax, rgb_func=rgb_func, vals=vals, labels=labels, q0=q0, q1=q1, 
        dq=dq, r=r, text_opt=text_opt)

    # Save the scalebar to a file
    path_out = Path('povray') / "curr_dens_scalebar.pov"
    with open(path_out, "w") as fh:
        fh.write(scale_pov)

# *************************************************************************************************
def main():
    # make_scalebar_streamline()
    # make_scalebar_soc()
    make_scalebar_overpot()
    make_scalebar_curr_dens()

# *************************************************************************************************
if __name__ == "__main__":
    main()
