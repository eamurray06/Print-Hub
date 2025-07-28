"""
Flask-based serverless function for Vercel that generates simple 3 D models
from component images and optional datasheets.

This module defines a WSGI application compatible with Vercel's Python
runtime. It exposes two routes:

  * ``/`` (GET) – serves an HTML form allowing users to upload a component
    image and, optionally, a PDF datasheet and manual thickness override.
  * ``/upload`` (POST) – processes the submitted files, extracts a height
    value from the datasheet (or uses a user provided value or a default),
    generates a heightmap based on the image silhouette, renders an
    interactive 3 D Plotly surface, produces an OBJ file of the extruded
    mesh, and returns these to the client in a single HTML response.

The implementation copies utility functions from the standalone
``component_modeler_refined/app.py`` script so that all logic resides
within this file. This simplifies deployment as a serverless function on
Vercel. Dependencies are declared in ``requirements.txt``.
"""

from __future__ import annotations

import base64
import html
import io
import re
from typing import Optional

import cv2  # type: ignore
import fitz  # type: ignore  # PyMuPDF
import numpy as np
import plotly.graph_objects as go  # type: ignore
from flask import Flask, request, Response
from plotly.offline import plot  # type: ignore

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Utility functions (adapted from component_modeler_refined/app.py)
#
def extract_thickness_from_pdf(data: bytes) -> Optional[float]:
    """Attempt to extract a thickness or height value from a PDF.

    The function looks for text patterns containing the words "height" or
    "thickness" followed by a numerical value and an optional unit (mm, cm,
    m, in). The first match encountered is returned as a value in
    millimetres. If no suitable information is found, ``None`` is returned.

    Args:
        data: Raw PDF bytes.

    Returns:
        The extracted thickness in millimetres or ``None`` if no value
        could be determined.
    """
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return None
    try:
        text = "\n".join(page.get_text() or "" for page in doc)
    finally:
        doc.close()
    # Regular expression to capture phrases such as "height: 1.2 mm" or "Thickness 0.3mm"
    pattern = re.compile(
        r"(?i)(?:height|thickness|body\s+height|body\s+thickness)\s*[:=]?\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|m|in)?"
    )
    match = pattern.search(text)
    if not match:
        return None
    value_str, unit = match.groups()
    try:
        value = float(value_str)
    except Exception:
        return None
    if not unit:
        return value
    unit = unit.lower()
    if unit == "mm":
        return value
    if unit == "cm":
        return value * 10
    if unit == "m":
        return value * 1000
    if unit == "in":
        return value * 25.4
    return None


def generate_3d_div(image_bytes: bytes, thickness: float = 1.0) -> str:
    """Generate an HTML ``div`` containing a Plotly surface from an image.

    The image is converted to a grayscale mask: any non zero pixel is
    considered part of the component. The mask is multiplied by the
    specified thickness to form a heightmap. The resulting surface is
    rendered into a Plotly figure and then converted into an embeddable
    ``<div>``.
    Args:
        image_bytes: Raw bytes of the uploaded image file.
        thickness: Height (in arbitrary units) assigned to pixels in the mask.

    Returns:
        A string containing a ``<div>`` with embedded Plotly figure.
    """
    np_data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Unsupported or corrupt image file.")
    _, mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    z = mask.astype(float) * thickness
    fig = go.Figure(data=[go.Surface(z=z, colorscale="Viridis", showscale=False)])
    fig.update_layout(
        title="Generated 3 D Model",
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="Thickness",
            aspectmode="data",
        ),
    )
    div = plot(fig, include_plotlyjs=True, output_type="div")
    return div


def generate_obj_from_image(image_bytes: bytes, thickness: float = 1.0) -> bytes:
    """Generate a simple extruded OBJ model from a binary image mask.

    The procedure finds the largest contour in the image and extrudes it along
    the z axis to the given thickness. The resulting mesh comprises two
    parallel faces (top and bottom) and rectangular side faces. All faces are
    triangulated. Coordinates are in the image pixel units.

    Args:
        image_bytes: Raw bytes of the uploaded image file.
        thickness: Extrusion height in the same units as the input thickness.

    Returns:
        A bytes object containing the OBJ file contents.
    """
    np_data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Unsupported or corrupt image file for OBJ generation.")
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image to generate OBJ.")
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    contour_pts = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
    n = len(contour_pts)
    if n < 3:
        raise ValueError("Contour too small for OBJ generation.")
    vertices = []
    for x, y in contour_pts:
        vertices.append((x, y, 0.0))
    for x, y in contour_pts:
        vertices.append((x, y, thickness))
    faces: list[tuple[int, int, int]] = []
    for i in range(1, n - 1):
        faces.append((1, i + 1, i + 2))
    for i in range(1, n - 1):
        faces.append((n + 1, n + i + 1, n + i + 2))
    for i in range(n):
        i_next = (i + 1) % n
        b_i = i + 1
        b_next = i_next + 1
        t_i = n + i + 1
        t_next = n + i_next + 1
        faces.append((b_i, b_next, t_next))
        faces.append((b_i, t_next, t_i))
    lines = []
    for v in vertices:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for f in faces:
        lines.append(f"f {f[0]} {f[1]} {f[2]}")
    obj_content = "\n".join(lines) + "\n"
    return obj_content.encode("utf-8")


# ---------------------------------------------------------------------------
# HTML templates
#

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>3 D Component Modeler</title>
    <style>
      body { font-family: sans-serif; background-color: #f7f9fc; margin: 0; padding: 0; }
      .container { max-width: 800px; margin: 40px auto; background: #fff; padding: 20px 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
      h1 { text-align: center; color: #333; }
      form { display: flex; flex-direction: column; gap: 1rem; }
      label { font-weight: bold; }
      input[type="file"], input[type="number"] { padding: 0.4rem; }
      input[type="submit"] { padding: 0.6rem 1.2rem; background-color: #007bff; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; }
      input[type="submit"]:hover { background-color: #0056b3; }
      .note { font-size: 0.9rem; color: #666; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>3 D Component Modeler</h1>
      <p class="note">Upload an image of your component and its datasheet. The application will attempt to derive a 3 D model by interpreting the image silhouette and the component height from the datasheet. If no height can be extracted, a default thickness of 5&nbsp;mm is used.</p>
      <form action="/upload" method="post" enctype="multipart/form-data">
        <label for="image">Component image:</label>
        <input type="file" id="image" name="image" accept="image/*" required>
        <label for="pdf">Datasheet (PDF, optional):</label>
        <input type="file" id="pdf" name="pdf" accept="application/pdf">
        <label for="manual_thickness">Manual thickness (mm, optional):</label>
        <input type="number" step="0.01" id="manual_thickness" name="manual_thickness" placeholder="e.g. 7.5">
        <input type="submit" value="Generate 3 D Model">
      </form>
    </div>
  </body>
</html>
"""


@app.get("/")
def index() -> Response:
    """Serve the upload form."""
    return Response(INDEX_HTML, mimetype="text/html")


@app.post("/upload")
def upload() -> Response:
    """Process the uploaded files and return the generated 3 D model."""
    # Ensure an image was uploaded
    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        return Response("No image uploaded", status=400)
    image_bytes = image_file.read()
    # Optional PDF datasheet
    pdf_file = request.files.get("pdf")
    pdf_bytes = pdf_file.read() if pdf_file and pdf_file.filename else None
    # Manual thickness override
    manual_thickness: Optional[float] = None
    manual_value = request.form.get("manual_thickness")
    if manual_value and manual_value.strip():
        try:
            manual_thickness = float(manual_value)
        except ValueError:
            manual_thickness = None
    # Determine thickness in mm
    thickness_mm: Optional[float] = None
    if manual_thickness is not None and manual_thickness > 0:
        thickness_mm = manual_thickness
    elif pdf_bytes:
        thickness_mm = extract_thickness_from_pdf(pdf_bytes)
    if thickness_mm is None or thickness_mm <= 0:
        thickness_mm = 5.0  # default thickness in mm
    # Rescale to arbitrary units for visualisation
    thickness_unit = thickness_mm / 2.0
    try:
        div = generate_3d_div(image_bytes, thickness=thickness_unit)
        obj_bytes = generate_obj_from_image(image_bytes, thickness=thickness_mm)
        obj_b64 = base64.b64encode(obj_bytes).decode("ascii")
        obj_href = f"data:model/obj;base64,{obj_b64}"
    except Exception as exc:
        error_html = f"""
            <html><body><h1>Processing Error</h1>
            <p>There was a problem processing the uploaded image: {html.escape(str(exc))}</p>
            <p><a href="/">Return to upload form</a></p>
            </body></html>
        """
        return Response(error_html, status=400, mimetype="text/html")
    result_html = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>Your 3 D Model</title>
    <style>
      body {{ font-family: sans-serif; background-color: #f7f9fc; margin: 0; padding: 0; }}
      .container {{ max-width: 900px; margin: 40px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
      h1 {{ text-align: center; color: #333; }}
      .model {{ margin-top: 20px; }}
      .back-link {{ margin-top: 20px; display: block; text-align: center; }}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Your 3 D Model</h1>
      <p>Estimated thickness: {thickness_mm:.2f}&nbsp;mm</p>
      <div class="model">{div}</div>
      <p><a href="{obj_href}" download="component_model.obj">Download OBJ file</a></p>
      <a class="back-link" href="/">&larr; Generate another model</a>
    </div>
  </body>
</html>
    """
    return Response(result_html, mimetype="text/html")


# In Vercel, the default export must be named "app" or "handler".
# Here we expose the Flask app as a module-level variable named "app".
