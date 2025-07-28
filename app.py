"""
A very simple HTTP server that accepts an image and an optional PDF datasheet
from a web form. The server generates a rudimentary 3‑D representation of the
component by thresholding the uploaded image to create a binary mask and
interpreting the mask as a heightmap. If a datasheet is supplied, the
application attempts to extract a thickness value (in millimetres) from the
document; otherwise a default thickness is used. The resulting 3‑D model is
rendered on the results page using Plotly and delivered as an interactive
visualisation without requiring external dependencies.

The implementation avoids third‑party web frameworks such as Flask or
FastAPI (which are unavailable in this environment) by building directly on
Python's built‑in ``http.server`` and ``cgi`` modules. This keeps the
application self‑contained and compatible with the sandboxed environment.
"""

import os
import re
import io
import html
import cgi
import numpy as np
import cv2  # type: ignore
import fitz  # PyMuPDF
import plotly.graph_objects as go  # type: ignore
from plotly.offline import plot  # type: ignore
from http.server import SimpleHTTPRequestHandler, HTTPServer


# ---------------------------------------------------------------------------
# Utility functions
#
def extract_thickness_from_pdf(data: bytes) -> float | None:
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
    # Concatenate all text from the document
    try:
        text = "\n".join(page.get_text() or "" for page in doc)
    finally:
        doc.close()

    # Regular expression to capture phrases such as "height: 1.2 mm" or
    # "Thickness 0.3mm"
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
        # assume millimetres if no unit
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
    # Unknown unit
    return None


def generate_3d_div(image_bytes: bytes, thickness: float = 1.0) -> str:
    """Generate an HTML ``div`` containing a Plotly surface from an image.

    The image is converted to a grayscale mask: any non‑zero pixel is
    considered part of the component. The mask is multiplied by the
    specified thickness to form a heightmap. The resulting surface is
    rendered into a Plotly figure and then converted into an embeddable
    ``<div>``. The Plotly JavaScript library is embedded inline to avoid
    external dependencies.

    Args:
        image_bytes: Raw bytes of the uploaded image file.
        thickness: Height (in arbitrary units) assigned to pixels in the mask.

    Returns:
        A string containing a ``<div>`` with embedded Plotly figure.
    """
    # Decode the image into a NumPy array. `cv2.imdecode` can handle various
    # formats (PNG, JPEG, etc.).
    np_data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Unsupported or corrupt image file.")
    # Create a binary mask: treat non‑zero pixels as part of the component.
    # Using a threshold of 127 (half of 255) works for most images. If
    # thresholding fails (e.g. all pixels are similar), the mask will still be
    # binary but may be empty; the resulting model will be blank.
    _, mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    # Multiply by thickness to create a heightmap
    z = mask.astype(float) * thickness
    # Create the 3‑D surface. Plotly automatically maps x,y indices to the
    # grid. A simple colour scale is applied for clarity.
    fig = go.Figure(data=[
        go.Surface(z=z, colorscale="Viridis", showscale=False)
    ])
    fig.update_layout(
        title="Generated 3‑D Model",
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="Thickness",
            aspectmode="data"
        )
    )
    # Convert the Plotly figure to an HTML div. ``include_plotlyjs=True``
    # embeds the Plotly library inline. ``full_html=False`` returns only the
    # <div> containing the figure. This keeps the returned HTML small and
    # suitable for embedding into our own page.
    # ``full_html`` is not supported in the version of plotly shipped with this
    # environment. When ``output_type='div'``, Plotly returns only the
    # <div> containing the figure when ``include_plotlyjs`` is True. This
    # embeds the Plotly library inline and avoids external dependencies.
    div = plot(fig, include_plotlyjs=True, output_type='div')
    return div


def generate_obj_from_image(image_bytes: bytes, thickness: float = 1.0) -> bytes:
    """Generate a simple extruded OBJ model from a binary image mask.

    The procedure finds the largest contour in the image and extrudes it along
    the z‑axis to the given thickness. The resulting mesh comprises two
    parallel faces (top and bottom) and rectangular side faces. All faces are
    triangulated. Coordinates are in the image pixel units; you may scale
    them externally if dimensional accuracy is required.

    Args:
        image_bytes: Raw bytes of the uploaded image file.
        thickness: Extrusion height.

    Returns:
        A bytes object containing the OBJ file contents.
    """
    # Decode image
    np_data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Unsupported or corrupt image file for OBJ generation.")
    # Threshold to binary mask
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Find contours; use RETR_EXTERNAL to get outer boundaries only
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image to generate OBJ.")
    # Pick the largest contour by area
    largest = max(contours, key=cv2.contourArea)
    # Approximate the contour to reduce complexity; epsilon is a fraction of the arc length
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    # Flatten the contour array to a list of (x, y) pairs
    contour_pts = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
    n = len(contour_pts)
    if n < 3:
        raise ValueError("Contour too small for OBJ generation.")
    # Build vertices: bottom and top
    # OBJ indexing starts at 1
    vertices = []
    for x, y in contour_pts:
        vertices.append((x, y, 0.0))  # bottom
    for x, y in contour_pts:
        vertices.append((x, y, thickness))  # top
    # Triangulate top and bottom faces via fan triangulation
    faces = []  # each face is a tuple of vertex indices (1-based)
    # bottom face (reverse order to maintain outward normals)
    for i in range(1, n - 1):
        faces.append((1, i + 1, i + 2))
    # top face (offset by n, same orientation)
    for i in range(1, n - 1):
        faces.append((n + 1, n + i + 1, n + i + 2))
    # Side faces: connect consecutive vertices; wrap around at the end
    for i in range(n):
        i_next = (i + 1) % n
        # indices (1-based): bottom_i, bottom_next, top_next, top_i
        b_i = i + 1
        b_next = i_next + 1
        t_i = n + i + 1
        t_next = n + i_next + 1
        # Triangulate quad into two triangles
        faces.append((b_i, b_next, t_next))
        faces.append((b_i, t_next, t_i))
    # Compose OBJ content
    lines = []
    for v in vertices:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for f in faces:
        lines.append(f"f {f[0]} {f[1]} {f[2]}")
    obj_content = "\n".join(lines) + "\n"
    return obj_content.encode('utf-8')


# ---------------------------------------------------------------------------
# HTTP request handler
#
class ModelServerHandler(SimpleHTTPRequestHandler):
    """Request handler that serves an upload form and processes submissions.

    ``ModelServerHandler`` inherits from ``SimpleHTTPRequestHandler`` so that
    it can serve static files if needed. The root path (``/``) serves a
    simple HTML form. The ``/upload`` endpoint accepts a ``POST`` request
    containing an image and optional PDF, processes them, and returns an
    HTML page with the generated 3‑D model. All other paths fall back to
    the default static file handling implemented by the superclass.
    """

    INDEX_HTML = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>3‑D Component Modeler</title>
    <style>
      body {{ font-family: sans-serif; background-color: #f7f9fc; margin: 0; padding: 0; }}
      .container {{ max-width: 800px; margin: 40px auto; background: #fff; padding: 20px 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
      h1 {{ text-align: center; color: #333; }}
      form {{ display: flex; flex-direction: column; gap: 1rem; }}
      label {{ font-weight: bold; }}
      input[type="file"] {{ padding: 0.4rem; }}
      input[type="submit"] {{ padding: 0.6rem 1.2rem; background-color: #007bff; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; }}
      input[type="submit"]:hover {{ background-color: #0056b3; }}
      .note {{ font-size: 0.9rem; color: #666; }}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>3‑D Component Modeler</h1>
      <p class="note">Upload an image of your component and its datasheet. The application will attempt to derive a 3‑D model by interpreting the image silhouette and the component height from the datasheet. If no height can be extracted, a default thickness of 5&nbsp;mm is used.</p>
      <form action="/upload" method="post" enctype="multipart/form-data">
        <label for="image">Component image:</label>
        <input type="file" id="image" name="image" accept="image/*" required>
        <label for="pdf">Datasheet (PDF, optional):</label>
        <input type="file" id="pdf" name="pdf" accept="application/pdf">
        <label for="manual_thickness">Manual thickness (mm, optional):</label>
        <input type="number" step="0.01" id="manual_thickness" name="manual_thickness" placeholder="e.g. 7.5">
        <input type="submit" value="Generate 3‑D Model">
      </form>
    </div>
  </body>
</html>
    """

    def do_GET(self):
        """Serve the upload form or fall back to static files."""
        if self.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(self.INDEX_HTML.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(self.INDEX_HTML.encode('utf-8'))
        else:
            # Delegate to SimpleHTTPRequestHandler for static files
            super().do_GET()

    def do_POST(self):
        """Handle an upload request and respond with the generated model."""
        if self.path != '/upload':
            self.send_error(404, 'Not found')
            return
        # Parse the multipart form data using cgi.FieldStorage.
        # The FieldStorage constructor needs the environment and headers to
        # understand the request. The boundary and content length are passed
        # implicitly through the headers.
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                'REQUEST_METHOD': 'POST',
                'CONTENT_TYPE': self.headers.get('Content-Type'),
            },
            keep_blank_values=True,
        )
        # Retrieve the uploaded image
        image_field = form['image'] if 'image' in form else None
        if not image_field or not getattr(image_field, 'file', None):
            self.send_error(400, 'No image uploaded')
            return
        image_bytes = image_field.file.read()
        # Retrieve optional PDF
        pdf_bytes = None
        if 'pdf' in form and getattr(form['pdf'], 'file', None):
            pdf_bytes = form['pdf'].file.read()
        # Determine thickness in millimetres. Check manual override first.
        thickness_mm: float | None = None
        manual_thickness_value = None
        if 'manual_thickness' in form:
            # getfirst returns a string or None. Use ``strip`` to detect empty
            manual_value = form.getfirst('manual_thickness')
            if manual_value and manual_value.strip():
                try:
                    manual_thickness_value = float(manual_value)
                except ValueError:
                    manual_thickness_value = None
        if manual_thickness_value and manual_thickness_value > 0:
            thickness_mm = manual_thickness_value
        else:
            # Attempt to extract thickness from the PDF
            if pdf_bytes:
                thickness_mm = extract_thickness_from_pdf(pdf_bytes)
        # Fallback thickness in millimetres
        if not thickness_mm or thickness_mm <= 0:
            thickness_mm = 5.0
        # Normalise thickness value for the model. We rescale the physical
        # dimension to a reasonable z range for visualisation by dividing by
        # a constant factor. This prevents extremely tall models for large
        # components. Feel free to adjust this scaling factor; here we
        # compress millimetres to arbitrary units.
        thickness_unit = thickness_mm / 2.0
        try:
            div = generate_3d_div(image_bytes, thickness=thickness_unit)
            # Generate OBJ model and data URI for download
            obj_bytes = generate_obj_from_image(image_bytes, thickness=thickness_mm)
            import base64
            obj_b64 = base64.b64encode(obj_bytes).decode('ascii')
            obj_href = f"data:model/obj;base64,{obj_b64}"
        except Exception as exc:
            # Provide a helpful error message if the image fails to process.
            error_html = f"""
                <html><body><h1>Processing Error</h1>
                <p>There was a problem processing the uploaded image: {html.escape(str(exc))}</p>
                <p><a href="/">Return to upload form</a></p>
                </body></html>
            """
            self.send_response(400)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(error_html.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(error_html.encode('utf-8'))
            return
        # Compose the result page
        result_html = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>Your 3‑D Model</title>
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
      <h1>Your 3‑D Model</h1>
      <p>Estimated thickness: {thickness_mm:.2f}&nbsp;mm</p>
      <div class="model">{div}</div>
      <p><a href="{obj_href}" download="component_model.obj">Download OBJ file</a></p>
      <a class="back-link" href="/">&larr; Generate another model</a>
    </div>
  </body>
</html>
        """
        # Send the response
        encoded = result_html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_server(host: str = '0.0.0.0', port: int = 8000):
    """Run the HTTP server on the given host and port."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, ModelServerHandler)
    print(f"Serving on http://{host}:{port} ...")
    httpd.serve_forever()


if __name__ == '__main__':
    # When executed directly, start the server. In the Jupyter environment
    # we typically won't run this automatically, but it can be invoked via
    # ``python app.py`` from a terminal.
    run_server()