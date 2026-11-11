import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import e57
import laspy
import numpy as np
import yaml
from flask import Flask, abort, flash, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(tempfile.gettempdir()) / "cloud2bim-web-results"
MAX_UPLOAD_SIZE = 10_000_000_000
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
JOB_TIMEOUT_SECONDS = 30 * 60
PROCESSING_LOCK = threading.Lock()

app = Flask(__name__)
app.config["SECRET_KEY"] = uuid.uuid4().hex
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE + MULTIPART_OVERHEAD_BYTES
app.logger.setLevel(logging.INFO)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for old_result in RESULTS_DIR.glob("*"):
    if old_result.is_file() and datetime.fromtimestamp(old_result.stat().st_mtime) < datetime.now() - timedelta(days=1):
        old_result.unlink()


def convert_scan_to_xyz(scan_path, xyz_path):
    scan = e57.read_points(str(scan_path))
    points = np.asarray(scan.points)
    colors = np.asarray(scan.color)

    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("The E57 file does not contain any 3D points.")
    if colors.shape != points.shape:
        colors = np.zeros(points.shape, dtype=points.dtype)

    with xyz_path.open("w", encoding="utf-8", newline="") as output:
        output.write("X\tY\tZ\tR\tG\tB\n")
        for start in range(0, len(points), 100_000):
            end = min(start + 100_000, len(points))
            chunk = np.column_stack((points[start:end], colors[start:end]))
            np.savetxt(output, chunk, delimiter="\t", fmt="%.6f")


def convert_las_to_xyz(scan_path, xyz_path):
    with laspy.open(str(scan_path)) as reader:
        if reader.header.point_count == 0:
            raise ValueError("The LAS file does not contain any points.")
        dimensions = {name.lower() for name in reader.header.point_format.dimension_names}
        has_rgb = {"red", "green", "blue"}.issubset(dimensions)
        with xyz_path.open("w", encoding="utf-8", newline="") as output:
            output.write("X\tY\tZ\tR\tG\tB\tIntensity\n")
            for records in reader.chunk_iterator(500_000):
                points = np.column_stack((records.x, records.y, records.z)).astype(np.float64)
                if has_rgb:
                    rgb = np.column_stack((records.red, records.green, records.blue)).astype(np.float64)
                    if np.max(rgb) > 255:
                        rgb *= 255.0 / 65535.0
                else:
                    rgb = np.zeros((len(points), 3), dtype=np.float64)
                intensity = (
                    np.asarray(records.intensity, dtype=np.float64)
                    if "intensity" in dimensions
                    else np.zeros(len(points), dtype=np.float64)
                )
                valid = np.isfinite(points).all(axis=1)
                values = np.column_stack((points[valid], rgb[valid], intensity[valid]))
                if len(values):
                    np.savetxt(output, values, delimiter="\t", fmt="%.6f")


def run_conversion(scan_path, job_dir, output_path):
    xyz_path = job_dir / "scan.xyz"
    config_path = job_dir / "config.yaml"
    log_path = job_dir / "pipeline.log"

    (job_dir / "images" / "pdf" / "space").mkdir(parents=True)
    (job_dir / "images" / "wall_outputs_images").mkdir(parents=True)
    if scan_path.suffix.lower() == ".e57":
        convert_scan_to_xyz(scan_path, xyz_path)
    elif scan_path.suffix.lower() == ".las":
        convert_las_to_xyz(scan_path, xyz_path)
    else:
        raise ValueError("Only .e57 and .las point-cloud files are supported.")
    config = yaml.safe_load((PROJECT_DIR / "config.yaml").read_text(encoding="utf-8"))
    config["e57_input"] = False
    config["xyz_files"] = [str(xyz_path)]
    config["output_ifc"] = str(output_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["CLOUD2BIM_HEADLESS"] = "1"
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "cloud2entities.py"), str(config_path)],
            cwd=job_dir,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=JOB_TIMEOUT_SECONDS,
            check=False,
        )

    if result.returncode != 0:
        details = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        app.logger.error("Cloud2BIM conversion failed:\n%s", details)
        raise RuntimeError("The point cloud could not be converted. Check the server console for details.")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("The conversion finished without creating an IFC file.")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/convert")
def convert():
    if not PROCESSING_LOCK.acquire(blocking=False):
        flash("Another scan is currently being processed. Please try again shortly.", "error")
        return render_template("index.html"), 429

    try:
        uploaded_file = request.files.get("scan")
        if uploaded_file is None or not uploaded_file.filename:
            flash("Choose an E57 or LAS point-cloud file to upload.", "error")
            return render_template("index.html"), 400
        file_extension = Path(uploaded_file.filename).suffix.lower()
        if file_extension not in {".e57", ".las"}:
            flash("Only .e57 and .las point-cloud files are supported.", "error")
            return render_template("index.html"), 400

        result_id = uuid.uuid4().hex
        result_path = RESULTS_DIR / f"{result_id}.ifc"
        report_path = RESULTS_DIR / f"{result_id}.json"
        download_name = f"{Path(secure_filename(uploaded_file.filename)).stem or 'cloud2bim'}.ifc"
        report_name = f"{Path(secure_filename(uploaded_file.filename)).stem or 'cloud2bim'}_interior_candidates.json"

        try:
            with tempfile.TemporaryDirectory(prefix="cloud2bim-job-") as job_name:
                job_dir = Path(job_name)
                scan_path = job_dir / f"upload{file_extension}"
                output_path = job_dir / "result.ifc"
                uploaded_file.save(scan_path)
                run_conversion(scan_path, job_dir, output_path)
                generated_report = job_dir / "interior_analysis" / "interior_candidates.json"
                if not generated_report.is_file():
                    raise RuntimeError("The conversion completed without creating the interior detection report.")
                shutil.copyfile(output_path, result_path)
                shutil.copyfile(generated_report, report_path)
                candidate_report = yaml.safe_load(
                    generated_report.read_text(encoding="utf-8")
                )
        except subprocess.TimeoutExpired:
            result_path.unlink(missing_ok=True)
            report_path.unlink(missing_ok=True)
            flash("The conversion took too long. Try a smaller point-cloud file.", "error")
            return render_template("index.html"), 408
        except (OSError, RuntimeError, ValueError, laspy.errors.LaspyException) as error:
            result_path.unlink(missing_ok=True)
            report_path.unlink(missing_ok=True)
            app.logger.exception("Unable to process uploaded point-cloud file")
            flash(str(error), "error")
            return render_template("index.html"), 400

        return render_template(
            "index.html",
            download_url=url_for("download", result_id=result_id, download_name=download_name),
            download_name=download_name,
            report_url=url_for("download_report", result_id=result_id, report_name=report_name),
            report_name=report_name,
            candidate_count=candidate_report.get("candidate_count", 0),
            furnishing_candidate_count=candidate_report.get("furnishing_candidate_count", 0),
        )
    finally:
        PROCESSING_LOCK.release()


@app.get("/download/<result_id>/<download_name>")
def download(result_id, download_name):
    try:
        parsed_id = uuid.UUID(hex=result_id)
    except ValueError:
        abort(404)

    result_path = RESULTS_DIR / f"{parsed_id.hex}.ifc"
    if not result_path.is_file():
        abort(404)

    safe_name = secure_filename(download_name)
    if not safe_name.lower().endswith(".ifc"):
        safe_name = f"{Path(safe_name).stem or 'cloud2bim'}.ifc"

    response = send_file(
        result_path,
        as_attachment=True,
        download_name=safe_name,
        mimetype="application/x-step",
        max_age=0,
    )
    response.call_on_close(lambda: result_path.unlink(missing_ok=True))
    return response


@app.get("/download-report/<result_id>/<report_name>")
def download_report(result_id, report_name):
    try:
        parsed_id = uuid.UUID(hex=result_id)
    except ValueError:
        abort(404)

    report_path = RESULTS_DIR / f"{parsed_id.hex}.json"
    if not report_path.is_file():
        abort(404)

    safe_name = secure_filename(report_name)
    if not safe_name.lower().endswith(".json"):
        safe_name = f"{Path(safe_name).stem or 'cloud2bim'}_interior_candidates.json"

    response = send_file(
        report_path,
        as_attachment=True,
        download_name=safe_name,
        mimetype="application/json",
        max_age=0,
    )
    response.call_on_close(lambda: report_path.unlink(missing_ok=True))
    return response


@app.errorhandler(413)
def upload_too_large(_error):
    flash("The upload is too large. The maximum file size is 10 GB.", "error")
    return render_template("index.html"), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
