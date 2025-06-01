from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import subprocess
from typing import Optional
from fastapi.staticfiles import StaticFiles
from companies import companies_list

companies_names = companies_list.keys()

app = FastAPI()
# Serve the 'output' folder at the '/output' path
app.mount("/output", StaticFiles(directory="output"), name="output")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    options_html = "".join(f'<option value="{company}">' for company in companies_names)
    return f"""
    <html>
    <head>
        <title>FastAPI with Autocomplete and Date Sliders</title>
        <script>
            function updateMinLabel(val) {{
                document.getElementById('minDateLabel').innerText = val;
                document.getElementById('maxDate').min = parseInt(val) + 1;
                if (parseInt(document.getElementById('maxDate').value) <= val) {{
                    document.getElementById('maxDate').value = parseInt(val) + 1;
                    updateMaxLabel(document.getElementById('maxDate').value);
                }}
            }}

            function updateMaxLabel(val) {{
                document.getElementById('maxDateLabel').innerText = val;
            }}
        </script>
    </head>
    <body>
        <h1>Enter company's name</h1>
        <form action="/submit" method="post">
            <label for="name">Name:</label>
            <input list="companies" id="name" name="name" required>
            <datalist id="companies">
                {options_html}
            </datalist>

            <br><br>

            <label for="frequency">Frequency:</label>
            <select name="frequency" id="frequency">
                <option value="D">Daily</option>
                <option value="Q">Quarterly</option>
                <option value="M">Monthly</option>
                <option value="Y">Yearly</option>
            </select>

            <br><br>

            <label for="minDate">Min Year: <span id="minDateLabel">1957</span></label><br>
            <input type="range" id="minDate" name="min_date" min="1957" max="2025" value="1957" step="1" oninput="updateMinLabel(this.value)">

            <br><br>

            <label for="maxDate">Max Year: <span id="maxDateLabel">2025</span></label><br>
            <input type="range" id="maxDate" name="max_date" min="1958" max="2025" value="2025" step="1" oninput="updateMaxLabel(this.value)">

            <br><br>

            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """

@app.post("/submit", response_class=HTMLResponse)
async def submit_name(
    name: str = Form(...),
    frequency: str = Form(...),
    min_date: str = Form(...),
    max_date: str = Form(...)
):
    try:
        result = subprocess.run(['python', 'script.py', name, frequency, min_date, max_date], capture_output=True, text=True)
        script_output = result.stdout.strip()
        error_output = result.stderr.strip()

        if error_output:
            return f"<h1>Error from script:<br><pre>{error_output}</pre></h1>"

        # Parse path from script output
        plot_path = None
        if "Plot saved to:" in script_output:
            plot_path = script_output.split("Plot saved to: ")[-1].strip()

        img_tag = f'<img src="/{plot_path}" alt="Plot">' if plot_path else "<p>No image generated.</p>"

        img_base64 = result.stdout.strip()

        return img_base64
    except Exception as e:
        return f"<h1>Unexpected error occurred:<br><pre>{str(e)}</pre></h1>"
