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
            <title>FastAPI with Autocomplete</title>
        </head>
        <body>
            <h1>Enter company's name</h1>
            <form action="/submit" method="post">
                <label for="name">Name:</label>
                <input list="companies" id="name" name="name" required>
                <datalist id="companies">
                    {options_html}
                </datalist>
                <button type="submit">Submit</button>
            </form>
        </body>
    </html>
    """

@app.post("/submit", response_class=HTMLResponse)
async def submit_name(name: str = Form(...)):
    try:
        result = subprocess.run(['python', 'script.py', name], capture_output=True, text=True)
        script_output = result.stdout.strip()
        error_output = result.stderr

        # if error_output:
        #     return f"<h1>Error occurred: {error_output}</h1>"

        # Parse path from script output
        plot_path = None
        if "Plot saved to:" in script_output:
            plot_path = script_output.split("Plot saved to: ")[-1].strip()

        img_tag = f'<img src="/{plot_path}" alt="Plot">' if plot_path else "<p>No image generated.</p>"

        img_base64 = result.stdout.strip()

        return img_base64
        return f"""
        <html>
            <head><title>FastAPI with Plot</title></head>
            <body>
                <h1>Plot generated for {name}</h1>
                {img_tag}
            </body>
        </html>
        """
    except Exception as e:
        return f"<h1>Error occurred: {error_output}</h1>"
        # return f"<h1>Error: {str(e)}</h1>"
