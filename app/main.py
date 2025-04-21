from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>FastAPI on Render</title>
        </head>
        <body>
            <h1>Hello from FastAPI deployed on Render.com 🚀</h1>
        </body>
    </html>
    """