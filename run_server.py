"""Server entrypoint for Vehicle Recognition FastAPI service."""

import logging

import uvicorn

if __name__ == "__main__":
    # Surface application logs (e.g. Groq image optimisation sizes) alongside uvicorn output.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
