
import uvicorn
import nest_asyncio
from app.APIs import app


nest_asyncio.apply()
uvicorn.run(app, port=8000)
