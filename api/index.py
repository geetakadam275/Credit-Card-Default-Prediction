from app import app

# Expose WSGI / ASGI entrypoints for all Vercel runtime variants
application = app
handler = app
