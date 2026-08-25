"""
Console entry point — installed as the `siteplan-backend` command.

Exists so end users (and the Rhino plugin's auto-start) never need to know
the uvicorn incantation: after `uv tool install`, "start the backend" is
one word on PATH. Keep this thin; anything real belongs in the app.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="siteplan-backend",
        description="Serve the Site Plan Drafter API (the Rhino plugin's "
                    "local backend).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8000,
                        help="port the Rhino plugin expects (default: 8000)")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("siteplan_backend.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
